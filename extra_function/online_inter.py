import os
import re
import argparse
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import MarkdownTextSplitter

# ================= 配置区域 =================
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "deepseek-ai/DeepSeek-R1"  # 按你硅基流动控制台的模型ID改

CHUNK_SIZE = 10000
CHUNK_OVERLAP = 0
MAX_TOKENS = 12000
TEMPERATURE = 0.1
# ===========================================


def read_api_key(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"API key file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        key = f.read().strip()
    if not key:
        raise ValueError(f"API key file is empty: {path}")
    return key


def setup_llm(api_key: str):
    return ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=api_key,
        openai_api_base=SILICONFLOW_BASE_URL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        # 如果接口不支持 extra_body 就删掉
        extra_body={"stop_token_ids": []},
    )


def clean_reasoning_content(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# --------- 关键：保护图片/链接等特殊Markdown结构 ---------

IMG_PATTERN = r"!\[[^\]]*?\]\([^)]+\)"  # 匹配标准Markdown图片：![alt](url)

def protect_images(text: str):
    """
    把Markdown图片语法替换成稳定占位符，避免模型改动/删除图片链接。
    返回 (protected_text, mapping)
    """
    mapping = {}
    counter = 0

    def repl(m: re.Match):
        nonlocal counter
        key = f"@@IMG_{counter:06d}@@"
        mapping[key] = m.group(0)
        counter += 1
        return key

    protected = re.sub(IMG_PATTERN, repl, text)
    return protected, mapping


def restore_images(text: str, mapping: dict):
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def count_images(text: str) -> int:
    return len(re.findall(IMG_PATTERN, text))


def get_translation_prompt(CATEGORY="finance"):
    system_prompt = f"""You are a professional {CATEGORY} and translator. Your task is to translate a geological report from English to Chinese.

STRICT REQUIREMENTS:
1. **Format Preservation**: You MUST preserve the original Markdown format exactly.
   - Do not translate content inside LaTeX formulas (e.g., $E=mc^2$, $$...$$).
   - Do not break Markdown tables. Keep the structure `| header |` unchanged, translate only the cell content.
   - Keep headers (#, ##), bold (**), and lists intact.

2. **Terminology**:
   - Use professional Chinese geological terminology.
   - For specific proper nouns (formation names, location names) that are rare, keep them in English or use format: "中文 (English)".

3. **Do NOT modify Markdown images/links**:
   - Keep ALL Markdown image and link syntax unchanged, including filenames and paths.
   - Examples that MUST remain EXACTLY unchanged:
     - ![caption](assets/page_001_img.png)
     - ![](assets/page_010_fig2.jpg)
     - [some text](assets/file.pdf)
   - Any token like @@IMG_000001@@ MUST remain EXACTLY unchanged.

4. **Output**:
   - Output ONLY the translated content.
   - Do NOT output explanations or notes like "Here is the translation".
"""
    user_prompt = "Original Text:\n{text}"

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])


def split_markdown(content: str):
    splitter = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.create_documents([content])


def build_output_path(input_path: Path, out_arg: str | None, suffix: str) -> Path:
    if out_arg:
        out_path = Path(out_arg).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    stem = input_path.stem
    ext = input_path.suffix or ""
    return input_path.parent / f"{stem}{suffix}{ext}"


def translate_chunk(chain, text: str, retry_if_mismatch: bool = True) -> str:
    """
    翻译单个chunk：保护图片 -> 翻译 -> 清理think -> 还原图片
    如果发现图片数量不一致，可选重试一次
    """
    protected, mapping = protect_images(text)

    # 第一次翻译
    response = chain.invoke({"text": protected})
    out = restore_images(clean_reasoning_content(response.content), mapping)

    if not retry_if_mismatch:
        return out

    in_cnt = count_images(text)
    out_cnt = count_images(out)
    if in_cnt != out_cnt:
        # 重试：把警告拼在输入前（不改chain结构，最简单有效）
        warning = (
            "CRITICAL: You must keep ALL Markdown image syntax unchanged. "
            "Do not delete or modify any image links.\n\n"
        )
        response2 = chain.invoke({"text": warning + protected})
        out2 = restore_images(clean_reasoning_content(response2.content), mapping)
        return out2

    return out


def main():
    parser = argparse.ArgumentParser(description="Translate markdown file (EN->ZH) using SiliconFlow/OpenAI-compatible API.")
    parser.add_argument("input_file", help="Input markdown file path")
    parser.add_argument("--out", default=None, help="Optional output file path. Default: same dir with suffix.")
    parser.add_argument("--suffix", default="_cn", help='Default output suffix, e.g. "_cn"')
    parser.add_argument("--apikey", default="data/api_key.txt", help='Default api key txt file')
    parser.add_argument("--category", default="finance", help='Specific topic of markdown, e.g. "finance"')
    parser.add_argument("--no-retry", action="store_true", help="Disable retry when image count mismatch")
    args = parser.parse_args()

    input_path = Path(args.input_file).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.is_dir():
        raise IsADirectoryError(f"Input path is a directory: {input_path}")

    output_path = build_output_path(input_path, args.out, args.suffix)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")

    llm = setup_llm(read_api_key(args.apikey))
    prompt_template = get_translation_prompt(args.category)
    chain = prompt_template | llm

    raw_text = input_path.read_text(encoding="utf-8")
    chunks = split_markdown(raw_text)
    print(f"Total chunks: {len(chunks)}")

    if output_path.exists():
        output_path.unlink()

    for i, chunk in enumerate(chunks):
        print(f"Translating chunk {i+1}/{len(chunks)}...")
        try:
            translated = translate_chunk(chain, chunk.page_content, retry_if_mismatch=(not args.no_retry))
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(translated + "\n\n")
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            continue

    print("Done.")


if __name__ == "__main__":
    main()
