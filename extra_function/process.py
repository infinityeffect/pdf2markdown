import re
import os
import shutil
import argparse
from pathlib import Path

# ================= 工具函数 =================

def is_special_block(line: str) -> bool:
    """判断是否是特殊块（标题、图片占位符、公式块）"""
    line = line.strip()
    return (
        line.startswith("#")
        or line.startswith("<<IMG:")
        or line.startswith("$$")
    )

def is_sentence_end(text: str) -> bool:
    """判断上一段是否看似结束"""
    if not text:
        return True
    text = text.strip()
    terminal_puncts = (".", "?", "!", '"', "”", "’", "…", "]", ")", "}")
    return text[-1] in terminal_puncts

def clean_html_wrappers(text: str) -> str:
    """
    清理 OCR 产生的 HTML 包装标签，但保留内容。
    解决 <center> 导致公式不渲染的问题。
    """
    text = re.sub(r"</?center>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?div.*?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"\[>", "", text)
    return text

def fix_latex_formulas(text: str) -> str:
    """标准 LaTeX -> Markdown 公式转换"""
    text = re.sub(r"\\\[(.*?)\\\]", r"\n$$\1$$\n", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text)
    return text

def default_output_dir(input_dir: Path) -> Path:
    """默认输出目录：原始目录名 + _clean"""
    return input_dir.parent / f"{input_dir.name}_clean"

def default_output_md_name(input_dir: Path) -> str:
    """默认输出 md 文件名：<输入目录名>_cleaned.md"""
    return f"{input_dir.name}_cleaned.md"

def find_main_mmd(input_dir: Path, mmd_name: str | None = None) -> Path:
    """
    找到待处理的 mmd 文件：
    - 如果用户指定了 mmd_name，就用 input_dir/mmd_name
    - 否则：优先找 *_ocr.mmd；若有多个选第一个；否则找目录下任意 .mmd
    """
    if mmd_name:
        p = input_dir / mmd_name
        return p

    candidates = sorted(input_dir.glob("*_ocr.mmd"))
    if candidates:
        return candidates[0]

    any_mmd = sorted(input_dir.glob("*.mmd"))
    if any_mmd:
        return any_mmd[0]

    # 如果没有 mmd，返回一个默认路径，后续会报错
    return input_dir / "UNKNOWN.mmd"

# ================= 主处理函数 =================

def process_ocr_final(input_dir: str, output_dir: str, mmd_filename: str | None = None, output_md_filename: str | None = None):
    input_dir_p = Path(input_dir).expanduser().resolve()
    output_dir_p = Path(output_dir).expanduser().resolve()

    input_mmd_path = find_main_mmd(input_dir_p, mmd_filename)
    if not input_mmd_path.exists():
        raise FileNotFoundError(f"❌ 找不到输入 mmd 文件: {input_mmd_path}")

    if output_md_filename is None:
        output_md_filename = default_output_md_name(input_dir_p)

    output_md_path = output_dir_p / output_md_filename
    output_assets_dir = output_dir_p / "assets"

    output_dir_p.mkdir(parents=True, exist_ok=True)
    output_assets_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 开始处理...")
    print(f"📥 输入目录: {input_dir_p}")
    print(f"📄 输入文件: {input_mmd_path.name}")
    print(f"📤 输出目录: {output_dir_p}")
    print(f"📝 输出MD : {output_md_path.name}")
    print(f"🖼️  输出图片目录: {output_assets_dir}")

    content = input_mmd_path.read_text(encoding="utf-8")

    # ========================================================
    # 阶段一：提取图片并生成占位符
    # ========================================================
    page_pattern = re.compile(r"<!-- Page (\d+) -->")
    chunks = page_pattern.split(content)

    processed_parts = []
    if chunks and chunks[0]:
        processed_parts.append(chunks[0])

    for i in range(1, len(chunks), 2):
        page_num = chunks[i]
        page_text = chunks[i + 1] if i + 1 < len(chunks) else ""

        page_folder = f"page_{int(page_num):03d}"
        src_img_dir = input_dir_p / page_folder / "images"

        def handle_image(match):
            alt_text = match.group(1)          # ![这里]
            original_name = match.group(2)     # (images/这里)
            new_name = f"{page_folder}_{original_name}"

            src_file = os.path.join(src_img_dir, original_name)
            dst_file = os.path.join(output_assets_dir, new_name)

            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
            else:
                print(f"⚠️  警告: 图片文件缺失 {src_file}")

            # ✅ 关闭占位符：改为正常 Markdown 图片引用（指向输出的 assets）
            return f"\n\n![{alt_text}](assets/{new_name})\n\n"

        # 兼容: ![...](images/xxx) / ![...] (images/xxx) / 中间有空格
        page_text_new = re.sub(r"!\[(.*?)\]\s*\(images/(.*?)\)", handle_image, page_text)
        processed_parts.append(page_text_new)

    full_text = "".join(processed_parts)

    # ========================================================
    # 阶段二：清理 HTML 和 伪影
    # ========================================================
    full_text = re.sub(r"<!-- Page \d+ -->", "", full_text)
    full_text = re.sub(r"\n---\n", "\n", full_text)
    full_text = re.sub(r"^---$", "", full_text, flags=re.MULTILINE)

    full_text = clean_html_wrappers(full_text)
    full_text = fix_latex_formulas(full_text)

    # 修复连字符单词
    full_text = re.sub(r"(\w)-\s+(\w)", r"\1-\2", full_text)

    # ========================================================
    # 阶段三：智能合并段落
    # ========================================================
    lines = full_text.split("\n")
    merged_paragraphs = []
    buffer = ""

    for line in lines:
        stripped = line.strip()

        # 1) 特殊块
        if is_special_block(stripped):
            if buffer:
                merged_paragraphs.append(buffer)
                buffer = ""
            merged_paragraphs.append(stripped)
            continue

        # 2) 空行
        if not stripped:
            if buffer and is_sentence_end(buffer):
                merged_paragraphs.append(buffer)
                buffer = ""
            continue

        # 3) 普通文本合并
        if buffer:
            if buffer.endswith("-"):
                buffer = buffer[:-1] + stripped
            else:
                buffer += " " + stripped
        else:
            buffer = stripped

    if buffer:
        merged_paragraphs.append(buffer)

    final_output = "\n\n".join(merged_paragraphs)
    final_output = re.sub(r"\n{3,}", "\n\n", final_output)

    # ========================================================
    # 阶段四：保存
    # ========================================================
    output_md_path.write_text(final_output, encoding="utf-8")

    print("-" * 40)
    print("✅ 处理完成！")
    print(f"📄 结果文件: {output_md_path}")
    print("🧹 已去除 <center>/<div> 标签，公式应可正常显示")
    print("🗑️  已去除 OCR 伪影符号 '[>'")
    print(f"🖼️  图片已复制到: {output_assets_dir}")
    print("-" * 40)

# ================= CLI 入口 =================

def main():
    parser = argparse.ArgumentParser(
        description="Process OCR mmd: extract images to assets/, replace with <<IMG:...>>, clean html, fix latex, merge paragraphs."
    )
    parser.add_argument(
        "input_dir",
        help="待处理的文件夹路径（包含 .mmd 和 page_XXX/images/ 等）"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="可选：输出目录。不填则默认 input_dir + '_clean'"
    )
    parser.add_argument(
        "--mmd",
        default=None,
        help="可选：指定要处理的 mmd 文件名（相对 input_dir）。不填则自动寻找 *_ocr.mmd 或任意 .mmd"
    )
    parser.add_argument(
        "--out-md",
        default=None,
        help="可选：输出 md 文件名（仅文件名，不含路径）。不填则默认 <输入目录名>_cleaned.md"
    )

    args = parser.parse_args()
    input_dir_p = Path(args.input_dir).expanduser().resolve()

    if not input_dir_p.exists() or not input_dir_p.is_dir():
        raise NotADirectoryError(f"输入目录不存在或不是目录: {input_dir_p}")

    if args.out is None:
        output_dir_p = default_output_dir(input_dir_p)
    else:
        output_dir_p = Path(args.out).expanduser().resolve()

    process_ocr_final(
        input_dir=str(input_dir_p),
        output_dir=str(output_dir_p),
        mmd_filename=args.mmd,
        output_md_filename=args.out_md
    )

if __name__ == "__main__":
    main()
