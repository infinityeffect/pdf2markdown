import argparse
from pathlib import Path

from tool.infer import pdf2mark
from tool.pipeline import pdf2mark_pipline


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown using DeepSeek-OCR."
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the PDF file to convert."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output markdown file path."
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel OCR (auto-detect GPU)."
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 不存在：{pdf_path}")

    print("[INFO] 输入 PDF:", pdf_path)

    # 串行 or 并行
    if args.parallel:
        print("[INFO] 使用并行模式（自动检测 GPU）")
        out_path = pdf2mark_pipline(
            str(pdf_path),
            args.out
        )
    else:
        print("[INFO] 使用串行模式")
        out_path = pdf2mark(
            str(pdf_path),
            args.out
        )

    print("\n[INFO] 转换完成！输出文件：", out_path)


if __name__ == "__main__":
    main()
