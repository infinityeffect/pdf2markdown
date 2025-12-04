# convert_p1.py

from tool.infer import pdf2mark
from tool.pipeline import pdf2mark_pipline

if __name__ == "__main__":
    pdf = "test_data/p2.pdf"

    # 串行版本（永远安全）
    # out_path = pdf2mark(pdf)

    # 并行版本：自动检测显存，不满足条件时会自动回退到串行
    out_path = pdf2mark_pipline(pdf)

    print("生成文件：", out_path)
