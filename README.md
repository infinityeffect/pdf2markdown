
# 内容
本项目实现了pdf转化为mmd文件。

## Quick Start
### 1.拉取项目
    
    git clone https://github.com/infinityeffect/pdf2markdown.git

### 2.配置conda环境（通过ymal 安装环境）

    conda env create -f environment.yml -n DeepSeek-OCR

### 3.从hugging face上下载deepseek ocr权重等文件，或者运行
    
    python .\download.py

### 4.项目结构


### 5.使用代码执行转化

    from tool.infer import pdf2mark
    from tool.pipeline import pdf2mark_pipline
    pdf = "test_data/p2.pdf"
    # 串行版本（永远安全）
    # out_path = pdf2mark(pdf)
    # 并行版本：自动检测显存，不满足条件时会自动回退到串行
    out_path = pdf2mark_pipline(pdf)
    print("生成文件：", out_path)

 或者

    python convert.py test_data/p2.pdf


