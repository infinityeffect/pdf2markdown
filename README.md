
# 内容
本项目借助ocr模型实现了pdf转化为markdown。

# Quick Start

## 环境配置

### 1.拉取项目
    
    git clone https://github.com/infinityeffect/pdf2markdown.git

    cd pdf2markdown

### 2.配置conda环境

    conda env create -n DeepSeek-OCR python=3.12

    conda activate DeepSeek-OCR

    pip install torch==2.5.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.5.0 --extra-index-url https://download.pytorch.org/whl/cu118

    pip install -r requirements.txt

    pip install flash_attn-2.7.0.post2-cp312-cp312-win_amd64.whl

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

## 配置建议
显卡显存 >= 8GB

测试环境：Win10


