
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

    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

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

# 配置建议
显卡显存 >= 8GB

测试环境：Win10

# 备注
本项目默认支持flash-attn，如果想关闭flash-attn，请在

    infer.py
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
       # _attn_implementation='flash_attention_2'
    )

将flash-attn选项注释掉即可

# 未来升级

## 1.bash上不在输出识别结果（提交一份新的deepseekocr推理文件即可）

## 2.跨平台支持（目前只在win平台调试）

## 3.test数据只留一个测试样例

## 4. 项目文件优化

