import os
from huggingface_hub import snapshot_download

def download_deepseek_ocr(save_dir="deepseek_ocr0", repo_id="deepseek-ai/DeepSeek-OCR"):
    """
    使用 huggingface_hub 从 HuggingFace 下载 DeepSeek-OCR 所有文件。

    Args:
        save_dir (str): 下载到的目录
        repo_id (str): HuggingFace 仓库名称
    """
    print(f"[INFO] 开始从 HuggingFace 下载 {repo_id} ...")
    print(f"[INFO] 保存路径: {save_dir}")

    # 创建目录
    os.makedirs(save_dir, exist_ok=True)

    # snapshot_download 会下载整个仓库（包括大文件）
    snapshot_download(
        repo_id=repo_id,
        local_dir=save_dir,
        local_dir_use_symlinks=False,  # 避免符号链接，保证所有文件真实写入
        revision="main"
    )

    print("[INFO] 下载完成！文件已保存到：", os.path.abspath(save_dir))


if __name__ == "__main__":
    download_deepseek_ocr()
