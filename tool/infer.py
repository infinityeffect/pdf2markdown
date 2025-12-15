# infer.py

import os
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm  # <--- 1. 引入 tqdm 库

__all__ = ["pdf2mark"]

# ================== 配置 ==================

MODEL_NAME = "deepseek_ocr"
CUDA_DEVICE = "0"

DEFAULT_OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "

DEFAULT_ZOOM = 2.0
DEFAULT_BASE_SIZE = 1024
DEFAULT_IMAGE_SIZE = 640

# 全局模型缓存（进程内）
_tokenizer = None
_model = None


# ========= 路径工具：给一个 pdf，算出它的输出根目录 =========

def _get_pdf_output_root(pdf_path: Path) -> Path:
    """
    比如 pdf_path = test_data/p2.pdf
    返回 test_data/ocr_output_p2 这个目录
    """
    return pdf_path.parent / f"ocr_output_{pdf_path.stem}"


# ================== 加载模型 ==================

def _load_ocr_model(
    model_name: str = MODEL_NAME,
    cuda_device: str = CUDA_DEVICE,
):
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    print(f"[OCR] Loading model: {model_name} on CUDA:{cuda_device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        # _attn_implementation='flash_attention_2'
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().cuda().to(torch.bfloat16)

    _tokenizer, _model = tokenizer, model
    print("[OCR] Model loaded.")
    return tokenizer, model


# ================== PDF -> 图片 ==================

def _pdf_to_images(
    pdf_path: Path,
    zoom: float = DEFAULT_ZOOM,
) -> List[Path]:
    """
    将 pdf 每一页渲染成 PNG 图片，返回“绝对路径”列表。
    图片路径统一放在 <root>/pages 下，例如：
        test_data/ocr_output_p2/pages/page_001.png
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    out_root = _get_pdf_output_root(pdf_path)           # test_data/ocr_output_p2
    pages_dir = out_root / "pages"
    pages_dir.mkdir(exist_ok=True, parents=True)

    print(f"[PDF] Open: {pdf_path}")
    doc = fitz.open(pdf_path)
    image_paths: List[Path] = []

    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img_path = pages_dir / f"page_{page_idx + 1:03d}.png"
        pix.save(img_path)
        img_path = img_path.resolve()   # ⭐ 转成绝对路径，防止后面 infer 找不到
        image_paths.append(img_path)
        print(f"[PDF2IMG] Page {page_idx + 1} -> {img_path}")

    print(f"[PDF] Total pages: {len(image_paths)}")
    return image_paths


# ================== 单页 infer + 找 result.mmd ==================

def _run_infer_and_get_mmd(
    image_file: Path,
    pdf_path: Path,
    page_idx: int,
    prompt: str = DEFAULT_OCR_PROMPT,
    base_size: int = DEFAULT_BASE_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    cuda_device: str = CUDA_DEVICE,
) -> Path:
    """
    对单张图片调用一次 model.infer（只写文件，不返回结果），
    然后在对应目录中找到该页的 result.mmd，并返回路径。

    输出目录统一放在：
        <root>/page_001/
        <root>/page_002/
    """
    tokenizer, model = _load_ocr_model(cuda_device=cuda_device)

    out_root = _get_pdf_output_root(pdf_path)
    page_output_dir = out_root / f"page_{page_idx:03d}"
    page_output_dir.mkdir(exist_ok=True, parents=True)

    if not isinstance(image_file, Path):
        image_file = Path(image_file)
    abs_image_path = image_file.resolve()

    print(f"[OCR] Infer on image: {abs_image_path} (cuda={cuda_device})")
    _ = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=str(abs_image_path),
        output_path=str(page_output_dir),
        base_size=base_size,
        image_size=image_size,
        crop_mode=True,
        save_results=True,
        test_compress=True,
    )

    # 优先找 <page_output_dir>/result.mmd
    result_path = page_output_dir / "result.mmd"
    if result_path.is_file():
        print(f"[OCR] Found result.mmd: {result_path}")
        return result_path

    # 兜底：找该目录下最新的 .mmd
    mmd_candidates = sorted(
        page_output_dir.rglob("*.mmd"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not mmd_candidates:
        raise RuntimeError(
            f"[OCR] 在 {page_output_dir} 中未找到 result.mmd 或任意 .mmd 文件，请检查 DeepSeek OCR 输出。"
        )

    fallback = mmd_candidates[0]
    print(f"[OCR] result.mmd 未找到，使用最新 .mmd 文件兜底: {fallback}")
    return fallback


# ================== 合并多页 .mmd ==================

def _merge_mmd_files(
    pdf_path: Path,
    page_mmd_paths: List[Path],
    output_mmd_path: Optional[Path] = None,
) -> Path:
    """
    将各页的 result.mmd 合并成一个总的 .mmd 文件。
    默认输出到：
        <root>/<pdf_stem>_ocr.mmd
    例如 test_data/ocr_output_p2/p2_ocr.mmd
    """
    out_root = _get_pdf_output_root(pdf_path)
    out_root.mkdir(exist_ok=True, parents=True)

    if output_mmd_path is None:
        output_mmd_path = out_root / f"{pdf_path.stem}_ocr.mmd"
    else:
        output_mmd_path = Path(output_mmd_path)

    print(f"[MERGE] 合并所有页面的 result.mmd -> {output_mmd_path}")
    with open(output_mmd_path, "w", encoding="utf-8") as out_f:
        for page_idx, mmd_path in enumerate(page_mmd_paths, start=1):
            with open(mmd_path, "r", encoding="utf-8") as in_f:
                page_text = in_f.read()

            header = f"\n\n---\n\n<!-- Page {page_idx} -->\n\n"
            out_f.write(header)
            out_f.write(page_text)

    print(f"[Done] 已生成合并后的 Markdown：{output_mmd_path.resolve()}")
    return output_mmd_path


# ================== 串行：整本 PDF -> mmd ==================

def _pdf_to_markdown(
    pdf_path: Path,
    output_mmd_path: Optional[Path] = None,
    *,
    zoom: float = DEFAULT_ZOOM,
    base_size: int = DEFAULT_BASE_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    cuda_device: str = CUDA_DEVICE,
) -> Path:
    """
    串行模式：单进程、单卡，从 pdf -> 多页图片 -> 每页 infer -> 合并 .mmd
    所有中间输出都收纳在 <root> = pdf.parent / f"ocr_output_{pdf.stem}"
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    # 1. 渲染 pdf -> pages
    image_paths = _pdf_to_images(pdf_path, zoom=zoom)
    if not image_paths:
        raise RuntimeError("PDF 中没有任何页面，无法转换。")

    # 2. 每一页调用 infer，收集各自的 result.mmd
    page_mmd_paths: List[Path] = []
    for idx, img_path in enumerate(tqdm(image_paths, desc="OCR 处理进度", unit="页"), start=1):
        mmd_path = _run_infer_and_get_mmd(
            image_file=img_path,
            pdf_path=pdf_path,
            page_idx=idx,
            base_size=base_size,
            image_size=image_size,
            cuda_device=cuda_device,
        )
        page_mmd_paths.append(mmd_path)

    # 3. 合并成一个总的 mmd
    result_path = _merge_mmd_files(pdf_path, page_mmd_paths, output_mmd_path)
    return result_path


# ================== 对外接口：pdf2mark ==================

def pdf2mark(
    pdf_path: str,
    output_mmd_path: Optional[str] = None,
    *,
    zoom: float = DEFAULT_ZOOM,
    base_size: int = DEFAULT_BASE_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    cuda_device: str = CUDA_DEVICE,
) -> str:
    """
    串行 OCR 入口：传入 pdf 路径，返回合并后的 .mmd 文件路径（字符串）。

    输入：比如 test_data/p2.pdf
    输出：test_data/ocr_output_p2/p2_ocr.mmd
    中间输出（图片 & DeepSeek 的文件）都在 test_data/ocr_output_p2 下。
    """
    pdf_path_obj = Path(pdf_path).resolve()
    out_path_obj = Path(output_mmd_path) if output_mmd_path else None

    result = _pdf_to_markdown(
        pdf_path_obj,
        out_path_obj,
        zoom=zoom,
        base_size=base_size,
        image_size=image_size,
        cuda_device=cuda_device,
    )
    return str(result)
