# tools/infer.py

import os
from pathlib import Path
from typing import List, Optional

import fitz
import torch
from transformers import AutoModel, AutoTokenizer

__all__ = ["pdf2mark"]  # ⭐ 对外只导出这个名字

# ===== 全局配置 / 变量 =====

MODEL_NAME = "deepseek_ocr"
CUDA_DEVICE = "0"
DEFAULT_OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "
DEFAULT_ZOOM = 2.0
DEFAULT_BASE_SIZE = 1024
DEFAULT_IMAGE_SIZE = 640
DEFAULT_PAGE_IMAGE_ROOT = "pdf_pages"
DEFAULT_OCR_OUTPUT_ROOT = "ocr_output"

_tokenizer = None
_model = None


# ===== 以下都是“内部函数”，以 _ 开头 =====

def _load_ocr_model(
    model_name: str = MODEL_NAME,
    cuda_device: str = CUDA_DEVICE,
):
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().cuda().to(torch.bfloat16)
    _tokenizer, _model = tokenizer, model
    return tokenizer, model


def _pdf_to_images(
    pdf_path: Path,
    image_root: str = DEFAULT_PAGE_IMAGE_ROOT,
    zoom: float = DEFAULT_ZOOM,
) -> List[Path]:
    pdf_stem = pdf_path.stem
    out_dir = Path(image_root) / pdf_stem
    out_dir.mkdir(exist_ok=True, parents=True)

    doc = fitz.open(pdf_path)
    image_paths: List[Path] = []

    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img_path = out_dir / f"page_{page_idx + 1:03d}.png"
        pix.save(img_path)
        image_paths.append(img_path)

    return image_paths


def _run_infer_and_get_mmd(
    image_file: Path,
    prompt: str = DEFAULT_OCR_PROMPT,
    output_dir: Optional[Path] = None,
    base_size: int = DEFAULT_BASE_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Path:
    tokenizer, model = _load_ocr_model()

    if output_dir is None:
        output_dir = Path(DEFAULT_OCR_OUTPUT_ROOT) / image_file.stem
    output_dir.mkdir(exist_ok=True, parents=True)

    _ = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=str(image_file),
        output_path=str(output_dir),
        base_size=base_size,
        image_size=image_size,
        crop_mode=True,
        save_results=True,
        test_compress=True,
    )

    result_path = output_dir / "result.mmd"
    if result_path.is_file():
        return result_path

    mmd_candidates = sorted(
        output_dir.rglob("*.mmd"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not mmd_candidates:
        raise RuntimeError(f"no .mmd found in {output_dir}")
    return mmd_candidates[0]


def _pdf_to_markdown(
    pdf_path: Path,
    output_mmd_path: Optional[Path] = None,
    *,
    zoom: float = DEFAULT_ZOOM,
    base_size: int = DEFAULT_BASE_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Path:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    image_paths = _pdf_to_images(pdf_path, DEFAULT_PAGE_IMAGE_ROOT, zoom=zoom)
    if not image_paths:
        raise RuntimeError("no pages in PDF")

    page_mmds: List[Path] = []
    for idx, img_path in enumerate(image_paths, start=1):
        page_output_dir = Path(DEFAULT_OCR_OUTPUT_ROOT) / f"{pdf_path.stem}_page_{idx:03d}"
        mmd_path = _run_infer_and_get_mmd(
            image_file=img_path,
            output_dir=page_output_dir,
            base_size=base_size,
            image_size=image_size,
        )
        page_mmds.append(mmd_path)

    if output_mmd_path is None:
        output_mmd_path = pdf_path.with_name(pdf_path.stem + "_ocr.mmd")
    output_mmd_path = Path(output_mmd_path)

    with open(output_mmd_path, "w", encoding="utf-8") as out_f:
        for idx, mmd_file in enumerate(page_mmds, start=1):
            with open(mmd_file, "r", encoding="utf-8") as in_f:
                text = in_f.read()
            out_f.write(f"\n\n---\n\n<!-- Page {idx} -->\n\n")
            out_f.write(text)

    return output_mmd_path


# ===== 对外唯一暴露的函数 =====

def pdf2mark(
    pdf_path: str,
    output_mmd_path: Optional[str] = None,
    *,
    zoom: float = DEFAULT_ZOOM,
    base_size: int = DEFAULT_BASE_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> str:
    """
    对外使用的唯一入口：
    传入 pdf 路径，返回合并后的 .mmd 文件路径（字符串）。
    其他所有函数都只作为内部实现使用。
    """
    pdf_path_obj = Path(pdf_path)
    out_path_obj = Path(output_mmd_path) if output_mmd_path else None
    result = _pdf_to_markdown(
        pdf_path_obj,
        out_path_obj,
        zoom=zoom,
        base_size=base_size,
        image_size=image_size,
    )
    return str(result)
