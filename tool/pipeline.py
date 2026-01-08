# tool/pipeline.py

import os
from itertools import cycle
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from . import infer as _infer

REQUIRED_VRAM_GB = 8.0  # 你估计 deepseek_ocr 需要的显存

__all__ = ["pdf2mark_pipline"]  # 并行版本入口名（按你之前的拼写）


# ================== 工具：检测 GPU & 可用显存 ==================

def _get_gpu_slots(required_gb: float = REQUIRED_VRAM_GB) -> List[str]:
    """
    检测当前机器上可用于并行跑 deepseek_ocr 的“GPU 插槽”。
    返回 ["0", "1", ...]，表示每块卡可以承载一个进程。
    """
    if not torch.cuda.is_available():
        print("[GPU] 未检测到 CUDA，无法并行。")
        return []

    num_devices = torch.cuda.device_count()
    slots: List[str] = []

    for dev in range(num_devices):
        torch.cuda.set_device(dev)
        free_gb = None
        total_gb = None

        if hasattr(torch.cuda, "mem_get_info"):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
        else:
            props = torch.cuda.get_device_properties(dev)
            total_gb = props.total_memory / (1024 ** 3)

        if free_gb is not None:
            print(f"[GPU] device {dev}: free={free_gb:.2f}GB total={total_gb:.2f}GB")
            if free_gb >= required_gb:
                slots.append(str(dev))
        else:
            print(f"[GPU] device {dev}: total={total_gb:.2f}GB (free未知)")
            if total_gb >= required_gb:
                slots.append(str(dev))

    if not slots:
        print("[GPU] 没有任何一块卡满足显存要求，退回串行。")
    else:
        print(f"[GPU] 可用 GPU 插槽: {slots}")

    return slots


# ================== worker 进程：处理单页 ==================

def _worker_ocr_page(args: Tuple[int, str, str, str, int, int]) -> Tuple[int, str]:
    """
    子进程执行函数：
      args: (page_idx, image_path, pdf_path, cuda_device, base_size, image_size)
      返回: (page_idx, mmd_path_str)
    """
    (page_idx, image_path, pdf_path, cuda_device, base_size, image_size) = args

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    image_path = Path(image_path)
    pdf_path = Path(pdf_path)

    mmd_path = _infer._run_infer_and_get_mmd(
        image_file=image_path,
        pdf_path=pdf_path,
        page_idx=page_idx,
        base_size=base_size,
        image_size=image_size,
        cuda_device=cuda_device,
    )

    return page_idx, str(mmd_path)


# ================== 并行版本入口：pdf2mark_pipline ==================

def pdf2mark_pipline(
    pdf_path: str,
    output_mmd_path: Optional[str] = None,
    *,
    zoom: float = _infer.DEFAULT_ZOOM,
    base_size: int = _infer.DEFAULT_BASE_SIZE,
    image_size: int = _infer.DEFAULT_IMAGE_SIZE,
    required_gb: float = REQUIRED_VRAM_GB,
    max_workers: Optional[int] = None,
) -> str:
    """
    并行 OCR 入口：
      - 自动检测 GPU 和显存，决定是否使用多进程；
      - 如果条件不满足，则退回 infer.pdf2mark 串行实现。

    返回值为最终合并后的 .mmd 文件路径（字符串）。
    """
    pdf_path_obj = Path(pdf_path).resolve()
    if not pdf_path_obj.is_file():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path_obj}")

    # 1. 用 infer 的内部函数渲染 PDF -> 图片
    image_paths = _infer._pdf_to_images(pdf_path_obj, zoom=zoom)
    if not image_paths:
        raise RuntimeError("PDF 中没有任何页面，无法转换。")

    # 2. 检查是否适合并行
    gpu_slots = _get_gpu_slots(required_gb=required_gb)

    # 条件：至少有 2 个 GPU 插槽 且 页数 > 1；否则回退串行
    if len(gpu_slots) < 2 or len(image_paths) < 2:
        print("[PIPELINE] 并行条件不满足，使用串行 pdf2mark。")
        return _infer.pdf2mark(
            str(pdf_path_obj),
            output_mmd_path,
            zoom=zoom,
            base_size=base_size,
            image_size=image_size,
        )

    # 3. 准备任务列表（轮流分配 GPU）
    if max_workers is not None:
        max_workers = max(1, max_workers)
        num_workers = min(max_workers, len(gpu_slots), len(image_paths))
    else:
        num_workers = min(len(gpu_slots), len(image_paths))

    gpu_cycle = cycle(gpu_slots[:num_workers])
    tasks: List[Tuple[int, str, str, str, int, int]] = []

    for idx, img_path in enumerate(image_paths, start=1):
        cuda_dev = next(gpu_cycle)
        tasks.append(
            (
                idx,
                str(img_path),
                str(pdf_path_obj),
                cuda_dev,
                base_size,
                image_size,
            )
        )

    # 4. 多进程运行
    print(f"[PIPELINE] 使用多进程并行 OCR，workers={num_workers}, tasks={len(tasks)}")
    page_results: List[Tuple[int, str]] = []

    with Pool(processes=num_workers) as pool:
        for page_idx, mmd_path in pool.imap_unordered(_worker_ocr_page, tasks):
            print(f"[PIPELINE] Page {page_idx} 完成: {mmd_path}")
            page_results.append((page_idx, mmd_path))

    # 5. 按页索引排序，并复用 infer._merge_mmd_files 合并
    page_results.sort(key=lambda x: x[0])
    page_mmd_paths = [Path(p[1]) for p in page_results]
    out_path_obj = Path(output_mmd_path) if output_mmd_path else None

    final_path = _infer._merge_mmd_files(pdf_path_obj, page_mmd_paths, out_path_obj)
    return str(final_path)
