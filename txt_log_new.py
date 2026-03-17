#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把推理日志里的 inner/outer 图像路径解析出来，拼成横向布局并写成多条 mp4。

功能：
1. 以 "==> this batch result:" 作为一帧结束标记解析日志
2. 从日志中提取：
   - inner_img_path
   - inputs_img_path
   - pred / gt
   - speed / acc
3. 按 inputs_img_path 对应的视频目录自动分组
4. 每组按帧号顺序写成一个 mp4
5. 车外图 / 车内图左右拼接（可配置谁在左边）
6. 左上信息区写 pred / gt、speed、acc（已去掉 tensor(...)）
7. gt == 1 的帧会加明显红框
8. 使用更稳妥的尺寸、字体、卡片和留白，让整体观感更舒服

示例：
python visualize_log_to_mp4.py \
    --log /path/to/log.txt \
    --out-dir ./vis_videos \
    --fps 10

如果想让 inner 放左边：
python visualize_log_to_mp4.py --log /path/to/log.txt --out-dir ./vis_videos --left-image inner

如果只想先检查日志解析结果：
python visualize_log_to_mp4.py --log /path/to/log.txt --dry-run
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


BATCH_MARKER = "==> this batch result:"


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize inference log into multi mp4 videos.")
    parser.add_argument("--log", required=True, help="推理日志 txt 路径")
    parser.add_argument("--out-dir", required=True, help="输出 mp4 目录")
    parser.add_argument("--fps", type=float, default=10.0, help="输出视频 fps，默认 10")
    parser.add_argument(
        "--left-image",
        choices=["outer", "inner"],
        default="outer",
        help="左右拼接时谁在左边，默认 outer",
    )
    # 为兼容旧脚本保留这个参数，但隐藏帮助。
    parser.add_argument(
        "--top-image",
        choices=["outer", "inner"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=None,
        help="输出视频总宽度；默认自动估计一个舒适尺寸",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="cv2.VideoWriter_fourcc codec，默认 mp4v",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=24,
        help="左右两个画面卡片之间的间隔像素，默认 24",
    )
    parser.add_argument(
        "--border-thickness",
        type=int,
        default=14,
        help="gt==1 时红框粗细，默认 14",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果 base_name.mp4 已存在则跳过",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help="只处理前 N 条视频，调试用",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析日志并打印摘要，不读取图片、不生成视频",
    )
    args = parser.parse_args()
    if args.top_image is not None:
        args.left_image = args.top_image
    return args


# -----------------------------
# Parsing utilities
# -----------------------------

def split_top_level_expr(expr: str) -> str:
    depth = 0
    for i, ch in enumerate(expr):
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth -= 1
        elif ch == "," and depth == 0:
            return expr[:i].strip()
    return expr.strip()


def parse_tensor_string(tensor_str: str) -> Any:
    s = tensor_str.strip()
    if s.startswith("tensor(") and s.endswith(")"):
        inner = s[len("tensor(") : -1].strip()
        inner = split_top_level_expr(inner)
    else:
        inner = s
    inner = inner.replace("nan", "None").replace("inf", "1e309").replace("-1e309", "-1e309")
    try:
        value = ast.literal_eval(inner)
    except Exception:
        return inner
    return value


def safe_literal_list(s: str) -> List[Any]:
    try:
        value = ast.literal_eval(s)
        if isinstance(value, list):
            return value
        return [value]
    except Exception as exc:
        raise ValueError(f"无法解析列表: {s}") from exc


def deep_equal(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return all(deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def normalize_structure(x: Any) -> Any:
    if isinstance(x, tuple):
        x = list(x)
    if isinstance(x, list):
        x = [normalize_structure(v) for v in x]
        while isinstance(x, list) and len(x) == 1:
            x = x[0]
        if isinstance(x, list) and x and all(deep_equal(v, x[0]) for v in x[1:]):
            x = x[0]
        return x
    return x


def extract_scalar(x: Any) -> Optional[float]:
    x = normalize_structure(x)
    if isinstance(x, (int, float)):
        return float(x)
    return None


def format_number(x: Any) -> str:
    if x is None:
        return "None"
    if isinstance(x, bool):
        return str(int(x))
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.4f}"
    return str(x)


def format_tensor_value(x: Any, max_items: int = 8) -> str:
    x = normalize_structure(x)
    if isinstance(x, list):
        if any(isinstance(v, list) for v in x):
            return " | ".join(format_tensor_value(v, max_items=max_items) for v in x)
        values = x[:max_items]
        text = ", ".join(format_number(v) for v in values)
        if len(x) > max_items:
            text += ", ..."
        return f"[{text}]"
    return format_number(x)


def parse_block(block: str, seq_idx: int) -> Optional[Dict[str, Any]]:
    inner_match = re.search(
        r"inner_img_path\s*:?\s*(\[[\s\S]*?\])\s*inputs_img_path",
        block,
        flags=re.S,
    )
    outer_match = re.search(
        r"inputs_img_path\s*:?\s*(\[[\s\S]*?\])\s*pred/gt:",
        block,
        flags=re.S,
    )
    pred_gt_match = re.search(
        r"pred/gt:\s*(tensor\([\s\S]*?\))\s*/\s*(tensor\([\s\S]*?\))\s*speed/acc:",
        block,
        flags=re.S,
    )
    speed_acc_match = re.search(
        r"speed/acc:\s*(tensor\([\s\S]*?\))\s*/\s*(tensor\([\s\S]*?\))\s*(?:use DDP mode|validating|$)",
        block,
        flags=re.S,
    )

    if not (inner_match and outer_match and pred_gt_match and speed_acc_match):
        return None

    inner_paths = safe_literal_list(inner_match.group(1).strip())
    outer_paths = safe_literal_list(outer_match.group(1).strip())
    if not inner_paths or not outer_paths:
        return None

    inner_path = str(inner_paths[0])
    outer_path = str(outer_paths[0])

    pred = parse_tensor_string(pred_gt_match.group(1))
    gt = parse_tensor_string(pred_gt_match.group(2))
    speed = parse_tensor_string(speed_acc_match.group(1))
    acc = parse_tensor_string(speed_acc_match.group(2))

    video_dir = str(Path(outer_path).parent.parent)
    frame_stem = Path(outer_path).stem
    try:
        frame_idx = int(re.search(r"(\d+)$", frame_stem).group(1))  # type: ignore[union-attr]
    except Exception:
        frame_idx = seq_idx

    pred_text = format_tensor_value(pred)
    gt_text = format_tensor_value(gt)
    speed_text = format_tensor_value(speed)
    acc_text = format_tensor_value(acc)
    gt_scalar = extract_scalar(gt)
    is_positive = gt_scalar is not None and gt_scalar >= 0.5

    return {
        "seq_idx": seq_idx,
        "frame_idx": frame_idx,
        "video_dir": video_dir,
        "video_name": Path(video_dir).name,
        "inner_path": inner_path,
        "outer_path": outer_path,
        "pred": pred,
        "gt": gt,
        "speed": speed,
        "acc": acc,
        "pred_text": pred_text,
        "gt_text": gt_text,
        "speed_text": speed_text,
        "acc_text": acc_text,
        "is_positive": is_positive,
    }


def parse_log(log_path: str) -> List[Dict[str, Any]]:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    parts = text.split(BATCH_MARKER)
    frames: List[Dict[str, Any]] = []
    dropped: List[int] = []
    for i, block in enumerate(parts[1:], start=1):
        item = parse_block(block, i)
        if item is not None:
            frames.append(item)
        else:
            dropped.append(i)

    if dropped:
        head = ", ".join(str(i) for i in dropped[:10])
        tail = " ..." if len(dropped) > 10 else ""
        print(f"[warn] 有 {len(dropped)} 个 batch 未解析成功，序号: {head}{tail}", file=sys.stderr)
    return frames


def group_frames(frames: Sequence[Dict[str, Any]]) -> "OrderedDict[str, List[Dict[str, Any]]]":
    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for item in frames:
        grouped.setdefault(item["video_dir"], []).append(item)
    for _, items in grouped.items():
        items.sort(key=lambda x: (x["frame_idx"], x["seq_idx"]))
    return grouped


# -----------------------------
# Drawing utilities
# -----------------------------

def ensure_even(x: int) -> int:
    x = int(round(x))
    return x if x % 2 == 0 else x + 1


@lru_cache(maxsize=64)
def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(8, int(size))
    candidates: List[str] = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ]
        )
    for fp in candidates:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size=size)
    return ImageFont.load_default()


def open_image_safe(path: str) -> Optional[Image.Image]:
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            return img.convert("RGB")
    except Exception:
        return None


def get_reference_sizes(frames: Sequence[Dict[str, Any]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    outer_size = None
    inner_size = None
    for item in frames:
        if outer_size is None:
            img = open_image_safe(item["outer_path"])
            if img is not None:
                outer_size = img.size
        if inner_size is None:
            img = open_image_safe(item["inner_path"])
            if img is not None:
                inner_size = img.size
        if outer_size is not None and inner_size is not None:
            break

    if outer_size is None and inner_size is None:
        # 两类图都读不到时，给一个稳定的默认比例，避免无法出图。
        outer_size = (1280, 720)
        inner_size = (1280, 720)
    elif outer_size is None:
        outer_size = inner_size
    elif inner_size is None:
        inner_size = outer_size

    assert outer_size is not None and inner_size is not None
    return outer_size, inner_size


def contain_and_pad(img: Image.Image, target_size: Tuple[int, int], bg: Tuple[int, int, int]) -> Image.Image:
    fitted = ImageOps.contain(img, target_size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, bg)
    x = (target_size[0] - fitted.width) // 2
    y = (target_size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


@lru_cache(maxsize=64)
def rounded_mask(size: Tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return mask


def paste_rounded(base: Image.Image, img: Image.Image, xy: Tuple[int, int], radius: int) -> None:
    mask = rounded_mask(img.size, min(radius, img.size[0] // 2, img.size[1] // 2))
    base.paste(img, xy, mask)


def text_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> int:
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    return max(1, bbox[3] - bbox[1])


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return max(1, bbox[2] - bbox[0])


def wrap_text_by_pixels(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    max_width: int,
) -> List[str]:
    if not text:
        return [""]
    if text_width(draw, text, font) <= max_width:
        return [text]

    words = text.split(" ")
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        trial = current + " " + word
        if text_width(draw, trial, font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)

    final_lines: List[str] = []
    for line in lines:
        if text_width(draw, line, font) <= max_width:
            final_lines.append(line)
            continue
        buf = ""
        for ch in line:
            trial = buf + ch
            if buf and text_width(draw, trial, font) > max_width:
                final_lines.append(buf)
                buf = ch
            else:
                buf = trial
        if buf:
            final_lines.append(buf)
    return final_lines or [""]


def shorten_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    left = (max_chars - 3) // 2
    right = max_chars - 3 - left
    return text[:left] + "..." + text[-right:]




def shorten_middle_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    max_width: int,
) -> str:
    if text_width(draw, text, font) <= max_width:
        return text
    lo, hi = 1, len(text)
    best = "..."
    while lo <= hi:
        keep = (lo + hi) // 2
        left = keep // 2
        right = keep - left
        candidate = text[:left] + "..." + text[-right:]
        if text_width(draw, candidate, font) <= max_width:
            best = candidate
            lo = keep + 1
        else:
            hi = keep - 1
    return best

def build_placeholder(
    target_size: Tuple[int, int],
    title: str,
    path: str,
    palette: Dict[str, Tuple[int, int, int]],
) -> Image.Image:
    canvas = Image.new("RGB", target_size, palette["img_frame_bg"])
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(max(18, min(32, target_size[0] // 24)), bold=True)
    path_font = load_font(max(14, min(22, target_size[0] // 34)), bold=False)

    pad = max(18, target_size[0] // 40)
    inner = (pad, pad, target_size[0] - pad, target_size[1] - pad)
    draw.rounded_rectangle(
        inner,
        radius=max(16, pad),
        outline=(86, 94, 108),
        width=2,
        fill=(22, 26, 33),
    )

    cx = target_size[0] // 2
    y = inner[1] + pad
    title_w = text_width(draw, title, title_font)
    draw.text((cx - title_w // 2, y), title, font=title_font, fill=(242, 245, 250))
    y += text_height(draw, title_font) + 18

    placeholder_hint = "image not found"
    hint_w = text_width(draw, placeholder_hint, path_font)
    draw.text((cx - hint_w // 2, y), placeholder_hint, font=path_font, fill=(167, 176, 189))
    y += text_height(draw, path_font) + 22

    wrapped = wrap_text_by_pixels(draw, path, path_font, max_width=target_size[0] - pad * 4)
    wrapped = wrapped[:4]
    for line in wrapped:
        line = shorten_middle_to_width(draw, line, path_font, target_size[0] - pad * 4)
        line_w = text_width(draw, line, path_font)
        draw.text((cx - line_w // 2, y), line, font=path_font, fill=(198, 205, 214))
        y += text_height(draw, path_font) + 6
    return canvas


def prepare_image(
    path: str,
    target_size: Tuple[int, int],
    missing_title: str,
    palette: Dict[str, Tuple[int, int, int]],
) -> Image.Image:
    img = open_image_safe(path)
    if img is None:
        return build_placeholder(target_size, missing_title, path, palette)
    return contain_and_pad(img, target_size, bg=palette["img_frame_bg"])


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "video"


def ensure_unique_output_path(out_dir: str, base_name: str) -> str:
    safe_base = safe_filename(base_name)
    candidate = Path(out_dir) / f"{safe_base}.mp4"
    if not candidate.exists():
        return str(candidate)
    idx = 2
    while True:
        candidate = Path(out_dir) / f"{safe_base}_{idx}.mp4"
        if not candidate.exists():
            return str(candidate)
        idx += 1


def build_palette() -> Dict[str, Tuple[int, int, int]]:
    return {
        "canvas_bg": (11, 14, 18),
        "card_bg": (20, 24, 31),
        "card_bg_2": (23, 28, 36),
        "card_border": (52, 59, 70),
        "panel_bg": (18, 22, 28),
        "panel_border": (50, 56, 66),
        "img_frame_bg": (8, 11, 15),
        "img_frame_border": (66, 73, 84),
        "text": (240, 243, 248),
        "muted": (161, 170, 184),
        "accent": (255, 228, 170),
        "accent_dim": (232, 210, 158),
        "positive": (232, 52, 70),
        "positive_bg": (86, 23, 30),
    }


def get_style(final_width: int) -> Dict[str, Any]:
    palette = build_palette()
    page_pad = max(18, min(34, final_width // 55))
    card_gap = max(16, min(26, final_width // 80))
    panel_inner_pad = max(14, min(24, final_width // 90))
    panel_title_gap = max(10, min(16, final_width // 120))

    style: Dict[str, Any] = {
        "palette": palette,
        "page_pad": page_pad,
        "card_gap": card_gap,
        "panel_inner_pad": panel_inner_pad,
        "panel_title_gap": panel_title_gap,
        "radius_l": max(18, min(28, final_width // 70)),
        "radius_m": max(14, min(22, final_width // 90)),
        "radius_s": max(10, min(16, final_width // 120)),
        "header_title_font": load_font(max(14, min(20, final_width // 110)), bold=True),
        "metric_font": load_font(max(19, min(30, final_width // 60)), bold=False),
        "metric_font_bold": load_font(max(19, min(30, final_width // 60)), bold=True),
        "meta_font": load_font(max(17, min(24, final_width // 76)), bold=False),
        "meta_small_font": load_font(max(14, min(18, final_width // 108)), bold=False),
        "panel_title_font": load_font(max(15, min(20, final_width // 105)), bold=True),
    }
    return style


def compute_auto_width(outer_ref: Tuple[int, int], inner_ref: Tuple[int, int]) -> int:
    out_w, _ = outer_ref
    in_w, _ = inner_ref
    guessed = int(max(out_w, in_w) * 1.7)
    guessed = max(1280, min(1920, guessed))
    return ensure_even(guessed)


def compute_layout(
    frames: Sequence[Dict[str, Any]],
    outer_ref: Tuple[int, int],
    inner_ref: Tuple[int, int],
    target_width: Optional[int],
    gap: int,
) -> Dict[str, Any]:
    final_width = ensure_even(target_width if target_width is not None else compute_auto_width(outer_ref, inner_ref))
    style = get_style(final_width)
    palette = style["palette"]

    page_pad = style["page_pad"]
    card_gap = style["card_gap"]
    panel_inner_pad = style["panel_inner_pad"]
    panel_title_gap = style["panel_title_gap"]
    gap = max(8, gap)

    header_gap = max(16, min(24, final_width // 85))
    meta_w = min(400, max(280, final_width // 4))
    metrics_w = final_width - 2 * page_pad - header_gap - meta_w

    dummy = Image.new("RGB", (final_width, 400), palette["canvas_bg"])
    draw = ImageDraw.Draw(dummy)

    section_font = style["header_title_font"]
    metric_font = style["metric_font"]
    meta_font = style["meta_font"]
    meta_small_font = style["meta_small_font"]

    metrics_pad_x = max(20, min(28, final_width // 82))
    metrics_pad_y = max(18, min(24, final_width // 95))
    metrics_text_w = metrics_w - 2 * metrics_pad_x

    metric_lines: List[List[str]] = []
    metric_gap = max(8, min(12, final_width // 150))
    section_gap = max(10, min(14, final_width // 130))
    line_gap = max(6, min(10, final_width // 170))

    for frame in frames:
        lines = []
        for raw in [
            f"Pred / GT   {frame['pred_text']}   /   {frame['gt_text']}",
            f"Speed       {frame['speed_text']}",
            f"Acc         {frame['acc_text']}",
        ]:
            lines.extend(wrap_text_by_pixels(draw, raw, metric_font, metrics_text_w))
        metric_lines.append(lines)

    max_metric_lines = max((len(lines) for lines in metric_lines), default=3)
    header_h_left = (
        metrics_pad_y * 2
        + text_height(draw, section_font)
        + section_gap
        + max_metric_lines * text_height(draw, metric_font)
        + max(0, max_metric_lines - 1) * line_gap
    )

    meta_pad_x = metrics_pad_x
    meta_pad_y = metrics_pad_y
    meta_title_h = text_height(draw, section_font)
    meta_h = text_height(draw, meta_font)
    meta_small_h = text_height(draw, meta_small_font)
    badge_h = meta_small_h + 12
    header_h_right = meta_pad_y * 2 + meta_title_h + section_gap + meta_h * 2 + 12 + badge_h

    header_h = ensure_even(max(112, header_h_left, header_h_right))

    content_y = page_pad + header_h + card_gap
    panel_w = ensure_even((final_width - 2 * page_pad - gap) // 2)
    image_frame_w = panel_w - 2 * panel_inner_pad

    out_w_ref, out_h_ref = outer_ref
    in_w_ref, in_h_ref = inner_ref
    outer_scaled_h = max(1, round(out_h_ref * image_frame_w / out_w_ref))
    inner_scaled_h = max(1, round(in_h_ref * image_frame_w / in_w_ref))
    image_frame_h = ensure_even(max(outer_scaled_h, inner_scaled_h))

    panel_title_font = style["panel_title_font"]
    panel_title_h = text_height(draw, panel_title_font)
    panel_top_pad = max(16, min(24, final_width // 90))
    panel_bottom_pad = max(16, min(24, final_width // 90))
    panel_h = panel_top_pad + panel_title_h + panel_title_gap + image_frame_h + panel_bottom_pad

    final_height = ensure_even(content_y + panel_h + page_pad)

    layout = {
        "final_width": final_width,
        "final_height": final_height,
        "header_h": header_h,
        "header_gap": header_gap,
        "metrics_w": metrics_w,
        "meta_w": meta_w,
        "metrics_pad_x": metrics_pad_x,
        "metrics_pad_y": metrics_pad_y,
        "meta_pad_x": meta_pad_x,
        "meta_pad_y": meta_pad_y,
        "panel_w": panel_w,
        "panel_h": panel_h,
        "panel_top_pad": panel_top_pad,
        "panel_bottom_pad": panel_bottom_pad,
        "image_frame_w": image_frame_w,
        "image_frame_h": image_frame_h,
        "content_y": content_y,
        "style": style,
        "gap": gap,
    }
    return layout


def draw_round_card(
    draw: ImageDraw.ImageDraw,
    rect: Tuple[int, int, int, int],
    radius: int,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    width: int = 1,
) -> None:
    draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=width)


def draw_badge(
    canvas: Image.Image,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int],
    text_fill: Tuple[int, int, int],
    radius: int,
) -> Tuple[int, int]:
    draw = ImageDraw.Draw(canvas)
    pad_x = 12
    pad_y = 6
    tw = text_width(draw, text, font)
    th = text_height(draw, font)
    x, y = xy
    rect = (x, y, x + tw + pad_x * 2, y + th + pad_y * 2)
    draw.rounded_rectangle(rect, radius=radius, fill=fill)
    draw.text((x + pad_x, y + pad_y - 1), text, font=font, fill=text_fill)
    return rect[2], rect[3]


def draw_header(
    canvas: Image.Image,
    frame: Dict[str, Any],
    frame_pos: int,
    total_frames: int,
    layout: Dict[str, Any],
) -> None:
    draw = ImageDraw.Draw(canvas)
    style = layout["style"]
    palette = style["palette"]
    page_pad = style["page_pad"]
    radius_l = style["radius_l"]
    radius_m = style["radius_m"]

    metrics_w = layout["metrics_w"]
    meta_w = layout["meta_w"]
    header_gap = layout["header_gap"]
    header_h = layout["header_h"]

    x1 = page_pad
    y1 = page_pad
    metrics_rect = (x1, y1, x1 + metrics_w, y1 + header_h)
    meta_rect = (metrics_rect[2] + header_gap, y1, metrics_rect[2] + header_gap + meta_w, y1 + header_h)

    draw_round_card(draw, metrics_rect, radius_l, palette["card_bg"], palette["card_border"], width=1)
    draw_round_card(draw, meta_rect, radius_l, palette["card_bg_2"], palette["card_border"], width=1)

    section_font = style["header_title_font"]
    metric_font = style["metric_font"]
    meta_font = style["meta_font"]
    meta_small_font = style["meta_small_font"]

    metrics_pad_x = layout["metrics_pad_x"]
    metrics_pad_y = layout["metrics_pad_y"]
    meta_pad_x = layout["meta_pad_x"]
    meta_pad_y = layout["meta_pad_y"]
    section_gap = max(10, min(14, canvas.width // 130))
    line_gap = max(6, min(10, canvas.width // 170))

    # 左侧信息卡
    mx = metrics_rect[0] + metrics_pad_x
    my = metrics_rect[1] + metrics_pad_y
    draw.text((mx, my), "Prediction & Motion", font=section_font, fill=palette["muted"])
    my += text_height(draw, section_font) + section_gap

    metric_text_w = metrics_w - 2 * metrics_pad_x
    metric_specs = [
        (f"Pred / GT   {frame['pred_text']}   /   {frame['gt_text']}", palette["accent"]),
        (f"Speed       {frame['speed_text']}", palette["text"]),
        (f"Acc         {frame['acc_text']}", palette["text"]),
    ]
    for raw, color in metric_specs:
        wrapped = wrap_text_by_pixels(draw, raw, metric_font, metric_text_w)
        for line in wrapped:
            draw.text((mx, my), line, font=metric_font, fill=color)
            my += text_height(draw, metric_font) + line_gap

    # 右侧 meta 卡
    rx = meta_rect[0] + meta_pad_x
    ry = meta_rect[1] + meta_pad_y
    draw.text((rx, ry), "Sequence", font=section_font, fill=palette["muted"])
    ry += text_height(draw, section_font) + section_gap

    video_name = shorten_middle_to_width(draw, frame["video_name"], meta_font, meta_rect[2] - meta_rect[0] - 2 * meta_pad_x)
    draw.text((rx, ry), video_name, font=meta_font, fill=palette["text"])
    ry += text_height(draw, meta_font) + 10

    draw.text(
        (rx, ry),
        f"Frame  {frame_pos:03d} / {total_frames:03d}",
        font=meta_font,
        fill=palette["text"],
    )
    ry += text_height(draw, meta_font) + 12

    badge_text = "GT = 1" if frame["is_positive"] else "GT = 0"
    badge_fill = palette["positive_bg"] if frame["is_positive"] else (38, 44, 54)
    badge_text_fill = (255, 232, 232) if frame["is_positive"] else palette["muted"]
    draw_badge(canvas, (rx, ry), badge_text, meta_small_font, badge_fill, badge_text_fill, radius_m)


def draw_panel(
    canvas: Image.Image,
    panel_rect: Tuple[int, int, int, int],
    view_title: str,
    img: Image.Image,
    style: Dict[str, Any],
) -> None:
    draw = ImageDraw.Draw(canvas)
    palette = style["palette"]
    radius_l = style["radius_l"]
    radius_m = style["radius_m"]
    panel_inner_pad = style["panel_inner_pad"]
    panel_title_gap = style["panel_title_gap"]
    panel_title_font = style["panel_title_font"]

    draw_round_card(draw, panel_rect, radius_l, palette["panel_bg"], palette["panel_border"], width=1)

    x1, y1, x2, y2 = panel_rect
    title_x = x1 + panel_inner_pad
    title_y = y1 + style["page_pad"] // 2
    badge_fill = (34, 39, 47)
    _, title_bottom = draw_badge(
        canvas,
        (title_x, title_y),
        view_title,
        panel_title_font,
        badge_fill,
        palette["text"],
        radius_m,
    )

    image_rect = (
        x1 + panel_inner_pad,
        title_bottom + panel_title_gap,
        x2 - panel_inner_pad,
        y2 - style["panel_inner_pad"],
    )
    draw.rounded_rectangle(
        image_rect,
        radius=radius_m,
        fill=palette["img_frame_bg"],
        outline=palette["img_frame_border"],
        width=1,
    )

    image_target_w = image_rect[2] - image_rect[0]
    image_target_h = image_rect[3] - image_rect[1]
    fitted = contain_and_pad(img, (image_target_w, image_target_h), bg=palette["img_frame_bg"])
    paste_rounded(canvas, fitted, (image_rect[0], image_rect[1]), radius=radius_m)


def add_red_border(img: Image.Image, thickness: int, radius: int) -> None:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    color = (235, 44, 60)
    for i in range(thickness):
        draw.rounded_rectangle((i, i, w - 1 - i, h - 1 - i), radius=max(4, radius - i), outline=color)


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# -----------------------------
# Rendering
# -----------------------------

def render_video_group(
    frames: Sequence[Dict[str, Any]],
    out_path: str,
    fps: float,
    left_image: str,
    target_width: Optional[int],
    gap: int,
    border_thickness: int,
    codec: str,
) -> None:
    if not frames:
        return

    outer_ref, inner_ref = get_reference_sizes(frames)
    layout = compute_layout(frames, outer_ref, inner_ref, target_width, gap)
    style = layout["style"]
    palette = style["palette"]

    final_width = layout["final_width"]
    final_height = layout["final_height"]
    panel_w = layout["panel_w"]
    panel_h = layout["panel_h"]
    content_y = layout["content_y"]
    page_pad = style["page_pad"]
    radius_l = style["radius_l"]

    # 这里 target_size 是 panel 内图片区域的目标大小；画 panel 时会再次 contain，一次性统一视觉效果。
    preview_box = (layout["image_frame_w"], layout["image_frame_h"])

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (final_width, final_height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件：{out_path}，请检查 codec={codec} 是否可用")

    left_rect = (page_pad, content_y, page_pad + panel_w, content_y + panel_h)
    right_rect = (page_pad + panel_w + layout["gap"], content_y, final_width - page_pad, content_y + panel_h)

    for idx, frame in enumerate(frames, start=1):
        outer_img = prepare_image(frame["outer_path"], preview_box, "OUTER IMAGE", palette)
        inner_img = prepare_image(frame["inner_path"], preview_box, "INNER IMAGE", palette)

        if left_image == "outer":
            left_img, left_title = outer_img, "OUTER VIEW"
            right_img, right_title = inner_img, "INNER VIEW"
        else:
            left_img, left_title = inner_img, "INNER VIEW"
            right_img, right_title = outer_img, "OUTER VIEW"

        canvas = Image.new("RGB", (final_width, final_height), palette["canvas_bg"])
        draw_header(canvas, frame, idx, len(frames), layout)
        draw_panel(canvas, left_rect, left_title, left_img, style)
        draw_panel(canvas, right_rect, right_title, right_img, style)

        if frame["is_positive"]:
            add_red_border(canvas, border_thickness, radius=radius_l)

        writer.write(pil_to_bgr(canvas))

    writer.release()


# -----------------------------
# Logging / summary
# -----------------------------

def print_summary(grouped: "OrderedDict[str, List[Dict[str, Any]]]") -> None:
    total_frames = sum(len(v) for v in grouped.values())
    total_positive = sum(sum(1 for item in v if item["is_positive"]) for v in grouped.values())
    print(f"解析完成：共 {len(grouped)} 条视频，{total_frames} 帧，gt==1 的帧共 {total_positive} 帧")
    for video_dir, items in grouped.items():
        pos = sum(1 for x in items if x["is_positive"])
        print(f"  - {Path(video_dir).name}: {len(items)} 帧, positive={pos}")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    frames = parse_log(args.log)
    if not frames:
        print("没有从日志里解析出任何 batch 帧信息。", file=sys.stderr)
        sys.exit(1)

    grouped = group_frames(frames)
    if args.limit_videos is not None:
        grouped = OrderedDict(list(grouped.items())[: args.limit_videos])

    print_summary(grouped)

    if args.dry_run:
        return

    for video_dir, items in grouped.items():
        base_name = Path(video_dir).name
        expected = str(Path(args.out_dir) / f"{safe_filename(base_name)}.mp4")
        if args.skip_existing and os.path.exists(expected):
            print(f"[skip] {expected}")
            continue

        out_path = ensure_unique_output_path(args.out_dir, base_name)
        print(f"[write] {base_name} -> {out_path}")
        try:
            render_video_group(
                frames=items,
                out_path=out_path,
                fps=args.fps,
                left_image=args.left_image,
                target_width=args.target_width,
                gap=args.gap,
                border_thickness=args.border_thickness,
                codec=args.codec,
            )
        except Exception as exc:
            print(f"[error] {base_name}: {exc}", file=sys.stderr)

    print("全部处理完成。")


if __name__ == "__main__":
    main()
