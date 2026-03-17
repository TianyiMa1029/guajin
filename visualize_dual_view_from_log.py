#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将双视角推理日志可视化为多个 MP4 视频。

功能：
1. 读取 log（JSON 数组或 JSON Lines 均可）。
2. 按 inputs_img_path 中 /imgs/ 之前的目录分组，每组导出为一个 mp4。
3. 每帧读取车外图（inputs_img_path）和车内图（inner_img_path），上下拼接。
4. 左上角绘制 pred / gt / speed，展示纯数值，不保留 tensor(...) 字样。
5. 当 gt == 1 时，给整帧加明显红框。
6. 尽量做更美观的排版：留白、半透明信息卡片、视角标签、统一缩放。

示例：
python visualize_dual_view_from_log.py \
    --log /path/to/vis_log.jon \
    --output_dir ./vis_videos \
    --fps 10 \
    --panel_width 1280
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# -----------------------------
# 配置
# -----------------------------
DEFAULT_BG = (18, 18, 20)           # BGR
DEFAULT_PANEL_BG = (28, 28, 32)     # BGR
DEFAULT_BORDER = (60, 60, 66)       # BGR
RED_BORDER = (30, 30, 235)          # BGR，OpenCV 用 BGR

FONT_CANDIDATES_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]

FONT_CANDIDATES_REGULAR = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


# -----------------------------
# 基础工具
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-view log -> per-video MP4 visualizer")
    parser.add_argument("--log", required=True, help="日志路径，支持 JSON 数组 / JSON Lines")
    parser.add_argument("--output_dir", default="./vis_videos", help="输出视频目录")
    parser.add_argument("--fps", type=float, default=10.0, help="输出视频帧率，默认 10")
    parser.add_argument("--panel_width", type=int, default=1280, help="单张图展示宽度，默认 1280")
    parser.add_argument("--codec", default="mp4v", help="视频编码，默认 mp4v")
    parser.add_argument(
        "--sort_mode",
        choices=["inputs", "inner", "log_order"],
        default="inputs",
        help="组内排序方式：按车外图帧号/车内图帧号/日志原顺序",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：若图片不存在/无法读取则直接报错；默认会生成占位图继续写视频",
    )
    parser.add_argument(
        "--pred_digits", type=int, default=4, help="pred/gt/speed 保留小数位，默认 4"
    )
    return parser.parse_args()


def load_log(log_path: str) -> List[Dict[str, Any]]:
    """支持 JSON 数组或 JSON Lines。"""
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    # 优先按 JSON 数组解析
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError(f"log 顶层不是 list，而是 {type(data).__name__}")
    except json.JSONDecodeError:
        pass

    # fallback: JSON Lines
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"第 {line_no} 行不是 JSON object")
            records.append(obj)
    return records


def clean_scalar(value: Any) -> Optional[float]:
    """将 int/float/np/tensor字符串 统一转成 float。"""
    if value is None:
        return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        val = float(value)
        if math.isnan(val):
            return None
        return val

    # torch tensor / numpy scalar / 其他带 item 的对象
    if hasattr(value, "item"):
        try:
            return clean_scalar(value.item())
        except Exception:
            pass

    if isinstance(value, str):
        text = value.strip()
        # 兼容: tensor(0.1234, device='cuda:0') / tensor([1.0]) / '0.1234'
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
        return None

    try:
        return float(value)
    except Exception:
        return None


def format_scalar(value: Any, digits: int = 4) -> str:
    val = clean_scalar(value)
    if val is None:
        return "N/A"
    fmt = f"{{:.{digits}f}}"
    return fmt.format(val)


def is_positive_gt(gt_value: Any, eps: float = 1e-6) -> bool:
    gt = clean_scalar(gt_value)
    return gt is not None and abs(gt - 1.0) <= eps


def extract_frame_number(path_str: str) -> int:
    stem = Path(path_str).stem
    m = re.search(r"(\d+)$", stem)
    if m:
        return int(m.group(1))
    return 10**18


def get_video_group_key(inputs_img_path: str) -> str:
    normalized = inputs_img_path.replace("\\", "/")
    if "/imgs/" in normalized:
        return normalized.split("/imgs/")[0]
    # fallback：取上两级目录
    return str(Path(inputs_img_path).parent.parent)


def make_output_name(group_key: str) -> str:
    name = Path(group_key).name
    safe = re.sub(r"[^0-9a-zA-Z._-]+", "_", name).strip("._")
    return safe or "video"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def imread_bgr(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    return cv2.imread(path, cv2.IMREAD_COLOR)


# -----------------------------
# 字体与绘制
# -----------------------------
def _first_existing(paths: Iterable[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = FONT_CANDIDATES_BOLD if bold else FONT_CANDIDATES_REGULAR
    font_path = _first_existing(candidates)
    if font_path is None:
        return ImageFont.load_default()
    try:
        return ImageFont.truetype(font_path, size=size)
    except Exception:
        return ImageFont.load_default()


def bgr_to_rgba_tuple(color_bgr: Tuple[int, int, int], alpha: int = 255) -> Tuple[int, int, int, int]:
    b, g, r = color_bgr
    return (r, g, b, alpha)


def rgb_to_bgr(arr_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(arr_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)


def draw_rounded_text_box(
    base_rgb: np.ndarray,
    xy: Tuple[int, int],
    text_lines: List[Tuple[str, Tuple[int, int, int, int], ImageFont.ImageFont]],
    padding: int = 18,
    line_gap: int = 8,
    radius: int = 20,
    box_fill: Tuple[int, int, int, int] = (18, 18, 18, 180),
    box_outline: Tuple[int, int, int, int] = (255, 255, 255, 50),
) -> np.ndarray:
    pil = Image.fromarray(base_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x, y = xy
    text_width = 0
    text_height = 0
    line_heights = []
    for text, _color, font in text_lines:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        text_width = max(text_width, w)
        line_heights.append(h)
    if line_heights:
        text_height = sum(line_heights) + line_gap * (len(line_heights) - 1)

    box = [x, y, x + text_width + padding * 2, y + text_height + padding * 2]
    draw.rounded_rectangle(box, radius=radius, fill=box_fill, outline=box_outline, width=2)

    cy = y + padding
    for (text, color, font), h in zip(text_lines, line_heights):
        draw.text((x + padding, cy), text, fill=color, font=font)
        cy += h + line_gap

    merged = Image.alpha_composite(pil, overlay).convert("RGB")
    return np.array(merged)


def draw_small_badge(
    base_rgb: np.ndarray,
    xy: Tuple[int, int],
    text: str,
    fill_rgba: Tuple[int, int, int, int],
    text_rgba: Tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    radius: int = 14,
    padding_x: int = 14,
    padding_y: int = 8,
) -> np.ndarray:
    pil = Image.fromarray(base_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    box = [x, y, x + w + 2 * padding_x, y + h + 2 * padding_y]
    draw.rounded_rectangle(box, radius=radius, fill=fill_rgba)
    draw.text((x + padding_x, y + padding_y - 1), text, font=font, fill=text_rgba)

    merged = Image.alpha_composite(pil, overlay).convert("RGB")
    return np.array(merged)


def get_badge_box_size(text: str, font: ImageFont.ImageFont, padding_x: int = 14, padding_y: int = 8) -> Tuple[int, int]:
    dummy = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 2 * padding_x
    h = bbox[3] - bbox[1] + 2 * padding_y
    return w, h


def draw_top_right_text(
    base_rgb: np.ndarray,
    text: str,
    right_margin: int,
    top_margin: int,
    font: ImageFont.ImageFont,
    fill_rgba: Tuple[int, int, int, int],
) -> np.ndarray:
    pil = Image.fromarray(base_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    x = pil.size[0] - right_margin - w
    draw.text((x, top_margin), text, font=font, fill=fill_rgba)
    merged = Image.alpha_composite(pil, overlay).convert("RGB")
    return np.array(merged)


# -----------------------------
# 图像排版
# -----------------------------
def make_placeholder(width: int, height: int, title: str, path_text: str) -> np.ndarray:
    canvas = np.full((height, width, 3), DEFAULT_PANEL_BG, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (85, 85, 90), 2)
    cv2.line(canvas, (0, 0), (width - 1, height - 1), (70, 70, 74), 2)
    cv2.line(canvas, (width - 1, 0), (0, height - 1), (70, 70, 74), 2)

    rgb = bgr_to_rgb(canvas)
    title_font = get_font(36, bold=True)
    body_font = get_font(20, bold=False)
    rgb = draw_rounded_text_box(
        rgb,
        (36, 36),
        [
            (title, (255, 255, 255, 255), title_font),
            (path_text, (200, 200, 205, 255), body_font),
        ],
        padding=16,
        line_gap=8,
        radius=18,
        box_fill=(10, 10, 10, 170),
        box_outline=(255, 255, 255, 30),
    )
    return rgb_to_bgr(rgb)



def fit_image_to_box(img_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    src_h, src_w = img_bgr.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return make_placeholder(target_w, target_h, "Invalid image", "image shape is empty")

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)

    canvas = np.full((target_h, target_w, 3), DEFAULT_PANEL_BG, dtype=np.uint8)
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    cv2.rectangle(canvas, (0, 0), (target_w - 1, target_h - 1), DEFAULT_BORDER, 2)
    return canvas



def load_or_placeholder(path: str, target_w: int, target_h: int, title: str, strict: bool) -> np.ndarray:
    img = imread_bgr(path)
    if img is not None:
        return fit_image_to_box(img, target_w, target_h)
    if strict:
        raise FileNotFoundError(f"图片不存在或读取失败: {path}")
    return make_placeholder(target_w, target_h, f"{title} missing", path)



def infer_box_height(records: List[Dict[str, Any]], key: str, panel_width: int, fallback_ratio: float) -> int:
    for rec in records:
        img = imread_bgr(rec.get(key, ""))
        if img is not None:
            h, w = img.shape[:2]
            return max(120, int(round(panel_width * h / max(w, 1))))
    return max(120, int(round(panel_width * fallback_ratio)))



def compose_frame(
    rec: Dict[str, Any],
    video_name: str,
    frame_index: int,
    total_frames: int,
    panel_width: int,
    outer_box_h: int,
    inner_box_h: int,
    strict: bool,
    digits: int,
) -> np.ndarray:
    margin_x = 30
    header_h = 118
    gap = 24
    bottom_margin = 24

    outer = load_or_placeholder(
        rec.get("inputs_img_path", ""), panel_width, outer_box_h, "Outside image", strict
    )
    inner = load_or_placeholder(
        rec.get("inner_img_path", ""), panel_width, inner_box_h, "Inside image", strict
    )

    canvas_w = panel_width + margin_x * 2
    canvas_h = header_h + outer_box_h + gap + inner_box_h + bottom_margin
    canvas = np.full((canvas_h, canvas_w, 3), DEFAULT_BG, dtype=np.uint8)

    y_outer = header_h
    y_inner = header_h + outer_box_h + gap
    canvas[y_outer:y_outer + outer_box_h, margin_x:margin_x + panel_width] = outer
    canvas[y_inner:y_inner + inner_box_h, margin_x:margin_x + panel_width] = inner

    rgb = bgr_to_rgb(canvas)

    title_font = get_font(24, bold=True)
    info_font = get_font(34, bold=True)
    badge_font = get_font(22, bold=True)
    meta_font = get_font(22, bold=False)

    pred_txt = format_scalar(rec.get("pred"), digits)
    gt_txt = format_scalar(rec.get("gt"), digits)
    speed_txt = format_scalar(rec.get("speed"), digits)
    gt_positive = is_positive_gt(rec.get("gt"))

    info_lines = [
        (f"Pred   {pred_txt}", (240, 240, 240, 255), info_font),
        (f"GT     {gt_txt}", (255, 110, 110, 255) if gt_positive else (240, 240, 240, 255), info_font),
        (f"Speed  {speed_txt}", (240, 240, 240, 255), info_font),
    ]
    rgb = draw_rounded_text_box(
        rgb,
        (28, 22),
        info_lines,
        padding=18,
        line_gap=6,
        radius=22,
        box_fill=(12, 12, 12, 180),
        box_outline=(255, 255, 255, 35),
    )

    outside_badge_w, _ = get_badge_box_size("Outside View", badge_font)
    inside_badge_w, _ = get_badge_box_size("Inside View", badge_font)

    rgb = draw_small_badge(
        rgb,
        (margin_x + panel_width - outside_badge_w - 16, y_outer + 14),
        "Outside View",
        fill_rgba=(0, 0, 0, 150),
        text_rgba=(255, 255, 255, 255),
        font=badge_font,
    )
    rgb = draw_small_badge(
        rgb,
        (margin_x + panel_width - inside_badge_w - 16, y_inner + 14),
        "Inside View",
        fill_rgba=(0, 0, 0, 150),
        text_rgba=(255, 255, 255, 255),
        font=badge_font,
    )

    meta_text = f"{video_name}   |   frame {frame_index + 1:03d}/{total_frames:03d}"
    rgb = draw_top_right_text(
        rgb,
        meta_text,
        right_margin=24,
        top_margin=30,
        font=title_font,
        fill_rgba=(220, 220, 225, 240),
    )

    out = rgb_to_bgr(rgb)

    # subtle outer frame
    cv2.rectangle(out, (1, 1), (canvas_w - 2, canvas_h - 2), DEFAULT_BORDER, 2)

    if gt_positive:
        # 明显红框：多层叠加 + 更粗的边框
        for t in (8, 14, 20):
            cv2.rectangle(out, (t // 2, t // 2), (canvas_w - 1 - t // 2, canvas_h - 1 - t // 2), RED_BORDER, t)

    return out


# -----------------------------
# 分组与导出
# -----------------------------
def sort_records_in_group(records: List[Dict[str, Any]], sort_mode: str) -> List[Dict[str, Any]]:
    if sort_mode == "log_order":
        return sorted(records, key=lambda x: x["_log_index"])
    if sort_mode == "inner":
        return sorted(
            records,
            key=lambda x: (extract_frame_number(x.get("inner_img_path", "")), x["_log_index"]),
        )
    return sorted(
        records,
        key=lambda x: (extract_frame_number(x.get("inputs_img_path", "")), x["_log_index"]),
    )



def group_records(records: List[Dict[str, Any]], sort_mode: str) -> OrderedDict[str, List[Dict[str, Any]]]:
    groups: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        rec = dict(rec)
        rec["_log_index"] = idx
        key = get_video_group_key(rec.get("inputs_img_path", ""))
        if key not in groups:
            groups[key] = []
        groups[key].append(rec)

    for key in list(groups.keys()):
        groups[key] = sort_records_in_group(groups[key], sort_mode)
    return groups



def create_writer(path: str, width: int, height: int, fps: float, codec: str) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if writer.isOpened():
        return writer

    # fallback
    for alt in ["mp4v", "avc1"]:
        if alt == codec:
            continue
        fourcc = cv2.VideoWriter_fourcc(*alt)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"[WARN] codec={codec} 打不开，已自动回退到 {alt}")
            return writer

    raise RuntimeError(
        f"无法创建 VideoWriter: {path}。可尝试安装 ffmpeg 支持的 opencv，或切换 --codec mp4v"
    )



def export_group_video(
    group_key: str,
    records: List[Dict[str, Any]],
    output_dir: str,
    fps: float,
    panel_width: int,
    codec: str,
    strict: bool,
    digits: int,
) -> str:
    video_name = make_output_name(group_key)
    output_path = os.path.join(output_dir, f"{video_name}.mp4")

    outer_box_h = infer_box_height(records, "inputs_img_path", panel_width, fallback_ratio=9 / 16)
    inner_box_h = infer_box_height(records, "inner_img_path", panel_width, fallback_ratio=9 / 16)

    sample = compose_frame(
        records[0],
        video_name=video_name,
        frame_index=0,
        total_frames=len(records),
        panel_width=panel_width,
        outer_box_h=outer_box_h,
        inner_box_h=inner_box_h,
        strict=strict,
        digits=digits,
    )
    h, w = sample.shape[:2]
    writer = create_writer(output_path, w, h, fps, codec)

    try:
        writer.write(sample)
        for idx, rec in enumerate(records[1:], start=1):
            frame = compose_frame(
                rec,
                video_name=video_name,
                frame_index=idx,
                total_frames=len(records),
                panel_width=panel_width,
                outer_box_h=outer_box_h,
                inner_box_h=inner_box_h,
                strict=strict,
                digits=digits,
            )
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
    finally:
        writer.release()

    return output_path



def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    records = load_log(args.log)
    if not records:
        raise ValueError("log 为空，未读取到任何记录")

    required_keys = {"inner_img_path", "inputs_img_path", "pred", "gt", "speed"}
    missing_keys = required_keys - set(records[0].keys())
    if missing_keys:
        print(f"[WARN] 第一条记录缺少字段: {sorted(missing_keys)}；脚本会尽量兼容处理")

    groups = group_records(records, args.sort_mode)
    print(f"[INFO] 总帧数: {len(records)}")
    print(f"[INFO] 分组后视频数: {len(groups)}")

    generated = []
    for idx, (group_key, group_records_list) in enumerate(groups.items(), start=1):
        print(
            f"[INFO] ({idx}/{len(groups)}) 正在导出: {make_output_name(group_key)} | "
            f"frames={len(group_records_list)}"
        )
        out_path = export_group_video(
            group_key=group_key,
            records=group_records_list,
            output_dir=args.output_dir,
            fps=args.fps,
            panel_width=args.panel_width,
            codec=args.codec,
            strict=args.strict,
            digits=args.pred_digits,
        )
        generated.append(out_path)
        print(f"[OK] 已写出: {out_path}")

    print("\n[FINISH] 全部导出完成：")
    for p in generated:
        print(p)


if __name__ == "__main__":
    main()
