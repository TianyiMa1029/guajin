
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把推理日志里的 inner/outer 图像路径解析出来，拼成上下布局并写成多条 mp4。

功能：
1. 以 "==> this batch result:" 作为一帧结束标记解析日志
2. 从日志中提取：
   - inner_img_path
   - inputs_img_path
   - pred / gt
   - speed / acc
3. 按 inputs_img_path 对应的视频目录自动分组
4. 每组按帧号顺序写成一个 mp4
5. 车外图 / 车内图上下拼接（可配置谁在上面）
6. 在左上角信息栏写 pred/gt、speed、acc（已去掉 tensor(...)）
7. gt == 1 的帧会加明显红框
8. 字体、间距、信息栏、分隔线做了较美观的默认样式

示例：
python visualize_log_to_mp4.py \
    --log /path/to/log.txt \
    --out-dir ./vis_videos \
    --fps 10

如果想让 inner 在上面：
python visualize_log_to_mp4.py --log /path/to/log.txt --out-dir ./vis_videos --top-image inner

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


BATCH_MARKER = "==> this batch result:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize inference log into multi mp4 videos.")
    parser.add_argument("--log", required=True, help="推理日志 txt 路径")
    parser.add_argument("--out-dir", required=True, help="输出 mp4 目录")
    parser.add_argument("--fps", type=float, default=10.0, help="输出视频 fps，默认 10")
    parser.add_argument(
        "--top-image",
        choices=["outer", "inner"],
        default="outer",
        help="上下拼接时谁在上面，默认 outer",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=None,
        help="统一输出宽度；默认使用该视频首个可读外图宽度",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="cv2.VideoWriter_fourcc codec，默认 mp4v",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=8,
        help="上下两张图之间的间隔像素，默认 8",
    )
    parser.add_argument(
        "--border-thickness",
        type=int,
        default=12,
        help="gt==1 时红框粗细，默认 12",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果 mp4 已存在则跳过",
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
    return parser.parse_args()


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
        inner = s[len("tensor("):-1].strip()
        inner = split_top_level_expr(inner)
    else:
        inner = s
    try:
        return ast.literal_eval(inner)
    except Exception:
        return inner


def squeeze_singletons(x: Any) -> Any:
    while isinstance(x, list) and len(x) == 1:
        x = x[0]
    if isinstance(x, list):
        return [squeeze_singletons(v) for v in x]
    return x


def extract_scalar(x: Any) -> Optional[float]:
    x = squeeze_singletons(x)
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, list) and len(x) == 1 and isinstance(x[0], (int, float)):
        return float(x[0])
    return None


def format_number(x: Any) -> str:
    if isinstance(x, bool):
        return str(int(x))
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.4f}"
    return str(x)


def format_tensor_value(x: Any) -> str:
    x = squeeze_singletons(x)
    if isinstance(x, list):
        if any(isinstance(v, list) for v in x):
            return " ; ".join(format_tensor_value(v) if isinstance(v, list) else format_number(v) for v in x)
        return "[" + ", ".join(format_number(v) for v in x) + "]"
    return format_number(x)


def safe_literal_list(s: str) -> List[Any]:
    try:
        value = ast.literal_eval(s)
        if isinstance(value, list):
            return value
        return [value]
    except Exception as exc:
        raise ValueError(f"无法解析列表: {s}") from exc


def parse_block(block: str, seq_idx: int) -> Optional[Dict[str, Any]]:
    inner_match = re.search(r"inner_img_path:\s*(\[[^\n]+\])", block)
    outer_match = re.search(r"inputs_img_path\s*(\[[^\n]+\])", block)
    pred_gt_match = re.search(
        r"pred/gt:\s*(tensor\(.*?\))\s*/\s*(tensor\(.*?\))\s*speed/acc:",
        block,
        flags=re.S,
    )
    speed_acc_match = re.search(
        r"speed/acc:\s*(tensor\(.*?\))\s*/\s*(tensor\(.*?\))(?=\s*(?:use DDP mode|validating|$))",
        block,
        flags=re.S,
    )

    if not (inner_match and outer_match and pred_gt_match and speed_acc_match):
        return None

    inner_paths = safe_literal_list(inner_match.group(1))
    outer_paths = safe_literal_list(outer_match.group(1))
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
        "pred_text": format_tensor_value(pred),
        "gt_text": format_tensor_value(gt),
        "speed_text": format_tensor_value(speed),
        "acc_text": format_tensor_value(acc),
        "is_positive": is_positive,
    }


def parse_log(log_path: str) -> List[Dict[str, Any]]:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    parts = text.split(BATCH_MARKER)
    frames: List[Dict[str, Any]] = []
    for i, block in enumerate(parts[1:], start=1):
        item = parse_block(block, i)
        if item is not None:
            frames.append(item)
    return frames


def group_frames(frames: Sequence[Dict[str, Any]]) -> "OrderedDict[str, List[Dict[str, Any]]]":
    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for item in frames:
        grouped.setdefault(item["video_dir"], []).append(item)
    for _, items in grouped.items():
        items.sort(key=lambda x: (x["frame_idx"], x["seq_idx"]))
    return grouped


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            ]
        )
    for fp in candidates:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size=size)
    return ImageFont.load_default()


def open_image_safe(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
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
        raise RuntimeError("该视频所有内/外图都无法读取。")
    if outer_size is None:
        outer_size = inner_size
    if inner_size is None:
        inner_size = outer_size
    assert outer_size is not None and inner_size is not None
    return outer_size, inner_size


def contain_and_pad(img: Image.Image, target_size: Tuple[int, int], bg=(0, 0, 0)) -> Image.Image:
    fitted = ImageOps.contain(img, target_size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, bg)
    x = (target_size[0] - fitted.width) // 2
    y = (target_size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def build_placeholder(target_size: Tuple[int, int], title: str, path: str) -> Image.Image:
    canvas = Image.new("RGB", target_size, (28, 28, 32))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(max(18, min(34, target_size[0] // 28)), bold=True)
    text_font = load_font(max(14, min(24, target_size[0] // 42)), bold=False)

    pad = 24
    draw.rounded_rectangle(
        (16, 16, target_size[0] - 16, target_size[1] - 16),
        radius=18,
        outline=(220, 80, 80),
        width=3,
        fill=(42, 42, 48),
    )
    draw.text((pad, pad), title, font=title_font, fill=(255, 225, 225))

    wrapped = wrap_text_by_pixels(draw, path, text_font, max_width=target_size[0] - 2 * pad)
    y = pad + text_height(draw, title_font) + 16
    for line in wrapped:
        draw.text((pad, y), line, font=text_font, fill=(230, 230, 235))
        y += text_height(draw, text_font) + 6
    return canvas


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

    # 对特别长、没有空格的 token 再做一次按字符拆分
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


def build_info_lines(frame: Dict[str, Any]) -> List[str]:
    return [
        f"pred/gt : {frame['pred_text']} / {frame['gt_text']}",
        f"speed   : {frame['speed_text']}",
        f"acc     : {frame['acc_text']}",
    ]


def compute_header_height(
    frames: Sequence[Dict[str, Any]],
    width: int,
    left_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    meta_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> int:
    dummy = Image.new("RGB", (width, 200))
    draw = ImageDraw.Draw(dummy)

    left_pad = 22
    top_pad = 18
    right_block_width = min(360, max(180, width // 4))
    usable_left_width = width - left_pad * 2 - right_block_width

    line_gap = 7
    max_h = 0
    for frame in frames:
        lines: List[str] = []
        for raw in build_info_lines(frame):
            lines.extend(wrap_text_by_pixels(draw, raw, left_font, usable_left_width))
        if not lines:
            lines = [""]
        h = top_pad * 2 + len(lines) * text_height(draw, left_font) + (len(lines) - 1) * line_gap
        meta_h = top_pad * 2 + 2 * text_height(draw, meta_font) + 8
        max_h = max(max_h, h, meta_h)
    return max(92, max_h)


def shorten_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    left = (max_chars - 3) // 2
    right = max_chars - 3 - left
    return text[:left] + "..." + text[-right:]


def draw_header(
    canvas: Image.Image,
    frame: Dict[str, Any],
    header_h: int,
    left_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    meta_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    frame_pos: int,
    total_frames: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    width = canvas.width

    bg = (18, 22, 28)
    draw.rectangle((0, 0, width, header_h), fill=bg)

    # 左侧信息
    left_pad = 22
    top_pad = 18
    right_block_width = min(360, max(180, width // 4))
    usable_left_width = width - left_pad * 2 - right_block_width
    line_gap = 7

    lines: List[str] = []
    for raw in build_info_lines(frame):
        lines.extend(wrap_text_by_pixels(draw, raw, left_font, usable_left_width))

    y = top_pad
    text_color = (242, 244, 248)
    for i, line in enumerate(lines):
        line_color = (255, 234, 180) if i == 0 else text_color
        draw.text((left_pad, y), line, font=left_font, fill=line_color)
        y += text_height(draw, left_font) + line_gap

    # 右侧 meta
    meta_x = width - right_block_width + 22
    meta_y = top_pad
    video_name = shorten_middle(frame["video_name"], 32)
    draw.text((meta_x, meta_y), video_name, font=meta_font, fill=(170, 180, 195))
    meta_y += text_height(draw, meta_font) + 8
    draw.text(
        (meta_x, meta_y),
        f"frame {frame_pos:03d}/{total_frames:03d}",
        font=meta_font,
        fill=(210, 216, 226),
    )

    # header 底部分隔线
    draw.line((0, header_h - 1, width, header_h - 1), fill=(56, 62, 72), width=1)


def draw_view_tag(img: Image.Image, label: str) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    font = load_font(max(16, min(26, img.width // 55)), bold=True)
    pad_x = 14
    pad_y = 8
    text_w = text_width(draw, label, font)
    text_h = text_height(draw, font)
    box_w = text_w + pad_x * 2
    box_h = text_h + pad_y * 2
    x1, y1 = 18, 18
    x2, y2 = x1 + box_w, y1 + box_h
    draw.rounded_rectangle((x1, y1, x2, y2), radius=12, fill=(10, 10, 10, 170))
    draw.text((x1 + pad_x, y1 + pad_y - 1), label, font=font, fill=(250, 250, 252, 255))


def add_red_border(img: Image.Image, thickness: int) -> None:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    color = (230, 35, 35)
    for i in range(thickness):
        draw.rectangle((i, i, w - 1 - i, h - 1 - i), outline=color)


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def ensure_unique_output_path(out_dir: str, base_name: str) -> str:
    candidate = Path(out_dir) / f"{base_name}.mp4"
    if not candidate.exists():
        return str(candidate)
    idx = 2
    while True:
        candidate = Path(out_dir) / f"{base_name}_{idx}.mp4"
        if not candidate.exists():
            return str(candidate)
        idx += 1


def prepare_image(
    path: str,
    target_size: Tuple[int, int],
    missing_title: str,
) -> Image.Image:
    img = open_image_safe(path)
    if img is None:
        return build_placeholder(target_size, missing_title, path)
    return contain_and_pad(img, target_size, bg=(0, 0, 0))


def render_video_group(
    frames: Sequence[Dict[str, Any]],
    out_path: str,
    fps: float,
    top_image: str,
    target_width: Optional[int],
    gap: int,
    border_thickness: int,
    codec: str,
) -> None:
    if not frames:
        return

    outer_ref, inner_ref = get_reference_sizes(frames)
    out_w_ref, out_h_ref = outer_ref
    in_w_ref, in_h_ref = inner_ref

    final_width = target_width if target_width is not None else out_w_ref
    if final_width <= 0:
        raise ValueError("target width 必须 > 0")

    outer_box = (final_width, max(1, round(out_h_ref * final_width / out_w_ref)))
    inner_box = (final_width, max(1, round(in_h_ref * final_width / in_w_ref)))

    left_font = load_font(max(20, min(34, final_width // 45)), bold=False)
    meta_font = load_font(max(16, min(24, final_width // 62)), bold=False)
    header_h = compute_header_height(frames, final_width, left_font, meta_font)

    final_height = header_h + outer_box[1] + gap + inner_box[1]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (final_width, final_height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件：{out_path}，请检查 codec={codec} 是否可用")

    for idx, frame in enumerate(frames, start=1):
        outer_img = prepare_image(frame["outer_path"], outer_box, "Missing OUTER image")
        inner_img = prepare_image(frame["inner_path"], inner_box, "Missing INNER image")

        if top_image == "outer":
            top_img, top_label = outer_img, "OUTER"
            bottom_img, bottom_label = inner_img, "INNER"
        else:
            top_img, top_label = inner_img, "INNER"
            bottom_img, bottom_label = outer_img, "OUTER"

        draw_view_tag(top_img, top_label)
        draw_view_tag(bottom_img, bottom_label)

        canvas = Image.new("RGB", (final_width, final_height), (0, 0, 0))
        draw_header(canvas, frame, header_h, left_font, meta_font, idx, len(frames))

        canvas.paste(top_img, (0, header_h))
        if gap > 0:
            gap_y1 = header_h + outer_box[1]
            gap_y2 = gap_y1 + gap
            ImageDraw.Draw(canvas).rectangle((0, gap_y1, final_width, gap_y2), fill=(30, 34, 40))
        canvas.paste(bottom_img, (0, header_h + outer_box[1] + gap))

        if frame["is_positive"]:
            add_red_border(canvas, border_thickness)

        writer.write(pil_to_bgr(canvas))

    writer.release()


def print_summary(grouped: "OrderedDict[str, List[Dict[str, Any]]]") -> None:
    total_frames = sum(len(v) for v in grouped.values())
    total_positive = sum(sum(1 for item in v if item["is_positive"]) for v in grouped.values())
    print(f"解析完成：共 {len(grouped)} 条视频，{total_frames} 帧，gt==1 的帧共 {total_positive} 帧")
    for video_dir, items in grouped.items():
        pos = sum(1 for x in items if x["is_positive"])
        print(f"  - {Path(video_dir).name}: {len(items)} 帧, positive={pos}")


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
        out_path = ensure_unique_output_path(args.out_dir, base_name)

        if args.skip_existing:
            expected = str(Path(args.out_dir) / f"{base_name}.mp4")
            if os.path.exists(expected):
                print(f"[skip] {expected}")
                continue

        print(f"[write] {base_name} -> {out_path}")
        try:
            render_video_group(
                frames=items,
                out_path=out_path,
                fps=args.fps,
                top_image=args.top_image,
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
