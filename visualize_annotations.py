
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量可视化 2D 框标注结果，支持两种 JSON 结构：

格式 1（children 风格）
- 根节点包含 `children`
- 每个框使用 `mincol / minrow / maxcol / maxrow`
- 类别字段通常为 `identity`
- 示例见 demo1.json

格式 2（FeatureCollection 风格）
- 根节点包含 `markResult.features`
- 每个框使用 `geometry.coordinates`
- 类别字段通常为 `properties.content.label`
- 图像尺寸通常在根节点 `info.width / info.height`
- 示例见 demo2.json

主要特性
- 支持按“团队文件夹”批量处理
- 自动识别两种 JSON 格式
- 同一类别在不同团队中使用统一配色
- 配色采用 NPG（Nature Publishing Group）风格色板
- 输出图片与 JSON 同名（例如 a.json -> a.png）
- 结果分别保存在以团队名命名的文件夹中
- 右侧附带验收信息面板，便于 QA 快速核验
- 若原图缺失，会自动生成占位底图，并基于标注范围推断画布大小

推荐依赖
    pip install pillow

示例
    python visualize_annotations.py \
        --team_inputs /data/team_a_jsons /data/team_b_jsons \
        --team_names TeamA TeamB \
        --image_root /data/images \
        --output_root /data/vis_results

如果不传 --team_names，则默认使用输入文件夹名作为团队名。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image, ImageColor, ImageDraw, ImageFont

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Nature Publishing Group (NPG) 风格常用配色
NPG_PALETTE = [
    "#E64B35",  # vermilion
    "#4DBBD5",  # cyan
    "#00A087",  # green
    "#3C5488",  # blue
    "#F39B7F",  # peach
    "#8491B4",  # lavender blue
    "#91D1C2",  # mint
    "#DC0000",  # red
    "#7E6148",  # brown
    "#B09C85",  # beige
]

FONT_REGULAR_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.otf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
FONT_BOLD_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansSC-Bold.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansSC-Bold.otf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


@dataclass
class Box:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    source_id: Optional[str] = None

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


@dataclass
class Record:
    team_root: Path
    team_name: str
    json_path: Path
    rel_json_path: Path
    format_name: str
    image_name: Optional[str]
    image_width: Optional[int]
    image_height: Optional[int]
    boxes: list[Box]
    warning: str = ""


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def round_up(v: int, base: int) -> int:
    if base <= 0:
        return v
    return int(math.ceil(v / base) * base)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = FONT_BOLD_CANDIDATES if bold else FONT_REGULAR_CANDIDATES
    for fp in candidates:
        path = Path(fp)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def rgb(hex_color: str) -> tuple[int, int, int]:
    return ImageColor.getrgb(hex_color)


def rgba(color: tuple[int, int, int], alpha: int) -> tuple[int, int, int, int]:
    return (int(color[0]), int(color[1]), int(color[2]), int(alpha))


def mix(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = clamp(t, 0.0, 1.0)
    return (
        int(round(c1[0] * (1 - t) + c2[0] * t)),
        int(round(c1[1] * (1 - t) + c2[1] * t)),
        int(round(c1[2] * (1 - t) + c2[2] * t)),
    )


def darken(color: tuple[int, int, int], strength: float = 0.22) -> tuple[int, int, int]:
    return mix(color, (0, 0, 0), strength)


def lighten(color: tuple[int, int, int], strength: float = 0.82) -> tuple[int, int, int]:
    return mix(color, (255, 255, 255), strength)


def luminance(color: tuple[int, int, int]) -> float:
    def _to_linear(v: int) -> float:
        x = v / 255.0
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4

    r, g, b = (_to_linear(v) for v in color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def text_color_on(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    return (255, 255, 255) if luminance(bg) < 0.28 else (25, 25, 25)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if not text:
        return 0, 0
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def ensure_int_bbox(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


def add_overlay(base: Image.Image, overlay: Image.Image) -> Image.Image:
    return Image.alpha_composite(base.convert("RGBA"), overlay.convert("RGBA"))


def draw_shadowed_rounded_rect(
    base: Image.Image,
    bbox: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    outline: Optional[tuple[int, int, int, int]] = None,
    width: int = 1,
    radius: int = 12,
    shadow_offset: int = 4,
    shadow_alpha: int = 28,
) -> None:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    sx1, sy1, sx2, sy2 = bbox
    shadow_box = (sx1 + shadow_offset, sy1 + shadow_offset, sx2 + shadow_offset, sy2 + shadow_offset)
    draw.rounded_rectangle(shadow_box, radius=radius, fill=(0, 0, 0, shadow_alpha))
    draw.rounded_rectangle(bbox, radius=radius, fill=fill, outline=outline, width=width)
    composed = add_overlay(base, overlay)
    base.paste(composed)


def draw_text(
    img: Image.Image,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int] | tuple[int, int, int],
) -> None:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.text(xy, text, font=font, fill=fill)
    composed = add_overlay(img, overlay)
    img.paste(composed)


def build_image_index(image_root: Optional[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    if image_root is None or not image_root.exists():
        return index
    for p in image_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            continue
        index[p.name.lower()].append(p)
        index[p.stem.lower()].append(p)
    return index


def collect_xy_from_coordinates(coords: Any, xs: list[float], ys: list[float]) -> None:
    if isinstance(coords, (list, tuple)):
        if len(coords) == 2 and all(isinstance(v, (int, float)) for v in coords):
            xs.append(float(coords[0]))
            ys.append(float(coords[1]))
            return
        for item in coords:
            collect_xy_from_coordinates(item, xs, ys)


def parse_children_format(data: dict[str, Any]) -> tuple[list[Box], Optional[str], Optional[int], Optional[int], str]:
    boxes: list[Box] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if (
                node.get("type") == "rect"
                and all(k in node for k in ("mincol", "minrow", "maxcol", "maxrow"))
            ):
                try:
                    x1 = float(node["mincol"])
                    y1 = float(node["minrow"])
                    x2 = float(node["maxcol"])
                    y2 = float(node["maxrow"])
                except Exception:
                    x1 = y1 = x2 = y2 = 0.0
                x1, x2 = sorted((x1, x2))
                y1, y2 = sorted((y1, y2))
                if x2 > x1 and y2 > y1:
                    label = str(node.get("identity") or node.get("label") or "unknown")
                    sid = node.get("trackid") or node.get("uniqueid")
                    boxes.append(Box(label=label, x1=x1, y1=y1, x2=x2, y2=y2, source_id=str(sid) if sid is not None else None))
            for child in node.get("children", []):
                walk(child)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data.get("children", []))
    image_name = data.get("imagename")
    return boxes, str(image_name) if image_name else None, None, None, "children_rect"


def parse_featurecollection_format(data: dict[str, Any]) -> tuple[list[Box], Optional[str], Optional[int], Optional[int], str]:
    boxes: list[Box] = []
    mark_result = data.get("markResult") or {}
    features = mark_result.get("features") or []
    for feat in features:
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties") or {}
        label = (
            props.get("content", {}).get("label")
            or props.get("label")
            or props.get("contentLabel")
            or "unknown"
        )
        sid = props.get("objectId") or props.get("layerId")
        geometry = feat.get("geometry") or {}
        coords = geometry.get("coordinates")
        xs: list[float] = []
        ys: list[float] = []
        collect_xy_from_coordinates(coords, xs, ys)
        if not xs or not ys:
            continue
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        if x2 > x1 and y2 > y1:
            boxes.append(Box(label=str(label), x1=x1, y1=y1, x2=x2, y2=y2, source_id=str(sid) if sid is not None else None))

    info = data.get("info") or mark_result.get("info") or {}
    width = info.get("width")
    height = info.get("height")
    image_name = data.get("imagename") or data.get("imageName") or data.get("image_name")
    width = int(width) if isinstance(width, (int, float)) else None
    height = int(height) if isinstance(height, (int, float)) else None
    return boxes, str(image_name) if image_name else None, width, height, "markResult_features"


def detect_and_parse(data: dict[str, Any]) -> tuple[list[Box], Optional[str], Optional[int], Optional[int], str]:
    if isinstance(data, dict) and "children" in data and isinstance(data.get("children"), list):
        return parse_children_format(data)
    if isinstance(data, dict) and "markResult" in data:
        return parse_featurecollection_format(data)
    raise ValueError("Unsupported JSON schema: 无法识别的标注结构。")


def parse_record(json_path: Path, team_root: Path, team_name: str) -> Record:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    boxes, image_name, width, height, fmt = detect_and_parse(data)
    rel_json_path = json_path.relative_to(team_root) if team_root.is_dir() else Path(json_path.name)

    warning = ""
    if not boxes:
        warning = "No valid boxes found"

    return Record(
        team_root=team_root,
        team_name=team_name,
        json_path=json_path,
        rel_json_path=rel_json_path,
        format_name=fmt,
        image_name=image_name,
        image_width=width,
        image_height=height,
        boxes=boxes,
        warning=warning,
    )


def infer_canvas_size(record: Record) -> tuple[int, int]:
    if record.image_width and record.image_height:
        return int(record.image_width), int(record.image_height)

    if record.boxes:
        max_x = max(b.x2 for b in record.boxes)
        max_y = max(b.y2 for b in record.boxes)
        min_x = min(b.x1 for b in record.boxes)
        min_y = min(b.y1 for b in record.boxes)
        box_w = max_x - min_x
        box_h = max_y - min_y
        w = int(max(max_x + max(48, 0.08 * box_w), 640))
        h = int(max(max_y + max(48, 0.10 * box_h), 480))
        return round_up(w, 32), round_up(h, 32)

    return 1280, 720


def make_placeholder_image(width: int, height: int) -> Image.Image:
    bg = Image.new("RGBA", (width, height), (250, 250, 248, 255))
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # subtle grid
    major = max(64, int(round(min(width, height) / 8 / 16) * 16))
    minor = max(32, major // 2)
    for x in range(0, width, minor):
        a = 18 if x % major else 34
        draw.line([(x, 0), (x, height)], fill=(60, 60, 60, a), width=1)
    for y in range(0, height, minor):
        a = 18 if y % major else 34
        draw.line([(0, y), (width, y)], fill=(60, 60, 60, a), width=1)

    # diagonal watermark-like hatch
    step = max(72, min(width, height) // 8)
    for k in range(-height, width, step):
        draw.line([(k, 0), (k + height, height)], fill=(120, 120, 120, 10), width=2)

    bg = add_overlay(bg, overlay)

    label_font = load_font(max(16, min(24, width // 50)), bold=True)
    sub_font = load_font(max(12, min(18, width // 72)), bold=False)

    card_w = min(460, int(width * 0.42))
    card_h = 88
    x1 = 24
    y1 = 24
    x2 = x1 + card_w
    y2 = y1 + card_h
    draw_shadowed_rounded_rect(
        bg,
        (x1, y1, x2, y2),
        fill=(255, 255, 255, 228),
        outline=(220, 220, 220, 255),
        width=1,
        radius=18,
        shadow_offset=5,
        shadow_alpha=35,
    )
    draw_text(bg, (x1 + 20, y1 + 18), "IMAGE NOT FOUND", label_font, (33, 33, 33, 255))
    draw_text(bg, (x1 + 20, y1 + 50), "Using inferred canvas from annotation geometry.", sub_font, (78, 78, 78, 255))
    return bg


def try_resolve_image(record: Record, image_index: dict[str, list[Path]]) -> Optional[Path]:
    candidates: list[Path] = []

    # 1) explicit image name from JSON
    if record.image_name:
        img_name = Path(record.image_name)
        candidates.extend([
            record.json_path.parent / img_name,
            record.team_root / img_name,
            record.team_root.parent / img_name,
        ])
        if img_name.name.lower() in image_index:
            candidates.extend(image_index[img_name.name.lower()])

    # 2) same stem with common image extensions
    for ext in SUPPORTED_IMAGE_EXTS:
        candidates.extend([
            record.json_path.with_suffix(ext),
            record.json_path.parent / f"{record.json_path.stem}{ext}",
            record.team_root / f"{record.json_path.stem}{ext}",
            record.team_root.parent / f"{record.json_path.stem}{ext}",
        ])

    # 3) image index by stem
    if record.json_path.stem.lower() in image_index:
        candidates.extend(image_index[record.json_path.stem.lower()])

    seen = set()
    uniq: list[Path] = []
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)

    for p in uniq:
        if p.exists() and p.is_file():
            return p
    return None


def assign_label_colors(labels: Iterable[str]) -> dict[str, tuple[int, int, int]]:
    label_list = sorted({str(x) for x in labels})
    color_map: dict[str, tuple[int, int, int]] = {}
    for i, label in enumerate(label_list):
        color_map[label] = rgb(NPG_PALETTE[i % len(NPG_PALETTE)])
    return color_map


def get_visual_params(image_w: int, image_h: int) -> dict[str, int]:
    long_edge = max(image_w, image_h)
    return {
        "box_width": int(clamp(round(long_edge / 700), 2, 6)),
        "font_label": int(clamp(round(long_edge / 85), 12, 24)),
        "font_small": int(clamp(round(long_edge / 125), 11, 18)),
        "font_panel_title": int(clamp(round(long_edge / 65), 16, 28)),
        "font_panel_body": int(clamp(round(long_edge / 95), 12, 20)),
        "radius": int(clamp(round(long_edge / 180), 8, 18)),
        "pad": int(clamp(round(long_edge / 160), 8, 18)),
        "badge_pad_x": int(clamp(round(long_edge / 220), 7, 14)),
        "badge_pad_y": int(clamp(round(long_edge / 360), 4, 10)),
    }


def fit_label_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_w: int,
) -> str:
    if max_w <= 8:
        return ""
    if text_size(draw, text, font)[0] <= max_w:
        return text
    ellipsis = "…"
    lo, hi = 0, len(text)
    best = ellipsis
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = text[:mid].rstrip() + ellipsis
        if text_size(draw, trial, font)[0] <= max_w:
            best = trial
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def draw_boxes_on_image(
    image: Image.Image,
    boxes: list[Box],
    color_map: dict[str, tuple[int, int, int]],
) -> Image.Image:
    img = image.convert("RGBA")
    w, h = img.size
    params = get_visual_params(w, h)
    label_font = load_font(params["font_label"], bold=True)
    small_font = load_font(max(params["font_small"] - 1, 10), bold=True)

    # draw larger boxes first so smaller boxes remain visible
    indexed_boxes = list(enumerate(boxes, start=1))
    draw_order = sorted(indexed_boxes, key=lambda x: x[1].area, reverse=True)

    for idx, box in draw_order:
        base_color = color_map.get(box.label, rgb(NPG_PALETTE[0]))
        outline = darken(base_color, 0.10)
        fill = rgba(lighten(base_color, 0.45), 74)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        x1, y1, x2, y2 = ensure_int_bbox(box.x1, box.y1, box.x2, box.y2)
        x1 = int(clamp(x1, 0, w - 1))
        y1 = int(clamp(y1, 0, h - 1))
        x2 = int(clamp(x2, x1 + 1, w))
        y2 = int(clamp(y2, y1 + 1, h))

        draw.rectangle((x1, y1, x2, y2), outline=outline + (255,), width=params["box_width"], fill=fill)

        composed = add_overlay(img, overlay)
        img.paste(composed)

    # labels in a second pass
    label_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label_overlay)

    for idx, box in indexed_boxes:
        base_color = color_map.get(box.label, rgb(NPG_PALETTE[0]))
        pill_color = darken(base_color, 0.02)
        pill_text_color = text_color_on(pill_color)

        x1, y1, x2, y2 = ensure_int_bbox(box.x1, box.y1, box.x2, box.y2)
        x1 = int(clamp(x1, 0, w - 1))
        y1 = int(clamp(y1, 0, h - 1))
        x2 = int(clamp(x2, x1 + 1, w))
        y2 = int(clamp(y2, y1 + 1, h))

        # 自动切换显示内容，避免小目标过于拥挤
        full_text = f"#{idx} {box.label}"
        compact_text = f"#{idx}"
        use_text = full_text if min(box.width, box.height) >= 28 and box.width >= 42 else compact_text

        font = label_font if use_text == full_text else small_font
        tw, th = text_size(label_draw, use_text, font)
        pad_x = params["badge_pad_x"]
        pad_y = params["badge_pad_y"]
        pill_w = tw + pad_x * 2
        pill_h = th + pad_y * 2
        radius = max(8, pill_h // 2)

        preferred_x = x1 + params["pad"]
        max_pill_w = max(28, min(pill_w, w - preferred_x - params["pad"]))
        use_text = fit_label_text(label_draw, use_text, font, max_pill_w - pad_x * 2)
        tw, th = text_size(label_draw, use_text, font)
        pill_w = tw + pad_x * 2
        pill_h = th + pad_y * 2
        radius = max(8, pill_h // 2)

        px = x1 + params["pad"]
        py = y1 - pill_h - params["pad"] // 2
        if py < 4:
            py = y1 + params["pad"] // 2
        if px + pill_w > w - 4:
            px = max(4, w - pill_w - 4)

        # shadow
        label_draw.rounded_rectangle(
            (px + 2, py + 2, px + pill_w + 2, py + pill_h + 2),
            radius=radius,
            fill=(0, 0, 0, 38),
        )
        label_draw.rounded_rectangle(
            (px, py, px + pill_w, py + pill_h),
            radius=radius,
            fill=rgba(pill_color, 236),
            outline=rgba(darken(base_color, 0.18), 255),
            width=1,
        )
        label_draw.text((px + pad_x, py + pad_y - 1), use_text, font=font, fill=rgba(pill_text_color, 255))

    img = add_overlay(img, label_overlay)
    return img


def draw_info_panel(
    canvas: Image.Image,
    panel_box: tuple[int, int, int, int],
    record: Record,
    image_status: str,
    color_map: dict[str, tuple[int, int, int]],
) -> None:
    x1, y1, x2, y2 = panel_box
    panel_w = x2 - x1
    panel_h = y2 - y1

    title_font = load_font(max(18, min(26, panel_w // 14)), bold=True)
    body_font = load_font(max(13, min(19, panel_w // 18)), bold=False)
    body_bold = load_font(max(13, min(19, panel_w // 18)), bold=True)
    tiny_font = load_font(max(11, min(15, panel_w // 22)), bold=False)

    # overall panel background
    draw_shadowed_rounded_rect(
        canvas,
        panel_box,
        fill=(255, 255, 255, 238),
        outline=(228, 228, 228, 255),
        width=1,
        radius=18,
        shadow_offset=6,
        shadow_alpha=28,
    )

    inner_x = x1 + 18
    inner_y = y1 + 18
    inner_w = panel_w - 36

    def card(height: int) -> tuple[int, int, int, int]:
        nonlocal inner_y
        box = (inner_x, inner_y, inner_x + inner_w, inner_y + height)
        inner_y += height + 14
        return box

    # File meta card
    file_card_h = 148
    c1 = card(file_card_h)
    accent = rgb(NPG_PALETTE[0])
    draw_shadowed_rounded_rect(
        canvas,
        c1,
        fill=(248, 249, 250, 255),
        outline=(230, 232, 235, 255),
        width=1,
        radius=14,
        shadow_offset=3,
        shadow_alpha=18,
    )

    # accent bar
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle((c1[0], c1[1], c1[2], c1[1] + 8), radius=14, fill=rgba(accent, 255))
    canvas.paste(add_overlay(canvas, overlay))

    draw_text(canvas, (c1[0] + 16, c1[1] + 22), record.team_name, title_font, (22, 22, 22, 255))
    draw_text(canvas, (c1[0] + 16, c1[1] + 58), record.json_path.name, body_bold, (44, 44, 44, 255))
    draw_text(canvas, (c1[0] + 16, c1[1] + 86), f"Format: {record.format_name}", body_font, (84, 84, 84, 255))
    draw_text(canvas, (c1[0] + 16, c1[1] + 110), image_status, tiny_font, (106, 106, 106, 255))

    # Stats card
    counts = Counter(b.label for b in record.boxes)
    stats_card_h = 120
    c2 = card(stats_card_h)
    draw_shadowed_rounded_rect(
        canvas,
        c2,
        fill=(248, 249, 250, 255),
        outline=(230, 232, 235, 255),
        width=1,
        radius=14,
        shadow_offset=3,
        shadow_alpha=18,
    )
    draw_text(canvas, (c2[0] + 16, c2[1] + 16), "Summary", body_bold, (38, 38, 38, 255))
    big_font = load_font(max(22, min(34, panel_w // 10)), bold=True)
    big_val = str(len(record.boxes))
    draw_text(canvas, (c2[0] + 16, c2[1] + 42), big_val, big_font, (33, 33, 33, 255))
    draw_text(canvas, (c2[0] + 16 + max(42, panel_w // 8), c2[1] + 52), "objects", body_font, (82, 82, 82, 255))
    draw_text(canvas, (c2[0] + 16, c2[1] + 84), f"{len(counts)} classes · indexed boxes for QA", tiny_font, (104, 104, 104, 255))

    # Label legend card
    remaining_h = y2 - inner_y - 18
    c3 = (inner_x, inner_y, inner_x + inner_w, y2 - 18)
    draw_shadowed_rounded_rect(
        canvas,
        c3,
        fill=(248, 249, 250, 255),
        outline=(230, 232, 235, 255),
        width=1,
        radius=14,
        shadow_offset=3,
        shadow_alpha=18,
    )
    draw_text(canvas, (c3[0] + 16, c3[1] + 16), "Labels", body_bold, (38, 38, 38, 255))

    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    available_w = c3[2] - c3[0] - 32
    available_h = c3[3] - c3[1] - 48
    row_h = 26
    cols = 1 if len(items) <= 10 else 2
    rows = int(math.ceil(len(items) / cols))
    if rows * row_h > available_h and len(items) > 0:
        row_h = max(18, available_h // rows)

    swatch = 12
    col_w = available_w // cols
    start_y = c3[1] + 46

    for idx, (label, count) in enumerate(items):
        col = idx // rows if cols > 1 else 0
        row = idx % rows if cols > 1 else idx
        lx = c3[0] + 16 + col * col_w
        ly = start_y + row * row_h
        if ly + row_h > c3[3] - 12:
            break

        color = color_map.get(label, rgb(NPG_PALETTE[0]))
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.rounded_rectangle((lx, ly + 4, lx + swatch, ly + 4 + swatch), radius=3, fill=rgba(color, 255))
        canvas.paste(add_overlay(canvas, overlay))

        label_text = fit_label_text(
            ImageDraw.Draw(canvas),
            f"{label}  {count}",
            body_font,
            col_w - swatch - 16,
        )
        draw_text(canvas, (lx + swatch + 10, ly), label_text, body_font, (64, 64, 64, 255))


def compose_visualization(
    record: Record,
    img: Image.Image,
    image_found: bool,
    color_map: dict[str, tuple[int, int, int]],
) -> Image.Image:
    annotated = draw_boxes_on_image(img, record.boxes, color_map)
    image_w, image_h = annotated.size

    outer_pad = 24
    panel_w = int(clamp(image_w * 0.24, 300, 420))
    min_canvas_h = 600
    canvas_w = image_w + panel_w + outer_pad * 3
    canvas_h = max(image_h + outer_pad * 2, min_canvas_h)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (248, 248, 246, 255))

    # image card
    img_x = outer_pad
    img_y = outer_pad
    draw_shadowed_rounded_rect(
        canvas,
        (img_x, img_y, img_x + image_w, img_y + image_h),
        fill=(255, 255, 255, 255),
        outline=(222, 222, 222, 255),
        width=1,
        radius=18,
        shadow_offset=6,
        shadow_alpha=24,
    )
    canvas.alpha_composite(annotated, dest=(img_x, img_y))

    # thin inner border
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle((img_x, img_y, img_x + image_w, img_y + image_h), radius=18, outline=(216, 216, 216, 255), width=1)
    canvas = add_overlay(canvas, overlay)

    # panel
    panel_x1 = img_x + image_w + outer_pad
    panel_y1 = outer_pad
    panel_x2 = panel_x1 + panel_w
    panel_y2 = canvas_h - outer_pad

    size_text = f"{image_w}×{image_h}"
    image_status = f"Image: {'found' if image_found else 'placeholder'} · {size_text}"
    draw_info_panel(canvas, (panel_x1, panel_y1, panel_x2, panel_y2), record, image_status, color_map)
    return canvas


def process_record(
    record: Record,
    output_root: Path,
    image_index: dict[str, list[Path]],
    color_map: dict[str, tuple[int, int, int]],
    overwrite: bool = True,
) -> dict[str, Any]:
    rel_parent = record.rel_json_path.parent
    out_dir = output_root / record.team_name / rel_parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{record.json_path.stem}.png"

    if out_path.exists() and not overwrite:
        return {
            "team_name": record.team_name,
            "json_path": str(record.json_path),
            "output_path": str(out_path),
            "format_name": record.format_name,
            "image_found": "",
            "num_boxes": len(record.boxes),
            "label_counts": json.dumps(dict(Counter(b.label for b in record.boxes)), ensure_ascii=False),
            "status": "skipped_exists",
            "warning": record.warning,
        }

    resolved = try_resolve_image(record, image_index)
    image_found = resolved is not None

    if image_found:
        try:
            base_img = Image.open(resolved).convert("RGBA")
        except Exception:
            image_found = False
            resolved = None
            width, height = infer_canvas_size(record)
            base_img = make_placeholder_image(width, height)
    else:
        width, height = infer_canvas_size(record)
        base_img = make_placeholder_image(width, height)

    vis = compose_visualization(record, base_img, image_found, color_map)
    vis.save(out_path)

    return {
        "team_name": record.team_name,
        "json_path": str(record.json_path),
        "output_path": str(out_path),
        "format_name": record.format_name,
        "image_found": str(resolved) if image_found and resolved else "",
        "num_boxes": len(record.boxes),
        "label_counts": json.dumps(dict(Counter(b.label for b in record.boxes)), ensure_ascii=False),
        "status": "ok",
        "warning": record.warning,
    }


def iter_json_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".json":
        return [path]
    if path.is_dir():
        return sorted([p for p in path.rglob("*.json") if p.is_file()])
    return []


def normalize_team_inputs(team_inputs: list[Path], team_names: Optional[list[str]]) -> list[tuple[Path, str]]:
    results: list[tuple[Path, str]] = []
    if team_names and len(team_names) != len(team_inputs):
        raise ValueError("--team_names 的数量必须与 --team_inputs 一致。")
    for i, p in enumerate(team_inputs):
        if not p.exists():
            raise FileNotFoundError(f"输入路径不存在: {p}")
        name = team_names[i] if team_names else p.stem if p.is_file() else p.name
        results.append((p, name))
    return results


def write_summary(output_root: Path, team_name: str, rows: list[dict[str, Any]]) -> None:
    team_dir = output_root / team_name
    team_dir.mkdir(parents=True, exist_ok=True)
    summary_path = team_dir / "__summary.csv"
    fieldnames = [
        "team_name",
        "json_path",
        "output_path",
        "format_name",
        "image_found",
        "num_boxes",
        "label_counts",
        "status",
        "warning",
    ]
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(
    team_inputs: list[Path],
    team_names: Optional[list[str]],
    output_root: Path,
    image_root: Optional[Path],
    overwrite: bool = True,
) -> None:
    normalized = normalize_team_inputs(team_inputs, team_names)

    all_records: list[Record] = []
    for input_path, team_name in normalized:
        team_root = input_path.parent if input_path.is_file() else input_path
        json_files = iter_json_files(input_path)
        if not json_files:
            print(f"[WARN] 未在 {input_path} 下找到 JSON 文件。", file=sys.stderr)
            continue

        for json_path in json_files:
            try:
                rec = parse_record(json_path, team_root=team_root, team_name=team_name)
                all_records.append(rec)
            except Exception as e:
                print(f"[ERROR] 解析失败: {json_path} -> {e}", file=sys.stderr)

    if not all_records:
        raise RuntimeError("没有可处理的标注 JSON。")

    # 跨团队统一配色，保证相同 label 在所有输出中颜色一致
    all_labels = [b.label for rec in all_records for b in rec.boxes]
    color_map = assign_label_colors(all_labels)

    image_index = build_image_index(image_root)

    rows_by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, rec in enumerate(all_records, start=1):
        print(f"[{idx}/{len(all_records)}] {rec.team_name}: {rec.json_path}")
        try:
            row = process_record(
                record=rec,
                output_root=output_root,
                image_index=image_index,
                color_map=color_map,
                overwrite=overwrite,
            )
            rows_by_team[rec.team_name].append(row)
        except Exception as e:
            print(f"[ERROR] 可视化失败: {rec.json_path} -> {e}", file=sys.stderr)
            rows_by_team[rec.team_name].append(
                {
                    "team_name": rec.team_name,
                    "json_path": str(rec.json_path),
                    "output_path": "",
                    "format_name": rec.format_name,
                    "image_found": "",
                    "num_boxes": len(rec.boxes),
                    "label_counts": json.dumps(dict(Counter(b.label for b in rec.boxes)), ensure_ascii=False),
                    "status": "failed",
                    "warning": str(e),
                }
            )

    for team_name, rows in rows_by_team.items():
        write_summary(output_root, team_name, rows)

    print(f"\nDone. Outputs saved to: {output_root}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量可视化 2D 框标注结果（支持两种 JSON 格式）")
    parser.add_argument(
        "--team_inputs",
        nargs="+",
        required=True,
        help="每个标注团队的 JSON 文件夹路径，或单个 JSON 文件路径。",
    )
    parser.add_argument(
        "--team_names",
        nargs="*",
        default=None,
        help="可选：与 --team_inputs 一一对应的团队名称；不传则默认取文件夹名。",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="可选：原图根目录。若 JSON 中只记录了图片名，脚本会在这里递归查找。",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="输出根目录。结果会保存为 output_root/团队名/相对路径/json同名.png",
    )
    parser.add_argument(
        "--no_overwrite",
        action="store_true",
        help="若输出已存在，则跳过，不覆盖。",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    team_inputs = [Path(p).expanduser().resolve() for p in args.team_inputs]
    team_names = args.team_names
    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else None
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run(
        team_inputs=team_inputs,
        team_names=team_names,
        output_root=output_root,
        image_root=image_root,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
