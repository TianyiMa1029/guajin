#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析根目录下所有“样本子文件夹”，并输出 CSV / JSON 报表。

功能覆盖：
1. imgs 分辨率分类：720p / 1080p / other
2. scores_v0.json 中 state 统计：
   - 是否所有二级 key 的 state 都是 null
   - online_raw_with_anno.json 中“碰撞时刻帧数（填数字）”对应 jpg 的 state 是否为 null
   - 记录 state 为 null / 非 null / 缺失 key 的文件名列表
3. inner 类别：
   - 对碰撞帧对应 jpg，在 scores_v0.json -> inner_top3 里找最高分项
   - 判断最高分是否 > 0.7；若是则记录类别名
4. 事故分类字段：
   - 是否存在判不清
   - 司机特征
   - 视频中自车是否发生碰撞事故
   - 碰撞对象类别
   - 静态障碍物类别
   - 自车行车状态
   - 天气情况
   - 环境光线
   - 事故画面是否存在遮挡
5. 输出“分类组合 -> 子文件夹列表”聚合结果

输出文件：
- folder_level_report.csv      每个样本子文件夹一行
- combination_report.csv       各分类组合一行，附带子文件夹列表
- combination_report.json      与上面相同，但保留数组结构，方便程序二次处理

用法示例：
    python analyze_case_folders.py /path/to/root
    python analyze_case_folders.py /path/to/root --output-dir /path/to/output
    python analyze_case_folders.py /path/to/root --frame-offset 0 --inner-threshold 0.7

说明：
- 默认递归遍历 root 下所有目录，但只分析“像样本目录”的文件夹：
  只要该目录包含 imgs 子目录 / scores_v0.json / online_raw_with_anno.json 之一，就会被识别为样本目录。
- 碰撞帧默认按 6 位补零匹配：24 -> 000024.jpg；
  同时也会尝试匹配 24.jpg，以及和帧号数值相同的其他 key。
- CSV 使用 utf-8-sig 编码，方便直接用 Excel 打开。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANNO_FIELDS = [
    "是否存在判不清",
    "司机特征",
    "视频中自车是否发生碰撞事故",
    "碰撞对象类别",
    "静态障碍物类别",
    "自车行车状态",
    "天气情况",
    "环境光线",
    "事故画面是否存在遮挡",
]
COMBINATION_FIELDS = [
    "分辨率分类",
    "state全量分类",
    "碰撞state分类",
    "inner分类",
    "是否存在判不清",
    "司机特征",
    "视频中自车是否发生碰撞事故",
    "碰撞对象类别",
    "静态障碍物类别",
    "自车行车状态",
    "天气情况",
    "环境光线",
    "事故画面是否存在遮挡",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析样本子文件夹并输出 CSV / JSON 报表")
    parser.add_argument("root", help="根目录，脚本会递归查找其中的样本子文件夹")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录，默认写到 root/_analysis_output",
    )
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help="碰撞帧号偏移量。默认 0；如果你的帧号需要整体 +1 或 -1，可在这里调整",
    )
    parser.add_argument(
        "--inner-threshold",
        type=float,
        default=0.7,
        help="inner_top3 最高分阈值，默认 0.7",
    )
    return parser.parse_args()


def ensure_text(value: Any, *, empty_placeholder: str = "__EMPTY__") -> str:
    if value is None:
        return empty_placeholder
    if isinstance(value, str):
        text = value.strip()
        return text if text else empty_placeholder
    return str(value)


def stringify_jsonable(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def join_list(items: Sequence[str]) -> str:
    return "; ".join(items)


def load_json(path: Path) -> Any:
    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk"]
    last_error: Optional[Exception] = None
    for enc in encodings:
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    assert last_error is not None
    raise last_error


def is_case_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        names = {child.name for child in path.iterdir()}
    except Exception:
        return False
    return (
        "imgs" in names
        or "scores_v0.json" in names
        or "online_raw_with_anno.json" in names
    )


def find_case_dirs(root: Path) -> List[Path]:
    case_dirs: List[Path] = []
    for path in root.rglob("*"):
        if path == root:
            continue
        if is_case_dir(path):
            case_dirs.append(path)
    case_dirs.sort(key=lambda p: p.as_posix())
    return case_dirs


# ------------------------------
# 图片尺寸读取（尽量不依赖第三方库）
# ------------------------------

SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def _read_png_size(fp) -> Tuple[int, int]:
    fp.seek(0)
    header = fp.read(24)
    if len(header) < 24 or not header.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("不是合法 PNG")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def _read_jpeg_size(fp) -> Tuple[int, int]:
    fp.seek(0)
    if fp.read(2) != b"\xff\xd8":
        raise ValueError("不是合法 JPEG")

    while True:
        marker_prefix = fp.read(1)
        if not marker_prefix:
            break
        if marker_prefix != b"\xff":
            continue

        marker = fp.read(1)
        while marker == b"\xff":
            marker = fp.read(1)
        if not marker:
            break

        code = marker[0]

        # 没有 segment length 的 marker
        if code in (0xD8, 0xD9) or 0xD0 <= code <= 0xD7 or code == 0x01:
            continue

        seg_len_bytes = fp.read(2)
        if len(seg_len_bytes) != 2:
            break
        seg_len = struct.unpack(">H", seg_len_bytes)[0]
        if seg_len < 2:
            raise ValueError("JPEG segment 长度非法")

        if code in SOF_MARKERS:
            data = fp.read(seg_len - 2)
            if len(data) < 5:
                raise ValueError("JPEG SOF 数据不足")
            height = struct.unpack(">H", data[1:3])[0]
            width = struct.unpack(">H", data[3:5])[0]
            return width, height
        fp.seek(seg_len - 2, 1)

    raise ValueError("无法从 JPEG 中读取尺寸")


def get_image_size(path: Path) -> Tuple[int, int]:
    suffix = path.suffix.lower()
    with path.open("rb") as fp:
        if suffix == ".png":
            return _read_png_size(fp)
        if suffix in {".jpg", ".jpeg"}:
            return _read_jpeg_size(fp)

        # 非 png/jpg 时尝试魔数判断
        header = fp.read(24)
        fp.seek(0)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return _read_png_size(fp)
        if header[:2] == b"\xff\xd8":
            return _read_jpeg_size(fp)

    raise ValueError(f"暂不支持读取该格式尺寸: {path.suffix}")


def analyze_imgs(imgs_dir: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "是否存在imgs": False,
        "imgs图片总数": 0,
        "imgs有效图片数": 0,
        "分辨率分类": "NO_IMGS",
        "分辨率明细": "",
        "无效图片文件名列表": "",
    }

    if not imgs_dir.exists() or not imgs_dir.is_dir():
        return result

    result["是否存在imgs"] = True
    image_files = sorted(
        [p for p in imgs_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name,
    )
    result["imgs图片总数"] = len(image_files)

    if not image_files:
        result["分辨率分类"] = "EMPTY_IMGS"
        return result

    size_counter: Counter[Tuple[int, int]] = Counter()
    invalid_files: List[str] = []

    for img_path in image_files:
        try:
            size = get_image_size(img_path)
            size_counter[size] += 1
        except Exception:
            invalid_files.append(img_path.name)

    result["imgs有效图片数"] = sum(size_counter.values())
    result["无效图片文件名列表"] = join_list(invalid_files)

    if not size_counter:
        result["分辨率分类"] = "NO_VALID_IMAGE"
        return result

    result["分辨率明细"] = " | ".join(
        f"{w}x{h}({cnt})" for (w, h), cnt in sorted(size_counter.items())
    )

    heights = {h for (_, h) in size_counter.keys()}
    if len(heights) == 1 and 720 in heights:
        result["分辨率分类"] = "720p"
    elif len(heights) == 1 and 1080 in heights:
        result["分辨率分类"] = "1080p"
    else:
        result["分辨率分类"] = "other"

    return result


def parse_collision_frame(anno: Dict[str, Any]) -> Optional[int]:
    raw_value = anno.get("碰撞时刻帧数（填数字）")
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    return int(match.group())


def extract_anno_fields(anno: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not isinstance(anno, dict):
        return {field: "__NO_ANNO__" for field in ANNO_FIELDS}

    inner_labels = anno.get("inner_labels")
    if not isinstance(inner_labels, dict):
        inner_labels = {}

    result: Dict[str, str] = {}
    result["是否存在判不清"] = ensure_text(
        inner_labels.get("是否存在判不清", anno.get("是否存在判不清")),
        empty_placeholder="__EMPTY__",
    )
    result["司机特征"] = ensure_text(
        inner_labels.get("司机特征", anno.get("司机特征", anno.get("inner_司机特征"))),
        empty_placeholder="__EMPTY__",
    )
    for field in ANNO_FIELDS:
        if field in result:
            continue
        result[field] = ensure_text(anno.get(field), empty_placeholder="__EMPTY__")
    return result


def normalize_score_key_name(name: str) -> Optional[int]:
    stem = Path(name).stem
    match = re.search(r"(\d+)$", stem)
    if not match:
        return None
    return int(match.group())


def resolve_collision_score_key(
    scores: Dict[str, Any], frame_no: Optional[int], frame_offset: int = 0
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    if frame_no is None:
        return None, None, None

    target_frame = frame_no + frame_offset
    candidates = [
        f"{target_frame:06d}.jpg",
        f"{target_frame}.jpg",
        f"{target_frame:06d}.jpeg",
        f"{target_frame}.jpeg",
    ]
    for key in candidates:
        if key in scores:
            return key, f"{target_frame:06d}.jpg", target_frame

    for key in scores.keys():
        key_frame = normalize_score_key_name(str(key))
        if key_frame is not None and key_frame == target_frame:
            return str(key), f"{target_frame:06d}.jpg", target_frame

    return None, f"{target_frame:06d}.jpg", target_frame


def analyze_scores(
    scores: Optional[Dict[str, Any]],
    collision_score_key: Optional[str],
    inner_threshold: float,
) -> Dict[str, Any]:
    threshold_text = format(inner_threshold, "g")
    result: Dict[str, Any] = {
        "是否存在scores_v0.json": False,
        "scores条目数": 0,
        "state为null的文件数": 0,
        "state非null的文件数": 0,
        "state缺失key的文件数": 0,
        "所有state是否全为null": "",
        "state全量分类": "NO_SCORES",
        "state为null文件名列表": "",
        "state非null文件名列表": "",
        "state缺失key文件名列表": "",
        "碰撞jpg在scores中匹配到的key": collision_score_key or "",
        "碰撞jpg是否在scores中存在": False,
        "碰撞jpg的state是否为null": "",
        "碰撞state分类": "NO_SCORES",
        "碰撞jpg的state原值": "",
        "碰撞inner_top1类别": "",
        "碰撞inner_top1分数": "",
        "碰撞inner_top1是否>阈值": "",
        "inner分类": "NO_SCORES",
    }

    if scores is None:
        return result

    result["是否存在scores_v0.json"] = True

    if not isinstance(scores, dict):
        result["state全量分类"] = "INVALID_SCORES"
        result["碰撞state分类"] = "INVALID_SCORES"
        result["inner分类"] = "INVALID_SCORES"
        return result

    result["scores条目数"] = len(scores)
    null_files: List[str] = []
    non_null_files: List[str] = []
    missing_state_key_files: List[str] = []

    for file_name, payload in sorted(scores.items(), key=lambda x: str(x[0])):
        if not isinstance(payload, dict) or "state" not in payload:
            missing_state_key_files.append(str(file_name))
            continue
        if payload.get("state") is None:
            null_files.append(str(file_name))
        else:
            non_null_files.append(str(file_name))

    result["state为null的文件数"] = len(null_files)
    result["state非null的文件数"] = len(non_null_files)
    result["state缺失key的文件数"] = len(missing_state_key_files)
    result["state为null文件名列表"] = join_list(null_files)
    result["state非null文件名列表"] = join_list(non_null_files)
    result["state缺失key文件名列表"] = join_list(missing_state_key_files)

    if len(scores) == 0:
        result["所有state是否全为null"] = ""
        result["state全量分类"] = "EMPTY_SCORES"
    else:
        all_null = len(null_files) == len(scores)
        result["所有state是否全为null"] = all_null
        result["state全量分类"] = "ALL_NULL" if all_null else "NOT_ALL_NULL"

    # 碰撞帧 state 分析
    if collision_score_key is None:
        result["碰撞state分类"] = "NO_COLLISION_KEY"
        result["inner分类"] = "NO_COLLISION_KEY"
        return result

    collision_entry = scores.get(collision_score_key)
    if not isinstance(collision_entry, dict):
        result["碰撞state分类"] = "MISSING_SCORE_ENTRY"
        result["inner分类"] = "MISSING_SCORE_ENTRY"
        return result

    result["碰撞jpg是否在scores中存在"] = True

    if "state" not in collision_entry:
        result["碰撞jpg的state是否为null"] = ""
        result["碰撞state分类"] = "MISSING_STATE_KEY"
    elif collision_entry.get("state") is None:
        result["碰撞jpg的state是否为null"] = True
        result["碰撞state分类"] = "NULL"
        result["碰撞jpg的state原值"] = "null"
    else:
        result["碰撞jpg的state是否为null"] = False
        result["碰撞state分类"] = "NOT_NULL"
        result["碰撞jpg的state原值"] = stringify_jsonable(collision_entry.get("state"))

    # inner_top3 分析
    inner_top3 = collision_entry.get("inner_top3")
    if not isinstance(inner_top3, dict) or not inner_top3:
        result["inner分类"] = "MISSING_INNER_TOP3"
        return result

    candidates: List[Tuple[str, float]] = []
    for cls_name, score_value in inner_top3.items():
        try:
            candidates.append((str(cls_name), float(score_value)))
        except Exception:
            continue

    if not candidates:
        result["inner分类"] = "MISSING_INNER_TOP3"
        return result

    top_label, top_score = max(candidates, key=lambda x: x[1])
    result["碰撞inner_top1类别"] = top_label
    result["碰撞inner_top1分数"] = top_score
    result["碰撞inner_top1是否>阈值"] = top_score > inner_threshold
    result["inner分类"] = top_label if top_score > inner_threshold else f"TOP1_LE_{threshold_text}"

    return result


def build_empty_row(case_dir: Path, root: Path) -> Dict[str, Any]:
    return {
        "子文件夹名": case_dir.name,
        "相对路径": case_dir.relative_to(root).as_posix(),
        "绝对路径": str(case_dir.resolve()),
        "是否存在imgs": False,
        "imgs图片总数": 0,
        "imgs有效图片数": 0,
        "分辨率分类": "NO_IMGS",
        "分辨率明细": "",
        "无效图片文件名列表": "",
        "是否存在scores_v0.json": False,
        "scores条目数": 0,
        "state为null的文件数": 0,
        "state非null的文件数": 0,
        "state缺失key的文件数": 0,
        "所有state是否全为null": "",
        "state全量分类": "NO_SCORES",
        "state为null文件名列表": "",
        "state非null文件名列表": "",
        "state缺失key文件名列表": "",
        "是否存在online_raw_with_anno.json": False,
        "碰撞时刻帧数_原始": "",
        "碰撞时刻帧数_按偏移修正后": "",
        "碰撞候选jpg": "",
        "碰撞jpg在scores中匹配到的key": "",
        "碰撞jpg是否在scores中存在": False,
        "碰撞jpg的state是否为null": "",
        "碰撞state分类": "NO_ANNO",
        "碰撞jpg的state原值": "",
        "碰撞inner_top1类别": "",
        "碰撞inner_top1分数": "",
        "碰撞inner_top1是否>阈值": "",
        "inner分类": "NO_ANNO",
        "是否存在判不清": "__NO_ANNO__",
        "司机特征": "__NO_ANNO__",
        "视频中自车是否发生碰撞事故": "__NO_ANNO__",
        "碰撞对象类别": "__NO_ANNO__",
        "静态障碍物类别": "__NO_ANNO__",
        "自车行车状态": "__NO_ANNO__",
        "天气情况": "__NO_ANNO__",
        "环境光线": "__NO_ANNO__",
        "事故画面是否存在遮挡": "__NO_ANNO__",
        "错误信息": "",
    }


def analyze_case_dir(
    case_dir: Path,
    root: Path,
    frame_offset: int,
    inner_threshold: float,
) -> Dict[str, Any]:
    row = build_empty_row(case_dir, root)
    errors: List[str] = []

    # 1) imgs 分辨率分析
    try:
        imgs_info = analyze_imgs(case_dir / "imgs")
        row.update(imgs_info)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"analyze_imgs失败: {exc}")
        row["分辨率分类"] = "ERROR"

    # 2) anno 分析
    anno_path = case_dir / "online_raw_with_anno.json"
    anno: Optional[Dict[str, Any]] = None
    collision_frame_no: Optional[int] = None
    if anno_path.exists():
        row["是否存在online_raw_with_anno.json"] = True
        try:
            anno_obj = load_json(anno_path)
            if isinstance(anno_obj, dict):
                anno = anno_obj
            else:
                errors.append("online_raw_with_anno.json 顶层不是对象")
            row.update(extract_anno_fields(anno))
            if anno is not None:
                raw_collision = anno.get("碰撞时刻帧数（填数字）")
                row["碰撞时刻帧数_原始"] = stringify_jsonable(raw_collision)
                collision_frame_no = parse_collision_frame(anno)
                if collision_frame_no is not None:
                    row["碰撞时刻帧数_按偏移修正后"] = collision_frame_no + frame_offset
                else:
                    row["碰撞时刻帧数_按偏移修正后"] = ""
        except Exception as exc:  # noqa: BLE001
            errors.append(f"读取 online_raw_with_anno.json 失败: {exc}")
            row.update({field: "__ANNO_READ_ERROR__" for field in ANNO_FIELDS})
            row["碰撞state分类"] = "ANNO_READ_ERROR"
            row["inner分类"] = "ANNO_READ_ERROR"
    else:
        row["碰撞state分类"] = "NO_ANNO"
        row["inner分类"] = "NO_ANNO"

    # 3) scores 分析
    scores_path = case_dir / "scores_v0.json"
    scores: Optional[Dict[str, Any]] = None
    collision_score_key: Optional[str] = None
    candidate_jpg: Optional[str] = None
    target_frame_after_offset: Optional[int] = None

    if scores_path.exists():
        try:
            scores_obj = load_json(scores_path)
            if isinstance(scores_obj, dict):
                scores = scores_obj
                collision_score_key, candidate_jpg, target_frame_after_offset = resolve_collision_score_key(
                    scores, collision_frame_no, frame_offset
                )
            else:
                errors.append("scores_v0.json 顶层不是对象")
                scores = None
        except Exception as exc:  # noqa: BLE001
            errors.append(f"读取 scores_v0.json 失败: {exc}")
            scores = None
    row["碰撞候选jpg"] = candidate_jpg or ""
    if target_frame_after_offset is not None:
        row["碰撞时刻帧数_按偏移修正后"] = target_frame_after_offset

    try:
        scores_info = analyze_scores(scores, collision_score_key, inner_threshold)
        row.update(scores_info)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"analyze_scores失败: {exc}")
        row["state全量分类"] = "ERROR"
        row["碰撞state分类"] = "ERROR"
        row["inner分类"] = "ERROR"

    # 如果根本没有 anno，但有 scores，则把碰撞相关分类修正得更明确一些
    if not row["是否存在online_raw_with_anno.json"] and row["是否存在scores_v0.json"]:
        row["碰撞state分类"] = "NO_ANNO"
        row["inner分类"] = "NO_ANNO"

    if row["是否存在online_raw_with_anno.json"] and collision_frame_no is None:
        # 有 anno 但碰撞帧拿不到
        if row["是否存在scores_v0.json"]:
            row["碰撞state分类"] = "NO_COLLISION_FRAME"
            row["inner分类"] = "NO_COLLISION_FRAME"

    row["错误信息"] = " | ".join(errors)
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_combination_rows(folder_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in folder_rows:
        key = tuple(row.get(field, "") for field in COMBINATION_FIELDS)
        grouped[key].append(row)

    csv_rows: List[Dict[str, Any]] = []
    json_rows: List[Dict[str, Any]] = []

    for key, rows in grouped.items():
        relpaths = sorted(str(r["相对路径"]) for r in rows)
        names = sorted(str(r["子文件夹名"]) for r in rows)
        base = {field: value for field, value in zip(COMBINATION_FIELDS, key)}
        csv_row = {
            **base,
            "子文件夹数量": len(rows),
            "子文件夹名列表": join_list(names),
            "子文件夹相对路径列表": join_list(relpaths),
        }
        csv_rows.append(csv_row)

        json_row = {
            **base,
            "子文件夹数量": len(rows),
            "子文件夹名列表": names,
            "子文件夹相对路径列表": relpaths,
        }
        json_rows.append(json_row)

    csv_rows.sort(
        key=lambda r: (-int(r["子文件夹数量"]), tuple(str(r.get(f, "")) for f in COMBINATION_FIELDS))
    )
    json_rows.sort(
        key=lambda r: (-int(r["子文件夹数量"]), tuple(str(r.get(f, "")) for f in COMBINATION_FIELDS))
    )
    return csv_rows, json_rows


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] root 不存在或不是目录: {root}", file=sys.stderr)
        return 1

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (root / "_analysis_output").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = find_case_dirs(root)
    if not case_dirs:
        print("[WARN] 没找到可分析的样本子文件夹。")
        print("识别规则：包含 imgs 子目录 / scores_v0.json / online_raw_with_anno.json 之一。")
        return 0

    folder_rows: List[Dict[str, Any]] = []
    for idx, case_dir in enumerate(case_dirs, 1):
        try:
            row = analyze_case_dir(
                case_dir=case_dir,
                root=root,
                frame_offset=args.frame_offset,
                inner_threshold=args.inner_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            row = build_empty_row(case_dir, root)
            row["分辨率分类"] = "ERROR"
            row["state全量分类"] = "ERROR"
            row["碰撞state分类"] = "ERROR"
            row["inner分类"] = "ERROR"
            row["错误信息"] = f"未捕获异常: {exc}\n{traceback.format_exc()}"
        folder_rows.append(row)
        print(f"[{idx}/{len(case_dirs)}] 已分析: {case_dir.relative_to(root).as_posix()}")

    folder_fieldnames = [
        "子文件夹名",
        "相对路径",
        "绝对路径",
        "是否存在imgs",
        "imgs图片总数",
        "imgs有效图片数",
        "分辨率分类",
        "分辨率明细",
        "无效图片文件名列表",
        "是否存在scores_v0.json",
        "scores条目数",
        "state为null的文件数",
        "state非null的文件数",
        "state缺失key的文件数",
        "所有state是否全为null",
        "state全量分类",
        "state为null文件名列表",
        "state非null文件名列表",
        "state缺失key文件名列表",
        "是否存在online_raw_with_anno.json",
        "碰撞时刻帧数_原始",
        "碰撞时刻帧数_按偏移修正后",
        "碰撞候选jpg",
        "碰撞jpg在scores中匹配到的key",
        "碰撞jpg是否在scores中存在",
        "碰撞jpg的state是否为null",
        "碰撞state分类",
        "碰撞jpg的state原值",
        "碰撞inner_top1类别",
        "碰撞inner_top1分数",
        "碰撞inner_top1是否>阈值",
        "inner分类",
        "是否存在判不清",
        "司机特征",
        "视频中自车是否发生碰撞事故",
        "碰撞对象类别",
        "静态障碍物类别",
        "自车行车状态",
        "天气情况",
        "环境光线",
        "事故画面是否存在遮挡",
        "错误信息",
    ]

    folder_csv = output_dir / "folder_level_report.csv"
    write_csv(folder_csv, folder_rows, folder_fieldnames)

    combination_rows_csv, combination_rows_json = build_combination_rows(folder_rows)
    combination_csv = output_dir / "combination_report.csv"
    combination_json = output_dir / "combination_report.json"

    combination_fieldnames = [
        *COMBINATION_FIELDS,
        "子文件夹数量",
        "子文件夹名列表",
        "子文件夹相对路径列表",
    ]
    write_csv(combination_csv, combination_rows_csv, combination_fieldnames)
    with combination_json.open("w", encoding="utf-8") as f:
        json.dump(combination_rows_json, f, ensure_ascii=False, indent=2)

    print("\n分析完成。输出文件：")
    print(f"- {folder_csv}")
    print(f"- {combination_csv}")
    print(f"- {combination_json}")
    print(f"\n共识别样本子文件夹: {len(case_dirs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
