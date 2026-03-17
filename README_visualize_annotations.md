
# 批量可视化 2D 框标注结果

这个脚本支持两种 JSON 结构，并按“团队文件夹”批量输出可视化结果：

- 格式 1：根节点下是 `children`，框坐标是 `mincol / minrow / maxcol / maxrow`
- 格式 2：根节点下是 `markResult.features`，框坐标在 `geometry.coordinates`

## 输出效果

- 同一类别跨团队统一配色
- 配色采用 NPG（Nature Publishing Group）风格
- 输出图片为 `json同名.png`
- 输出目录结构为：

```text
output_root/
├── 团队A/
│   ├── xxx.png
│   ├── yyy.png
│   └── __summary.csv
└── 团队B/
    ├── aaa.png
    ├── bbb.png
    └── __summary.csv
```

其中 `__summary.csv` 会记录每个 JSON 的处理状态、框数量、标签计数等，方便批量验收。

## 依赖

```bash
pip install pillow
```

## 用法

### 1）两个团队一起处理

```bash
python visualize_annotations.py \
  --team_inputs /path/to/team_a_jsons /path/to/team_b_jsons \
  --team_names TeamA TeamB \
  --image_root /path/to/images \
  --output_root /path/to/vis_results
```

### 2）如果不传团队名，就默认使用输入文件夹名

```bash
python visualize_annotations.py \
  --team_inputs /path/to/team_a_jsons /path/to/team_b_jsons \
  --image_root /path/to/images \
  --output_root /path/to/vis_results
```

### 3）只处理单个团队

```bash
python visualize_annotations.py \
  --team_inputs /path/to/team_a_jsons \
  --image_root /path/to/images \
  --output_root /path/to/vis_results
```

## 图片查找逻辑

脚本会优先尝试以下方式寻找原图：

1. JSON 中显式记录的图片名（如 `imagename`）
2. 与 JSON 同目录、同 stem 的图片
3. `--image_root` 下递归检索匹配图片名或同 stem 图片

如果没有找到原图，脚本会自动生成占位底图，并根据标注框范围或 JSON 中的 `info.width / info.height` 推断画布大小。

## 说明

- 输出图片文件名与 JSON 保持同 stem，例如：`abc.json -> abc.png`
- 如果输入目录下还有子目录，会在输出目录中保留相对层级，避免同名 JSON 相互覆盖
- 相同 label 在同一次运行中会映射到相同颜色，便于跨团队对照验收
