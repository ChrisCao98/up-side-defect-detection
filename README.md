# UP 面缺陷检测

注塑件 UP 面灰度图的异常检测。目标是**零漏检**：阈值取 149 张缺陷库中的最低原始分，统计没进训练集的良品里有多少张越线。下文所有百分比都是这个口径（零漏检误报率）。

[English](README.en.md)

## 结果

| 数据集 | 图片 | 训练 | 纯 PatchCore | l2+l3 融合 | + 判别头 @1% 预算 |
|---|---:|---:|---:|---:|---:|
| class_3（12 个子组） | 82,478 | 5,170 | 4.36% | 2.68% | **0.95%** |
| class1 / A-D_20 | 4,480 | 720 | 3.67% | 2.71% | **0.88%** |
| class2 / A-E_16 | 2,880 | 200 | 0.60% | **0.30%** | 不需要 |

- **纯 PatchCore** = layer3 + 削库；**融合** = l2/l3 各自削库后取各自最优权重；**判别头** = 每组在误报 ≤1% 的约束下取最大的 PC 权重。
- class2 的纯 PatchCore 已在 1% 预算内，按规则不加头。
- 三批数据的 9 位前缀交集为 0。报告：[class1](reports/report_class1_AD20.md)、[class2](reports/report_class2_AE16.md)、[class3](reports/report_class3_all_groups.md)（中文）。

> 阈值全部借自 149 张异族缺陷库。**低误报不等于抓得住本产品的缺陷**：class1 和 class3 的多数产品没有自有缺陷标签，召回只能靠留出法估计。

## 方法

当前最优链路：

1. **特征**：冻结 `wide_resnet50_2`，只用 `layer3`，输入 256×256，FP32。不要用 anomalib 的多层特征拼接——它以最浅层网格为准，会把 layer3 的定位精度抹掉。
2. **记忆库**：每个产品（或每个曝光组）独立建库，coreset 比例 0.10。
3. **削库**：取 149 张缺陷里分数最低的 7 张作锚点，每张删掉记忆库中最近的 k=20 行，取并集。只删约 0.5% 的行，误报降 2–5 倍，召回不变。
4. **打分**：用削过的库算 PatchCore 原始距离。不要用归一化后的 `pred_score`：归一化常数在训练期标定，跨 ckpt 不可比，而且会把分数 clamp 到 1.0。
5. **阈值**：149 张缺陷的最低原始分。
6. **融合（可选）**：l2、l3 各自建库削库，图像级加权 `w·z_l2 + (1−w)·z_l3`；z 用训练良品的均值和标准差。不要在 patch 级融合。
7. **判别头（可选，需缺陷标签）**：layer3 的 3072 维 mean/max/std 池化 + 逻辑回归（C=0.2，balanced）。正类 149 张缺陷，负类本组训练良品。融合 `w·z_pc + (1−w)·z_head`，权重取「误报 ≤1% 下最大的 PC 权重」。

**训练集抽样同样决定结果：**

- class_3 每个产品每字母 20 张。产品大小失衡时，总预算不变、**按产品大小分配**：class1 的 C57G/C65C/KA1U 分别为 120/480/120 张，误报从 6.70% 降到 3.67%。
- 必须**按帧序号分层**抽样。按文件名字典序等距取点会与 20 帧的块长产生公因数，整档漏掉序号；class1 实测首末帧没有训练样本，误报 45–97%。`prep_group_dataset.py` 默认已是 `--sample stratify`。
- 亮度只在**跨簇亮度比 ≥1.5×** 时拆档；1.1–1.2× 不拆。

**评估 batch 必须为 1。** batch > 1 会漏记样本，分数偏移约 3%；AUROC 恰好 1.0000 是危险信号。

## 已验证无效

多层特征拼接、patch 级融合、layer4、往库里补良品、平场、逐图 MAD 归一化、三通道先验、CLAHE、输入降到 224、跨家族合库、coreset < 0.10、FP16 训练。实测数字见 [class1 报告](reports/report_class1_AD20.md) §10 和 [class3 报告](reports/report_class3_all_groups.md)。

## 仓库结构

```
runs/run_group.sh          训练 l2/l3 → 打分 → 削库 → 再打分
runs/analyze_group.sh      汇总 → l2+l3 融合 → 抽特征 → 判别头 → 两种留出
scripts/data/              分组、亮度分档、硬链接目录树、训练集抽样
scripts/train/             训练、图像级特征抽取
scripts/eval/              原始分打分、削库索引、应用削库
scripts/analyze/           汇总、融合、判别头、权重扫描、固定误报漏检
scripts/report/            Markdown → 单文件 HTML/PDF
reports/                   三份最终验证报告
```

## 复现

### 环境

Docker + NVIDIA GPU。镜像：Python 3.11、torch 2.5.0+cu121、anomalib 2.2.0、lightning 2.5.5、timm 1.0.20。

宿主机需要 `python3` + `numpy`（`summarize_ae16.py`、`fuse_ae16.py` 在宿主机跑）；导出报告另需 `python-markdown` 和 `libreoffice`。

```bash
mkdir -p weights
python3 -c "import timm, torch; torch.save(timm.create_model('wide_resnet50_2', pretrained=True).state_dict(), 'weights/pytorch_model.bin')"
docker build -t up_patchcore-v1 .
```

### 数据布局

```
/path/to/data/<9 位前缀>/<字母>/*.jpg       # 正常图
eval_defect149/**/*.jpg                     # 149 张缺陷，文件名 <前缀>_<字母><序号>_UP.jpg
```

### 完整链路

```bash
# 1 分组：stripe(前 4 位) + 每字母张数 → 产品 → 签名组
python3 scripts/data/group_products.py --root /path/to/data --out-dir scores

# 2 量亮度；只在跨簇亮度比 >=1.5x 时按明暗拆档
python3 scripts/data/exposure_groups.py --src /path/to/data
python3 scripts/data/split_by_brightness.py --group A-E_16 \
  --groups scores/product_groups.csv \
  --signatures scores/signature_groups.csv \
  --brightness scores/prefix_brightness.csv \
  --out scores/signature_groups_split.csv

# 3 建硬链接目录树。--base 必须与 --src 同一文件系统；
#   若 --base 不在仓库内，在仓库根建同名软链接指过去。
python3 scripts/data/build_product_layout.py --by-signature --with-stripe \
  --groups scores/signature_groups_split.csv \
  --src /path/to/data --base .

# 4 抽训练集（stratify 是默认）
python3 scripts/data/prep_group_dataset.py \
  --src products_by_signature/A-E_16/bright --name AE16bright --per-stripe 100

# 5 训练 l2/l3 → 打分 → 削库 → 再打分
EVAL_BS=1 bash runs/run_group.sh products_by_signature/A-E_16/bright AE16bright ae16

# 6 汇总、融合、判别头、留出检验
bash runs/analyze_group.sh products_by_signature/A-E_16/bright AE16bright ae16

# 7 1% 预算下的判别头权重扫描（先改 head_weight_sweep.py 顶部的 GROUPS）
docker run --rm -u "$(id -u):$(id -g)" -w /app \
  -v "$PWD/scores:/app/scores" \
  -v "$PWD/scripts/analyze/head_weight_sweep.py:/app/h.py" \
  --entrypoint python up_patchcore-v1 /app/h.py --step 0.01
```

**固定误报率下的漏检数**（横比模型时更稳，不受最弱缺陷影响）：

```bash
docker run --rm -u "$(id -u):$(id -g)" -w /app \
  -v "$PWD/scores:/app/scores" \
  -v "$PWD/scripts/analyze/miss_at_fp.py:/app/m.py" \
  --entrypoint python up_patchcore-v1 /app/m.py \
  --l2 ae16_l2_pruned --l3 ae16_l3_pruned \
  --train-list scores/AE16bright_train_list.csv \
  --feats scores/feats_ae16.npz --fp 0.01
```

**class1 的按产品分配预算**：

```bash
python3 scripts/data/prep_group_dataset.py --src products_by_signature_class_1/A-D_20 \
  --name class1_AD20r --per-stripe 240 \
  --per-stripe-map "C57G=120,C65C=480,KA1U=120"
```

**报告导出**（宿主机需要 `python-markdown` 和 `libreoffice`）：

```bash
python3 scripts/report/export_report.py reports/report_class1_AD20.md
```

## 限制

- 缺陷库只有 149 张、来自 17 个家族；class1/class3 的多数产品无自有缺陷，阈值是借来的。
- 零漏检阈值由最弱的 2–3 张缺陷决定，横比时同时看固定 1% 误报下的漏检数。
- class1 剩余误报集中在 C65C 的首末帧：20 帧连拍期间画面缩放漂移（宽度 569→559→569 px），应在采集端解决；加训练样本的边际收益已很薄。
- `head_weight_sweep.py` 的对照组列表写死在文件顶部，换数据集时需改 `GROUPS`。
