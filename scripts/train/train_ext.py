# Import the required modules
from anomalib.data import MVTecAD

from anomalib.models import Patchcore

from anomalib.engine import Engine

import torch



from pathlib import Path
from anomalib.data import Folder
from lightning import seed_everything

# 建议的最稳妥改法：在脚本最上方添加环境变量，强制 timm 指向本地
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
# 训练/验证时显存碎片较严重，开可扩展段；必须在 torch 初始化 CUDA 前设置
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="./datasets/testimages_20260824",
                    help="数据集根目录（下面有 Good/ 和 Defect/）")
parser.add_argument("--out", default="./result_train", help="训练输出目录")
parser.add_argument("--name", default="F_my", help="datamodule 名称，决定输出子目录名")
parser.add_argument("--train-batch-size", type=int, default=32)
parser.add_argument("--eval-batch-size", type=int, default=32,
                    help="验证 batch 越大显存越吃紧（记忆库距离矩阵与 batch 成正比）")
# ---- 以下为扩展参数；train.py 原版不含这些 ----
parser.add_argument("--coreset-ratio", type=float, default=0.1,
                    help="coreset 采样比例；越小记忆库越小、训练和推理都更快")
parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"],
                    help="特征层；去掉 layer2 可把 patch 数降到 1/4")
parser.add_argument("--image-size", type=int, default=256,
                    help="模型输入边长；默认 256")
parser.add_argument("--precision", default="32-true",
                    help="Lightning 精度，如 32-true / 16-mixed / bf16-mixed")
parser.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader 进程数；0 = 单进程（默认，保持旧行为）")
args = parser.parse_args()
print(f"模型配置: layers={tuple(args.layers)} coreset={args.coreset_ratio} "
      f"image_size={args.image_size} precision={args.precision} "
      f"bs={args.train_batch_size} workers={args.num_workers}", flush=True)

SEED = 42
seed_everything(SEED, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"训练数据: {Path(args.root).resolve()}")
print(f"输出目录: {Path(args.out).resolve()}")

datamodule = Folder(
    num_workers=args.num_workers,
    name=args.name,
    root=Path(args.root),
    normal_dir="Good",  # Subfolder containing normal images
    abnormal_dir="Defect",  # Subfolder containing anomalous images
    train_batch_size=args.train_batch_size,
    eval_batch_size=args.eval_batch_size,
    seed=SEED,
)


# image_size 由 PreProcessor 的 transform 决定；256 时保持默认 True，行为与 train.py 一致
pre_processor = (True if args.image_size == 256
                 else Patchcore.configure_pre_processor((args.image_size, args.image_size)))

model = Patchcore(
    backbone="wide_resnet50_2",           # Feature extraction backbone
    layers=tuple(args.layers),            # Layers to extract features from
    pre_trained=False,                    # Use pretrained weights
    coreset_sampling_ratio=args.coreset_ratio,
    pre_processor=pre_processor,
    # num_neighbors=9,                    # Number of nearest neighbors
)

# --- 手动加载本地权重 ---
# 假设你在 Dockerfile 中把权重放到了 /app/weights/pytorch_model.bin
local_weight_path = "/app/weights/pytorch_model.bin"

if os.path.exists(local_weight_path):
    print(f"检测到本地权重，正在加载: {local_weight_path}")
    # 加载权重字典
    state_dict = torch.load(local_weight_path, map_location="cpu", weights_only=True)

    # Anomalib 2.2 中真正的 TIMM 骨干位于下一层 feature_extractor。
    try:
        backbone = model.model.feature_extractor.feature_extractor
        target_state = backbone.state_dict()

        missing_keys = sorted(set(target_state) - set(state_dict))
        shape_mismatches = sorted(
            key
            for key in set(target_state) & set(state_dict)
            if target_state[key].shape != state_dict[key].shape
        )
        if missing_keys or shape_mismatches:
            raise RuntimeError(
                f"缺失参数: {missing_keys}; 形状不匹配参数: {shape_mismatches}"
            )

        load_result = backbone.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            raise RuntimeError(f"加载后仍有缺失参数: {load_result.missing_keys}")

        loaded_state = backbone.state_dict()
        unequal_keys = [
            key
            for key in target_state
            if not torch.equal(loaded_state[key].cpu(), state_dict[key].cpu())
        ]
        if unequal_keys:
            raise RuntimeError(f"加载后内容不一致的参数: {unequal_keys}")

        print(
            f"本地权重加载及强校验成功！已加载 {len(target_state)} 个参数，"
            f"忽略 {len(load_result.unexpected_keys)} 个未使用参数。"
        )
    except Exception as e:
        print(f"权重加载失败，请检查模型与权重是否匹配。报错: {e}")
        raise
else:
    raise FileNotFoundError(
        f"错误：在 {local_weight_path} 未找到权重文件，请检查 Docker 封装路径。"
    )

engine = Engine(
    default_root_dir=args.out,  # 结果保存的路径
    # max_epochs=1,  # Patchcore typically needs only one epoch
    accelerator="auto",  # Automatically detect GPU/CPU
    # devices=1,  # Number of devices to use
    precision=args.precision,
)

# Train the model
engine.train(datamodule=datamodule, model=model)
