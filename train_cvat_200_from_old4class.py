import os
import torch

from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

def main():
    setup_logger()

    # ====== 1) 路径：改成你自己的 ======
    DATA_ROOT = r"C:\capstone\detectron2-main\train with annotation"
    COCO_JSON = os.path.join(DATA_ROOT, "instances_default.json")
    IMG_ROOT  = DATA_ROOT  # 因为你的 file_name 形如 "train images/xxx.jpg" :contentReference[oaicite:1]{index=1}

    OLD_WEIGHTS_4CLASS = r"C:\capstone\detectron2-main\output_4_classes_final3\model_final.pth"
    OUT_DIR = r"C:\capstone\detectron2-main\output_cvat200_from_old4class"

    # ====== 2) 注册数据集 ======
    register_coco_instances("ffe_train", {}, COCO_JSON, IMG_ROOT)

    # ====== 3) 配置 ======
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # 用你的旧模型初始化（4类 -> 3类时 predictor 会自动跳过不匹配的层）
    cfg.MODEL.WEIGHTS = OLD_WEIGHTS_4CLASS

    cfg.DATASETS.TRAIN = ("ffe_train",)
    cfg.DATASETS.TEST = ()  # 先不做val也能跑起来
    cfg.DATALOADER.NUM_WORKERS = 2

    # 你的当前 COCO 是 3 类：door/stairs/building :contentReference[oaicite:2]{index=2}
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    # 小数据集微调：建议小学习率 + 少迭代
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 2000
    cfg.SOLVER.STEPS = (1400, 1800)
    cfg.SOLVER.CHECKPOINT_PERIOD = 500

    cfg.OUTPUT_DIR = OUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", cfg.MODEL.DEVICE)

    trainer = DefaultTrainer(cfg)
    # resume=False：用旧权重初始化，但优化器/iter 从 0 重新开始（推荐）
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()