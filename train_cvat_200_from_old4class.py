import os
import torch

from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

def main():
    setup_logger()
    
    DATA_ROOT = r"C:\capstone\detectron2-main\train with annotation"
    COCO_JSON = os.path.join(DATA_ROOT, "instances_default.json")
    IMG_ROOT  = DATA_ROOT  

    OLD_WEIGHTS_4CLASS = r"C:\capstone\detectron2-main\output_4_classes_final3\model_final.pth"
    OUT_DIR = r"C:\capstone\detectron2-main\output_cvat200_from_old4class"

    
    register_coco_instances("ffe_train", {}, COCO_JSON, IMG_ROOT)

    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    
    cfg.MODEL.WEIGHTS = OLD_WEIGHTS_4CLASS

    cfg.DATASETS.TRAIN = ("ffe_train",)
    cfg.DATASETS.TEST = ()  
    cfg.DATALOADER.NUM_WORKERS = 2

    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    
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
    
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()