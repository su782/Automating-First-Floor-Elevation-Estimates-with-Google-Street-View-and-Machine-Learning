import os
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

def main():
    setup_logger()

    # 1. Register the new 4-class dataset
    register_coco_instances(
        "train_4_classes", 
        {}, 
        "./datasets/4_classes_final/labels.json", 
        "./datasets/4_classes_final/data"
    )

    # 2. Configure the model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    cfg.DATASETS.TRAIN = ("train_4_classes",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # Key setting: now we have 4 classes (Door, Window, Building, Stairs)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    
    # Slightly increase the number of iterations because there are more classes
    cfg.SOLVER.MAX_ITER = 5000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # Output to a new folder
    cfg.OUTPUT_DIR = "./output_4_classes_rcnn"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 3. Start training
    print("Starting training for the 4-class model (Door, Window, Building, Stairs)...")
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()