import os
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

def main():
    setup_logger()

    # 1. Register all datasets
    register_coco_instances(
        "train_main_5k",
        {},
        "./datasets/4_classes_final_5k/labels.json",
        "./datasets/4_classes_final_5k/data"
    )
    register_coco_instances(
        "train_stairs_large",
        {},
        "./datasets/stairs_large/labels_fixed.json",
        "./datasets/stairs_large/data"
    )

    # 2. Configuration
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    
    # Key setting: load the previously trained model as the starting point
    # Make sure this path points to the model_final.pth that you want to further improve
    cfg.MODEL.WEIGHTS = r"C:\capstone\detectron2-main\output_4_classes_final1\model_final.pth"

    cfg.DATASETS.TRAIN = ("train_main_5k", "train_stairs_large")
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.SOLVER.IMS_PER_BATCH = 4
    
    # Key setting: lower the learning rate
    # Since the model has already learned part of the task, we use a smaller learning rate for fine-tuning
    cfg.SOLVER.BASE_LR = 0.001 
    
    # Key setting: increase the number of iterations
    # Run another 1500 iterations for fine-tuning
    cfg.SOLVER.MAX_ITER = 1500 
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # Output to a new folder to avoid overwriting previous results
    cfg.OUTPUT_DIR = "./output_4_classes_final3"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("Starting continued training / fine-tuning...")
    trainer = DefaultTrainer(cfg)
    
    # resume=False means we load the weights but reset the iteration counter and optimizer
    # This allows us to train with the new learning rate and new MAX_ITER setting
    trainer.resume_or_load(resume=False)
    
    trainer.train()

if __name__ == "__main__":
    main()