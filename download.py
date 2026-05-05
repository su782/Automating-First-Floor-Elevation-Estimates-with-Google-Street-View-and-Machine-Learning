import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import os

def main():
    classes = ["Door", "Window", "Building", "Stairs"]
    
    dataset_name = "open_images_4_classes_v1"

    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    print(f"Downloading classes: {classes} (Bounding Boxes)...")
    
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"], 
        classes=classes,
        max_samples=10000, 
        only_matching=True,
        dataset_name=dataset_name, 
    )

    print("\nDownload complete! Cleaning data...")

    export_view = dataset.match(F("ground_truth").exists())
    
    print(f"Number of samples after cleaning: {len(export_view)}")
    print("Converting to COCO format...")

    export_dir = os.path.abspath("./datasets/4_classes_final_5k")
    
    export_view.export(
        export_dir=export_dir,
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth", 
    )

    print(f"\n✅ Success! Dataset saved to: {export_dir}")

if __name__ == "__main__":
    main()