import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import os

def main():
    # Download only Stairs
    classes = ["Stairs"]
    
    # Use a new dataset name to avoid conflicts
    dataset_name = "open_images_stairs_large_v1"

    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    print("Downloading 10000 Stairs images...")
    
    # 1. Download 10000 images
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
    
    # 2. Filter samples
    export_view = dataset.match(F("ground_truth").exists())
    
    # 3. Export to datasets/stairs_large
    export_dir = os.path.abspath("./datasets/stairs_large")
    
    export_view.export(
        export_dir=export_dir,
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth", 
    )

    print(f"\n✅ Large Stairs dataset is ready: {export_dir}")

if __name__ == "__main__":
    main()