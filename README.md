# First Floor Elevation (FFE) Estimation Pipeline

This repository implements an end-to-end pipeline for estimating First Floor Elevation (FFE) of single-family houses using Google Street View (GSV), instance segmentation, and depth maps.

The system retrieves building addresses from Census Tracts, downloads multiple Street View images per property, selects the best frontal view, refines masks using Segment Anything (SAM), retrieves depth maps, and estimates FFE automatically.

---

## Pipeline Overview

### Input
- Census Tract ID(s)

### Processing Steps
1. Retrieve single-family building addresses  
2. Download multiple Google Street View images (walk-15 strategy)  
3. Select best frontal building view  
4. Refine door and ground masks using SAM  
5. Download depth map for selected panorama  
6. Estimate FFE using depth difference  

### Output
- Estimated FFE per building  
- Structured intermediate outputs for reproducibility and debugging  


---

## Installation (Linux Recommended)

### 1. Create Virtual Environment

```bash
python3 -m venv ffe_env
source ffe_env/bin/activate
```
### 2. Install PyTorch

Adjust CUDA version if needed.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Detectron2
```
pip install detectron2==0.6 \
-f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```
### 4. Install Remaining Dependencies
```
pip install -r requirements.txt
```

### 5. Install Segment Anything
```
pip install segment-anything
```
## Running the Pipeline
### Single Census Tract
```
python run_pipeline.py 36061000100
```
### Multiple Census Tracts
```
python run_pipeline.py 36061000100 36061000200
```
### From CSV File
```
python run_pipeline.py tract_list.csv
```
### CSV Format
```
tract_id
36061000100
36061000200
```
## Output Structure

### All results are saved under:
```
outputs/<tract_id>/
```

Each tract folder contains:

Retrieved address list
Downloaded GSV images
Selected top views
Refined segmentation masks
Depth maps
FFE estimation results (CSV)

All intermediate files are preserved for transparency and reproducibility.

## Requirements

See requirements.txt for Python package dependencies.

Detectron2 and PyTorch must be installed separately.