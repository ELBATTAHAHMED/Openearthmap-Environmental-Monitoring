![image_alt](https://github.com/ELBATTAHAHMED/Openearthmap-Environmental-Monitoring/blob/6ea244cca391ba22ea6f5b11b737328336c0ae4d/Openearthmap-Environmental-Monitoring.png)

# Vision Model for Environmental Monitoring (OpenEarthMap)

This project builds an end-to-end deep learning pipeline for environmental monitoring from aerial imagery.
It uses semantic segmentation to map land-cover classes, with a focus on **vegetation** and **water** analysis.

The workflow is implemented in two notebooks:

1. `01_OpenEarthMap_Data_Preparation.ipynb`
2. `02_Baseline_UNet_Segmentation_OpenEarthMap.ipynb`

The report/documentation is in:

- `DL_ARTICLE.pdf`

---

## Project Goal

Given an RGB aerial image, predict a pixel-wise land-cover mask for 8 OpenEarthMap classes:

- Bareland
- Rangeland
- Developed
- Road
- Tree
- Water
- Agriculture
- Building

The project then uses these segmentation maps as a proxy for environmental monitoring (vegetation/water change indicators).

---

## What Each Notebook Does

### 1) Data Preparation Notebook

`01_OpenEarthMap_Data_Preparation.ipynb`

Main tasks:

- Mounts Google Drive and sets `DL_project` paths
- Downloads OpenEarthMap from KaggleHub (if not already available)
- Validates raw split files and image/label consistency
- Decodes/remaps masks into training class IDs (with `IGNORE_INDEX=255` support)
- Creates labeled splits from raw train+val (80/10/10)
- Computes class distribution (imbalance check)
- Computes dataset normalization statistics
- Exports prepared dataset artifacts

Prepared split sizes exported by the notebook:

- Train: **2149**
- Val: **268**
- Test (labeled): **270**
- Infer test (images only): **1151**

Important exported files/folders:

- `prepared/images/{train,val,test,infer_test}`
- `prepared/masks/{train,val,test}`
- `prepared/splits/*.txt`
- `prepared/meta.json` and `prepared/metadata.json`
- `data_locations.json`

### 2) Training + Evaluation Notebook

`02_Baseline_UNet_Segmentation_OpenEarthMap.ipynb`

Main tasks:

- Loads prepared dataset and metadata
- Builds and trains a baseline U-Net
- Optionally trains an improved model (`smp_unet_resnet34`)
- Uses weighted segmentation losses (default CE + Dice)
- Evaluates with mIoU, per-class IoU, pixel accuracy, macro metrics
- Generates qualitative prediction plots
- Runs a synthetic change-detection demo for vegetation and water
- Exports checkpoints, metrics, plots, and summary artifacts

---

## Results

### Baseline U-Net (validation)

- mIoU: **0.4416**
- Pixel Accuracy: **0.6385**
- Water IoU: **0.3974**
- Vegetation superclass IoU: **0.8135**

Per-class IoU (baseline):

| Class | IoU |
|---|---:|
| Bareland | 0.2057 |
| Rangeland | 0.3569 |
| Developed | 0.3663 |
| Road | 0.4749 |
| Tree | 0.5704 |
| Water | 0.3974 |
| Agriculture | 0.5536 |
| Building | 0.6072 |

### Improved model (`smp_unet_resnet34`, validation)

- mIoU: **0.4961**
- Pixel Accuracy: **0.6691**
- Water IoU: **0.6045**
- Vegetation superclass IoU: **0.8545**

### Synthetic change-detection demo

Reported mean deltas:

- Ground truth vegetation delta: **-0.690%**
- Ground truth water delta: **+0.105%**
- Predicted vegetation delta: **-0.527%**
- Predicted water delta: **+0.335%**

---

## How to Run in Google Colab

### A) Before you start

1. Use a Google account with enough Drive space (dataset download/extraction is large, around 8.5 GB compressed plus prepared outputs).
2. Create this folder in Drive: `MyDrive/DL_project`
3. Upload both notebooks into your Drive (same folder is recommended).

### B) Open notebooks in Colab

1. In Google Drive, right-click `01_OpenEarthMap_Data_Preparation.ipynb`.
2. Click **Open with -> Google Colaboratory**.
3. In Colab, set runtime to GPU: **Runtime -> Change runtime type -> T4 GPU**.

### C) Run Notebook 1 (Data Preparation)

1. Run all cells (`Runtime -> Run all`).
2. Wait for dataset download, checks, remapping, and export to finish.
3. Confirm generated outputs under `MyDrive/DL_project/prepared`.

### D) Run Notebook 2 (Training + Evaluation)

1. Open `02_Baseline_UNet_Segmentation_OpenEarthMap.ipynb` in Colab.
2. Keep default settings for full run:
   - `RUN_BASELINE = True`
   - `RUN_IMPROVED = True`
3. Run all cells.
4. At the end, check metrics and artifacts in:
   - `outputs/unet_baseline/`
   - `outputs/model_optionA/`
   - `outputs/comparison/`
   - `outputs/change_detection/`

> Note: by default, `OUTPUTS_DIR = Path("outputs")` (runtime-local path). If you want persistent outputs in Drive, set it to `PROJECT_DIR / "outputs"` before training.

---

## Expected Folder Structure (after Notebook 1)

```text
DL_project/
  data_locations.json
  raw_openearthmap/
  prepared/
    images/
      train/
      val/
      test/
      infer_test/
    masks/
      train/
      val/
      test/
    splits/
      train_labeled.txt
      val_labeled.txt
      test_labeled.txt
      infer_test.txt
    meta.json
    metadata.json
```

---

## Notes

- The notebooks include sanity checks for split consistency and mask value validity.
- Class imbalance is handled using class weighting in the loss.
- A benign DataLoader cleanup warning can appear at runtime shutdown in some sessions; metrics remain valid.

---

## Author

EL BATTAH Ahmed
