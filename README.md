# EvLine: Event-based Semantic Line Detection (Anonymous Repository)

> **Notice: This repository is currently under construction.**
> To comply with the double-blind review process for ECCV 2026 (Paper ID #10068), all identifying information (authors, affiliations, etc.) has been temporarily removed.
> The full source code is continuously being updated. At this stage, we provide the **ESL dataset** and a **preliminary version of the EvHoughformer model** focusing on the core dual-stream transformer architecture. The complete and polished codebase will be fully released upon acceptance.

Welcome to the anonymous repository for the paper: *"EvLine: Event-based Semantic Line Detection"*.

Semantic line detection in dynamic scenes with fast ego-motion and extreme illumination is highly challenging for frame-based sensors. To address this, we introduce the **Event-based Semantic Line (ESL) dataset** and propose **EvHoughformer**, an event-specialized dual-domain transformer network that effectively integrates event and Hough representations.

## 📂 Dataset Preparation (ESL Dataset)
We introduce the first large-scale benchmark for event-based semantic line detection. The ESL dataset contains 3,012 event sequences collected in diverse indoor and outdoor environments, providing 605,192 semantic line annotations.

Please download the dataset from the Google Drive links below and place them in the `data/` directory.

### Download Links
| Dataset Split | Clips / Frames | Size (Zip / Unzipped) | Google Drive Link |
| :--- | :--- | :--- | :--- |
| **Train Set** | 2,409 / 220,122 | 36 GB / 157 GB | [📥 Download ESL_Train.zip](https://drive.google.com/file/d/1_Osf9b2obTbBQ7o7_VCaBKhaNVM2GY7-/view?usp=drive_link) |
| **Test Set** | 603 / 54,084 | 9 GB / 41 GB | [📥 Download ESL_Test.zip](https://drive.google.com/file/d/1Cfg-RhQ9SY6KTpN_LsDEuM134JX0Sw9p/view?usp=drive_link) |

### Dataset Structure
After downloading and extracting the dataset, your project directory should be structured as follows:

```text
ESL_Dataset/
├── train/
│   ├── seq_01/
│   │   ├── events.txt
│   │   └── groundtruth.txt
│   ├── seq_02/
│   │   ├── events.txt
│   │   └── groundtruth.txt
│   └── ...
└── test/
    ├── seq_01/
    │   ├── events.txt
    │   └── groundtruth.txt
    ├── seq_02/
    │   ├── events.txt
    │   └── groundtruth.txt
    └── ...
```

### 🚀 Quick Start (Preliminary Code)
We provide the core architecture of the EvHoughformer. You can run a standalone preliminary test to verify the model's structural pipeline and parameter counts using synthetic event inputs.

Run Core Architecture Test
```text
python EvHoughformer.py
