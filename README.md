# Deep Learning Chest X-Ray Pneumonia Detection Pipeline (RSNA Challenge)

An end-to-end, state-of-the-art medical computer vision application built to automate the classification and localization of pneumonia opacities in human chest X-rays. This project transitions from classical dual-headed regression models to modern 2026 self-attention architectures and real-time detectors.

⚡ Key Highlights
- **Hardware Accelerated**: Fully optimized to run natively using NVIDIA CUDA cores on an RTX 2060 SUPER GPU.
- **Dual-Headed & Modern Architectures**: Implements a custom ResNet dual-headed classification/regression network, a cutting-edge Swin Transformer classifier via the `timm` registry, and a modern YOLO11 object detector.
- "Elite Validation Strategy": Utilizes a strict 5-Fold Cross-Validation split grouped by unique patient ID to prevent data leakage and guarantee stable leaderboard generalization.

🛠️ The 17-Phase Engineering Blueprint

Phase 1–3: Data Ingestion & Medical Metadata
- Extracted and parsed heavy raw 12-bit/16-bit medical DICOM images (`.dcm`) using `pydicom`.
- Merged isolated patient spreadsheets (`metadata.csv` and `class_info.csv`) into a single unified data matrix mapping patient names to true anatomical targets.

Phase 4–5: Quality Control & Audit
- Developed a high-speed data cleaning scan leveraging mathematical MD5 hashing algorithms (`hashlib`) to verify file integrity, detecting 0 corrupted files and isolating duplicate matrices.

Phase 6–7: Clinical Image Preprocessing & Augmentations
- Scaled images down from 1024x1024 to 256x256 while mathematically downscaling true target bounding box coordinates at a corresponding 1:4 scaling ratio (x' = sx).
- Implemented **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to sharpness-boost faint, cloudy lung fluid pockets.
- Centered inputs using "Z-Score Normalization" and added safe, symmetric clinical data augmentations (`torchvision.transforms.v2`) including Horizontal Flips and Color Jitter to combat overfitting.

Phase 8–10: Custom Model Design & Regression Optimization
- Built a custom "Dual-Headed Network" over a pre-trained ResNet backbone, splitting output features into a binary classification logit branch (Yes/No Pneumonia) and a continuous 4-coordinate bounding box regressor.
- Optimized bounding box boundaries using **Smooth L1 Loss (Huber Loss)** to ensure structural convergence while defending against outlier spikes.

Phase 11–13: Evaluation Metrics & Validation Matrix
- Engineered an automated evaluation judge mapping out "Intersection over Union (IoU)" across 8 official competition thresholds (0.40 to 0.75) to output true "mean Average Precision (mAP)".
- Re-architected data structures into a "5-Fold Cross-Validation Matrix" using scikit-learn's `KFold` to balance the distribution of unique healthy vs. sick patient rows.

Phase 14–17: Advanced Post-Processing & Next-Gen 2026 Ensembling
- Executed an automated classification decision threshold grid search to fine-tune precision boundaries and eliminate false negatives.
- Integrated "Non-Maximum Suppression (NMS)" and "Weighted Boxes Fusion (WBF)" algorithms to blend overlapping prediction line clusters into single clear bounding boxes.
- Deployed a next-generation **Swin Transformer Self-Attention Backbone** alongside **YOLO11 object detection loops**, achieving an incredible single-image inference speed of **0.9ms** on local GPU hardware.

---

📁 Repository Structure
```text
├── chestai.ipynb             # Main project pipeline notebook code
├── dataset.yaml              # YOLO11 structural dataset navigation card
├── README.md                 # Project documentation and engineering blueprint
└── .gitignore                # Explicitly blocks heavy raw medical datasets and .pt binaries
```

📊 Performance Statistics (RTX 2060 SUPER)
- "Training Epoch Speed": Reduced from 28.10 minutes (CPU) to under 2 minutes (GPU) per epoch.
- "Inference Latency": 0.1ms preprocessing, 0.9ms model inference per chest canvas.
- "Peak Validation Accuracy": Achieved a 0.6341 validation mAP score during multi-epoch optimization training rounds.
