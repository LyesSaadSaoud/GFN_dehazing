# **Gated Fusion Network (GFN) for Adaptive Underwater Image Dehazing**

Welcome to the official repository for **Gated Fusion Network (GFN)**, an advanced architecture designed for enhancing underwater images by addressing challenges like light scattering, color distortion, and low illumination. This repository hosts the datasets, methodologies, experimental results, and code associated with the GFN project.

---

#### **Figure 1**: Comparative Visual Results of GFN and Baseline Models
- **Description:** 
  - (a) Input image
  - (b) Output from state-of-the-art baseline models
  - (c) Enhanced output from GFN.

![Comparative visual results](images/img1.png)

#### **Figure 2**: Performance Metrics Comparison
- **Description:** Comparative analysis of GFN against baseline models based on metrics such as PSNR, SSIM, and UIQM.

![Performance metrics comparison](images/img2.png)

---
## **Overview**

Underwater environments pose unique challenges for imaging due to light absorption, scattering, and noise, which degrade image quality and hinder usability in critical domains such as marine exploration, underwater robotics, and environmental monitoring. To address these issues, we introduce the **Gated Fusion Network (GFN)**, which leverages **Reprogramming Adaptive Transformation Units (RATUs)** and a **Swin Transformer backbone** for dynamic and robust underwater image enhancement.

---

## **Dataset**

### **Key Details**
- **Training Data:** LSUI3879 dataset (3,879 paired images)
- **Testing Data:**
  - LSUI400, UIEB100, Ocean_ex, EUVP_Test515
  - Non-reference datasets: U45, RUIE_Color90, UPoor200, Challenging-60
- **Custom Dataset:**
  - **KUMP (KU Marine Pool):** 134,000 frames, with a curated subset of 400 representative images for testing.

### **Features**
- **Diverse Conditions:** Covers turbidity, lighting variability, and color distortion.
- **Comprehensive Annotations:** Paired ground truth and degraded images for objective evaluation.
- **Augmentation:** Degradation effects applied to simulate real-world underwater conditions.

The dataset supports training and evaluation of state-of-the-art underwater image enhancement models.

---

## **Experimental Setup and Methodology**

### **Hardware**
- **Platform:** NVIDIA GeForce RTX 4090 GPU
- **Specifications:**
  - CUDA 11.8, cuDNN 8.7.0
  - 24 GB GDDR6X memory
- **Environment:** Python 3.11.5, PyTorch 2.2.1

### **Software Pipeline**
- **Algorithms:**
  - Image Enhancement: Gated Fusion Network (GFN) with RATUs
  - Multi-stage Feature Fusion: Swin Transformer
- **Testing Scenarios:**
  - Reference-based datasets for objective metrics (e.g., PSNR, SSIM)
  - Non-reference datasets for perceptual quality evaluation (e.g., UIQM, UCIQE).

---

## **Key Features and Contributions**

1. **Dynamic Image Enhancement:** GFN dynamically adapts to underwater conditions using gated fusion mechanisms.
2. **Advanced Architectures:**
   - Swin Transformer for global and local feature extraction.
   - RATUs for adaptive feature refinement.
3. **Comprehensive Evaluation:**
   - State-of-the-art performance across diverse datasets.
   - Significant improvement in PSNR, SSIM, and perceptual quality metrics.
4. **Scalability:** Designed for real-world applications, including autonomous underwater vehicles (AUVs).

---

## **Code Examples**

### **Image Enhancement Using GFN**
```python
from gfn_model import GatedFusionNetwork
from dataset_loader import load_dataset
import torch

# Load the model
model = GatedFusionNetwork()
model.load_state_dict(torch.load("model_checkpoint.pth"))
model.eval()

# Load test dataset
test_loader = load_dataset("path/to/test/dataset")

# Perform enhancement
for batch in test_loader:
    inputs = batch["input_images"]
    with torch.no_grad():
        outputs = model(inputs)

    # Save the results
    save_results(outputs, "output/folder")
```

---

## **Results**

### **Quantitative Performance**
- **LSUI400 Dataset:**
  - PSNR: 30.00 dB
  - SSIM: 0.904
- **Ocean_ex Dataset:**
  - UIQM: 1.110
  - UCIQE: 0.567

### **Visual Enhancements**
- GFN significantly improves image clarity, color balance, and noise reduction compared to baseline models like WaterNet and RAUNE-Net.
- Demonstrated robust performance under extreme conditions, including severe turbidity and low contrast.

---

## **Future Work**

1. Optimizing GFN for real-time applications on resource-constrained devices.
2. Expanding datasets to include:
   - Extreme underwater conditions.
   - Multi-modal data (e.g., sonar, hyperspectral imaging).
3. Aligning enhancement outputs with downstream tasks such as species detection and habitat mapping.

---

## **Citation**

If you use the GFN model or datasets, please cite our work:

```bibtex
@article{GFNPaper,
  author = {Saad Saoud, Lyes and Hussain, Irfan},
  title = {Gated Fusion Network with Reprogramming Transformer Refiners for Adaptive Underwater Image Dehazing},
  year = {2024},
  publisher = {Journal},
  doi = {......},
  url = {https://doi.org/.....}
}
```

---

## **Contact**

For inquiries or collaboration, please contact:

- **Lyes Saad Saoud** (Primary Contributor): `lyes.saoud@kustar.ac.ae`
- **Irfan Hussain** (Corresponding Author): `irfan.hussain@ku.ac.ae`

---

### **License**

This repository is shared under the **CC BY 4.0 License**, allowing reuse for non-commercial purposes with proper attribution.
