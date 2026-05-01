# SpatialEx

> **High-Parameter Spatial Multi-Omics through Histology-Anchored Integration**

[![Nature Methods](https://img.shields.io/badge/Published-Nature%20Methods-blue)](https://www.nature.com/articles/s41592-025-02926-6)
[![BioRxiv](https://img.shields.io/badge/Preprint-BioRxiv-green)](https://www.biorxiv.org/content/10.1101/2025.02.23.639721v2.abstract)
[![Tutorials](https://img.shields.io/badge/Tutorials-Read%20the%20Docs-orange)](https://spatialex-tutorials.readthedocs.io/en/latest)

SpatialEx is a powerful tool for high-parameter spatial multi-omics analysis through histology-anchored integration. This repository contains the source code and implementation for our published work.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

SpatialEx enables high-parameter spatial multi-omics analysis by integrating histology information with multi-omics data. The published version and preprint of our work are available at:

- **Published Version**: [Nature Methods](https://www.nature.com/articles/s41592-025-02926-6)
- **Preprint**: [BioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.23.639721v2.abstract)

📘 **Step-by-step tutorials** are available at our [documentation site](https://spatialex-tutorials.readthedocs.io/en/latest).

---

## Architecture

### Overall Architecture of SpatialEx and SpatialEx+

![Architecture](https://github.com/KEAML-JLU/SpatialEx/blob/main/figure.jpg)

---

## Installation

### Quick Install

Install SpatialEx directly from PyPI:

```bash
pip install SpatialEx
```

### Install from Requirements

Alternatively, install from the requirements file:

```bash
pip install -r requirements.txt
```

### Manual Installation

If you prefer to install packages individually to avoid potential conflicts, here are the key dependencies:

```bash
pip install anndata==0.8.0
pip install scanpy==1.9.3
pip install numpy==1.23.5
pip install pandas==2.0.3
pip install cellpose==3.0.10
pip install scikit-image==0.21.0
pip install scikit-learn==1.3.2
pip install scikit-misc==0.2.0
pip install torch==2.3.1
pip install huggingface-hub==0.24.6
pip install timm==1.0.8
pip install torchvision==0.18.1
```

> ⚠️ **Note**: We recommend installing the above Python packages one by one to avoid potential dependency conflicts.

---

## Usage

We have packaged our implementation into an easy-to-use Python library for the research community. 

- 📖 **Detailed Installation Guide**: [Tutorial Documentation](https://spatialex-tutorials.readthedocs.io/en/latest/Installation.html)
<!-- - 📓 **Comprehensive Examples**: See `Demonstration.ipynb` for detailed guides to all applications in the paper--> 

---

## Datasets

### Processed Data

The processed data generated in this study are available on Google Drive:

🔗 [Download Processed Data](https://drive.google.com/drive/folders/1mg5TK__G0540rcdGWYjNs6rv51l-PHZA?usp=drive_link)

### Public Datasets

#### 🧬 Xenium Human Breast Cancer Tissue Dataset

- **Source**: [10x Genomics Dataset Explorer](https://www.10xgenomics.com/products/xenium-in-situ/human-breast-dataset-explorer)

#### 🧬 10x Xenium Human Breast (Entire Sample Area)

- **Source**: [10x Genomics Public Datasets](https://www.10xgenomics.com/datasets/ffpe-human-breast-using-the-entire-sample-area-1-standard)

#### 🧬 Spatial Multimodal Analysis (SMA) Dataset

- **Source**: [Mendeley Data](https://data.mendeley.com/datasets/w7nw4km7xd/1)

### Preprocessed Data from Other Studies

The preprocessed **IF data** for Xenium Human Breast Cancer Rep1 and **mouse brain SMA** data were obtained from another study:

- 📄 **Citation**: [NicheTrans](https://doi.org/10.1101/2024.12.05.626986)
- 🔗 **Data Repository**: [Zenodo](https://zenodo.org/records/15706278)
- 📘 **Tutorials**: [NicheTrans Tutorials](https://nichetrans-tutorials.readthedocs.io/)

> ⚠️ **Important**: Please cite the NicheTrans study when using their preprocessed data.

---

## Citation

If you find our work useful, please cite our paper:

```bibtex
@article{liu2025high,
  title={High-Parameter Spatial Multi-Omics through Histology-Anchored Integration},
  author={Liu, Yonghao and Wang, Chuyao and Wang, Zhikang and Chen, Liang and Li, Zhi and Song, Jiangning and Zou, Qi and Gao, Rui and Qian, Binzhi and Feng, Xiaoyue and Guan, Renchu and Yuan, Zhiyuan},
  journal={Nature Methods},
  year={2025}
}
```

---

## Contact

If you have any questions or need support, please feel free to contact us:

- 📩 **Yonghao Liu**: [yonghao20@mails.jlu.edu.cn](mailto:yonghao20@mails.jlu.edu.cn)
- 📩 **Chuyao Wang**: [wcy22@mails.jlu.edu.cn](mailto:wcy22@mails.jlu.edu.cn)

For a more prompt response, we kindly recommend reaching out via email.
