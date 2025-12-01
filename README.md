# SpatialEx
The source code for "High-Parameter Spatial Multi-Omics through Histology-Anchored Integration".

📘 Our step-by-step **tutorials** can be found [here](https://spatialex-tutorials.readthedocs.io/en/latest).

## Overall architecture of SpatialEx and SpatialEx+
![](https://github.com/KEAML-JLU/SpatialEx/blob/main/figure.jpg)
## Usage
You can run the following the command.

```
pip install -r requirements.txt
```

```
pip install SpatialEx
```
Several important packages are listed below in case you prefer not to install too many packages.
```
anndata==0.8.0
scanpy==1.9.3
numpy==1.23.5
pandas==2.0.3
cellpose==3.0.10
scikit-image==0.21.0
scikit-learn==1.3.2
scikit-misc==0.2.0
torch==2.3.1
huggingface-hub==0.24.6
timm==1.0.8
torchvision==0.18.1
```

We recommend installing the above Python packages one by one to avoid potential errors.

We have packaged our implementation into an easy-to-use Python library for use by the research community. The accompanying [tutorial](https://spatialex-tutorials.readthedocs.io/en/latest/Installation.html) offers detailed guidance on how to use the package effectively.

A detailed guide to all applications in the paper is available in ```Demonstration.ipynb```.

### Prepare datasets
🧬 The Xenium Human Breast Cancer tissue dataset is available at this [site](https://www.10xgenomics.com/products/xenium-in-situ/human-breast-dataset-explorer).

🧬 The 10x Xenium Human Breast Using the Entire Sample Area dataset is publicly available at this [site](https://www.10xgenomics.com/datasets/ffpe-human-breast-using-the-entire-sample-area-1-standard). 

🧬 The Spatial Multimodal Analysis (SMA) dataset is available at this [site](https://data.mendeley.com/datasets/w7nw4km7xd/1). 

The preprocessed **IF data** for Xenium Human Breast Cancer Rep1, and **mouse brain SMA** data were obtained from another study, please cite the study using the following link: 📄<a href="https://doi.org/10.1101/2024.12.05.626986">NicheTrans</a>, the data are stored in 🔗[Zenodo](https://zenodo.org/records/15706278), the usage can also be found in 📘[NicheTrans Tutorials](https://nichetrans-tutorials.readthedocs.io/).



If you have any questions, please feel free to contact 📩 [yonghao](yonghao20@mails.jlu.edu.cn) or 📩 [chuyao](wcy22@mails.jlu.edu.cn).
