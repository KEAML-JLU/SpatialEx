# SpatialEx
The source code for "High-Parameter Spatial Multi-Omics through Histology-Anchored Integration".

Our step-by-step tutorial can be found [here](https://spatialex-tutorials.readthedocs.io/en/latest).

## Overall architecture of SpatialEx and SpatialEx+
![](https://github.com/KEAML-JLU/SpatialEx/blob/main/figure.jpg)
## Usage
You can run the following the command.

```
pip install requirements.txt
```
or
```
pip install SpatialEx
```
We have packaged our implementation into an easy-to-use Python library for use by the research community. The accompanying [tutorial](https://spatialex-tutorials.readthedocs.io/en/latest/Installation.html) offers detailed guidance on how to use the package effectively.
### Prepare datasets
The Xenium Human Breast Cancer tissue dataset is available at this [site](https://www.10xgenomics.com/products/xenium-in-situ/human-breast-dataset-explorer).

The 10x Xenium Human Breast Using the Entire Sample Area dataset is publicly available at this [site](https://www.10xgenomics.com/datasets/ffpe-human-breast-using-the-entire-sample-area-1-standard). 

The Visium Spatial Multimodal Analysis (SMA) dataset is available at this [site](https://data.mendeley.com/datasets/w7nw4km7xd/1).

Here, we provide demonstrations of tasks "SpatialEx Translates Histology to Omics at Single-Cell Resolution" and "SpatialEx+ Enables Larger Panel Spatial Analysis through Panel Diagonal Integration" the for reproducibility. 

For convenience, we have provided the visual features processed by the H&E foundation model.

### SpatialEx Translates Histology to Omics at Single-Cell Resolution
#### step 1
Download the visual representations of the [first](https://drive.google.com/file/d/1730OXeBG6TDQ6ejs5oRGKYhdNXbIU19i/view?usp=sharing) and [second](https://drive.google.com/file/d/17WhaKtG3iXuZuubIJEi4Y0_0z1TMKIRx/view?usp=sharing) slices and place them in the 'dataset' folder.
#### step 2

```
python HE_to_omics.py
```

### SpatialEx+ Enables Larger Panel Spatial Analysis through Panel Diagonal Integration

#### step 1
Download the visual representations of the [Panel A](https://drive.google.com/file/d/1W2QBrb0AQH0f0I7sS8vsdhZmWuUozkwf/view?usp=sharing) from the first slice and [Panel B](https://drive.google.com/file/d/1PLSJM2qYs3BNbLXyGravhMooRT5YuRHH/view?usp=sharing) from the second slice, and place them in the 'dataset' folder.
#### step 2
```
python panel_diagonal_integration.py
```

The processed data used for the following three tasks will be released soon.
### Scalability on Million-Cell Tissue Sections
```
python panel_diagonal_integration_big.py
```

### SpatialEx+ Enables Spatial Multi-omics through Omics Diagonal Integration

#### Application 1: Transcriptomic-protein integration
```
python omics_diagonal_integration_rna_protein.py
```
#### Application 2: metabolomic-transcriptomic
```
python omics_diagonal_integration_rna_metabolism.py
```

If you have any questions, please feel free to contact us via [email](yonghao20@mails.jlu.edu.cn).
