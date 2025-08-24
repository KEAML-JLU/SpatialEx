from setuptools import Command, find_packages, setup

__lib_name__ = "SpatialEx"
__lib_version__ = "0.1.4"
__description__ = "computational frameworks that leverage histology as a universal anchor to integrate spatial molecular data across tissue sections"
__url__ = "https://github.com/KEAML-JLU/SpatialEx"
__author__ = "Yonghao Liu and Chuyao Wang"
__author_email__ = "yonghao20@mails.jlu.edu.cn"
__license__ = "MIT"
__requires__ = ["requests",]

with open("README.md", "r", encoding="utf-8") as f:
    __long_description__ = f.read()

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ["SpatialEx"],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
    long_description = '''The foundational SpatialEx model combines a pre-trained H&E foundation model with hypergraph learning and contrastive learning to predict single-cell omics profiles from histology, encoding multi-neighborhood spatial dependencies and global tissue context. Building upon SpatialEx, SpatialEx+ introduces an omics cycle module that encourages cross-omics consistency across adjacent sections via slice-invariant mapping functions, achieving seamless diagonal integration without requiring co-measured multi-omics data for training.''',
    long_description_content_type="text/markdown"
)

