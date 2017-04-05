# Python Machine Learning Snippets
These __Machine Learning Snippets__ are tested with Python 3.5.x. (http://conda.pydata.org/)

## Get started...
To get started create a virtual environment and install the required packages. 
The following example shows how to crate an environment with _conda_ and Python 3.5 with
the required packages.

```bash
conda create -n py35-ml python=3.5 -f requirements.txt

source activate py35-ml

pip install -r requirements.txt
``` 

## The snippets...
Here we have the Jupyter (Python) Notebook __Machine Learning Snippets__.

- __Supervised learning__
    - Classification
        - [Text Classification (scikit-learn)](supervised/text_classification) - Classification with Naive Bayes (scikit-learn)
    - Regression (tbd)
- __Unsupervised learning__ (tbd)
    - Clustering (tbd)
- Dimension reduction
    - linear
        - [PCA](unsupervised/pca/dimensionality_reduction_pca.ipynb)
    - nonlinear
        - [MDS](unsupervised/mds/dimensionality_reduction_mds.ipynb)
        - [Isomap](unsupervised/isomap/dimensionality_reduction_isomap.ipynb)
        - [t-SNE](unsupervised/tsne/dimensionality_reduction_tsne.ipynb)