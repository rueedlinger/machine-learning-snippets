# Python Machine Learning Snippets
These __Machine Learning Snippets__ are tested with Python 3.5.x. (http://conda.pydata.org/)

## Get started...
To get started create a virtual environment and install the required packages. 
The following example shows how to crate an environment with _conda_ and Python 3.5 with
the required packages.


### Virtualenv
The following example shows how to create an environment with 
_"virtualenv"_ (https://virtualenv.pypa.io/)
and Python 3.5 with the required packages.

```bash
virtualenv --python=/usr/bin/python3.5 py35-ps

source py35-ps/bin/activate

pip install -r requirements.txt
``` 

### Conda
The following example shows how to create an environment with _"conda"_ 
(http://conda.pydata.org/) and Python 3.5 with
the required packages.

```bash
conda create -n py35-ps python=3.5

source activate py35-ps

pip install -r requirements.txt
``` 


## The snippets...
Here we have the Jupyter (Python) Notebook __Machine Learning Snippets__.

- __Supervised learning__
    - Classification
        - [Text Classification with Naive Bayes](supervised/text_classification) (scikit-learn)
    - Regression (tbd)
- __Unsupervised learning__ (tbd)
    - Clustering
        - [K-means](unsupervised/clustering/kmeans/clustering_kmeans.ipynb) (scikit-learn)
        - [MeanShift](unsupervised/clustering/meanshift/clustering_meanshift.ipynb) (scikit-learn)
        - [DBSCAN](unsupervised/clustering/dbscan/clustering_dbscan.ipynb) (scikit-learn)
- Dimension reduction
    - linear
        - [PCA with SVD](unsupervised/dimensionality_reduction/pca/dimensionality_reduction_pca.ipynb) (scikit-learn)
        - [PCA with Eigenvector and Correlation Matrix](unsupervised/dimensionality_reduction/eigen/dimensionality_reduction_eigen.ipynb) (numpy)
    - nonlinear (Manifold learning)
        - [MDS](unsupervised/dimensionality_reduction/mds/dimensionality_reduction_mds.ipynb) (scikit-learn)
        - [Isomap](unsupervised/dimensionality_reduction/isomap/dimensionality_reduction_isomap.ipynb) (scikit-learn)
        - [t-SNE](unsupervised/dimensionality_reduction/tsne/dimensionality_reduction_tsne.ipynb) (scikit-learn)