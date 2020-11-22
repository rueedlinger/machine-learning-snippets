#  Python Machine Learning Snippets (pymls)
_Python Machine Learning Snippets (pymls)_ is an ongoing project. This project contains various machine learning 
examples as Jupyter Notebooks with scikit-learn, statsmodel, numpy and other libraries.
The examples are tested with Python 3.7.x.

> __Note:__ This is an ongoing project and is far away from complete.

## Getting Started

All the required Python packages can be installed with `pipenv`.

### Project Setup

First you nee to install pipenv.

```bash
$ pip install --user pipenv
```

Install all the required packages

```bash
$ pipenv install --dev
```

### Run the Notebook

```bash
pipenv run jupyter-lab
```

### Run the Tests
To the teh Notebooks this project use `[nbval](https://github.com/computationalmodelling/nbval)` a py.test plugin for validating Jupyter notebooks.

This will check all Jupyter notebooks for errors.

```bash
py.test --nbval-lax
```

### Upgrade Packages
Check which packages have changed.

```
pipenv update --outdated
```

Run for upgrading everything.

```bash
pipenv update
```


## The snippets...
At the moment there are the following machine learning snippets available as Jupyter (Python) Notebook.

- __Supervised learning__
    - Classification
        - [Text Classification with Naive Bayes](notebooks/supervised/text_classification/text_classification.ipynb) (scikit-learn)
    - Regression
        - [Multiple Linear Regression with sklearn](notebooks/supervised/linear_regression/multiple_linear_regression_sklearn.ipynb) (scikit-learn)
        - [Multiple Linear Regression with statsmodels](notebooks/supervised/linear_regression/multiple_linear_regression_statsmodels.ipynb) (statsmodels)
- __Unsupervised learning__ 
    - Examples
        - [Clustering Basics and Model Evaluation](notebooks/unsupervised/clustering/clustering_basics_model_evaluation.ipynb) (scikit-learn)
        - [Text Clustering Basics](notebooks/unsupervised/clustering/clustering_text.ipynb) (scikit-learn)
    - Centroid-based clustering
        - [K-means](notebooks/unsupervised/clustering/kmeans/clustering_kmeans.ipynb) (scikit-learn)
    - Density-based clustering
        - [MeanShift](notebooks/unsupervised/clustering/meanshift/clustering_meanshift.ipynb) (scikit-learn)
        - [DBSCAN](notebooks/unsupervised/clustering/dbscan/clustering_dbscan.ipynb) (scikit-learn)
    - Connectivity based clustering
        - [Agglomerative Clustering (Hierarchical Clustering)](notebooks/unsupervised/clustering/agglomerative/clustering_agglomerative.ipynb) (scikit-learn)
        - [Hierarchical Clustering](notebooks/unsupervised/clustering/hclust/clustering_hclust.ipynb) (SciPy)
    - Distribution-based clustering
        - [Gaussian Mixture Model](notebooks/unsupervised/clustering/gaussian_mixture/clustering_gaussian_mixture.ipynb) (scikit-learn)
        
       
- Dimension reduction
    - linear
        - [PCA with SVD](notebooks/unsupervised/dimensionality_reduction/pca/dimensionality_reduction_pca.ipynb) (scikit-learn)
        - [PCA with Eigenvector and Correlation Matrix](notebooks/unsupervised/dimensionality_reduction/eigen/dimensionality_reduction_eigen.ipynb) (numpy)
    - nonlinear (Manifold learning)
        - [MDS](notebooks/unsupervised/dimensionality_reduction/mds/dimensionality_reduction_mds.ipynb) (scikit-learn)
        - [Isomap](notebooks/unsupervised/dimensionality_reduction/isomap/dimensionality_reduction_isomap.ipynb) (scikit-learn)
        - [t-SNE](notebooks/unsupervised/dimensionality_reduction/tsne/dimensionality_reduction_tsne.ipynb) (scikit-learn)

