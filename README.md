#  Python Machine Learning Snippets (pymls)
_Python Machine Learning Snippets (pymls)_ is an ongoing project. This project contains various machine learning 
examples as Jupyter notebooks with scikit-learn, statsmodel, numpy and other libraries.
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
You can start `jupyter-lab` to playadround withe the Juypter notebooks.

```bash
pipenv run jupyter-lab
```

### Run the Tests (nbval)

To test the Jupyter notebooks this project uses [nbval](https://github.com/computationalmodelling/nbval), which is a `py.test` plugin for validating Jupyter notebooks.

This will check all Jupyter notebooks for errors.

```bash
pipenv run py.test --nbval-lax
```

### Upgrade Python Packages
Check which packages have changed.

```
pipenv update --outdated
```

This will upgrade everything.

```bash
pipenv update
```

### CI Build (GitHub Actions)
See the GitHub Actions [build.yml](.github/workflows/build.yml) file for more details. 
![CI Build](https://github.com/rueedlinger/machine-learning-snippets/workflows/CI%20Build/badge.svg)

### Export the Jupyter Notebooks
To export the Jupyter notebooks to another format like Markdonw, HTML or PDF. You can adapt the [export-notebooks.sh](export-notebooks.sh) script. 
This scrip uses `nbconvert` to convert the Jupyter notebooks to other formats.

## The Snippets...
At the moment there are the following machine learning snippets available as Jupyter (Python) Notebook.

- __Supervised learning__
    - Classification
        - [Text Classification with Naive Bayes](notebooks/supervised/text_classification/text_classification.md) (scikit-learn)
    - Regression
        - [Multiple Linear Regression with sklearn](notebooks/supervised/linear_regression/multiple_linear_regression_sklearn.md) (scikit-learn)
        - [Multiple Linear Regression with statsmodels](notebooks/supervised/linear_regression/multiple_linear_regression_statsmodels.md) (statsmodels)
- __Unsupervised learning__ 
    - Examples
        - [Clustering Basics and Model Evaluation](notebooks/unsupervised/clustering/clustering_basics_model_evaluation.md) (scikit-learn)
        - [Text Clustering Basics](notebooks/unsupervised/clustering/clustering_text.md) (scikit-learn)
    - Centroid-based clustering
        - [K-means](notebooks/unsupervised/clustering/kmeans/clustering_kmeans.md) (scikit-learn)
    - Density-based clustering
        - [MeanShift](notebooks/unsupervised/clustering/meanshift/clustering_meanshift.md) (scikit-learn)
        - [DBSCAN](notebooks/unsupervised/clustering/dbscan/clustering_dbscan.md) (scikit-learn)
    - Connectivity based clustering
        - [Agglomerative Clustering (Hierarchical Clustering)](notebooks/unsupervised/clustering/agglomerative/clustering_agglomerative.md) (scikit-learn)
        - [Hierarchical Clustering](notebooks/unsupervised/clustering/hclust/clustering_hclust.md) (SciPy)
    - Distribution-based clustering
        - [Gaussian Mixture Model](notebooks/unsupervised/clustering/gaussian_mixture/clustering_gaussian_mixture.md) (scikit-learn)
        
       
- Dimension reduction
    - linear
        - [PCA with SVD](notebooks/unsupervised/dimensionality_reduction/pca/dimensionality_reduction_pca.md) (scikit-learn)
        - [PCA with Eigenvector and Correlation Matrix](notebooks/unsupervised/dimensionality_reduction/eigen/dimensionality_reduction_eigen.md) (numpy)
    - nonlinear (Manifold learning)
        - [MDS](notebooks/unsupervised/dimensionality_reduction/mds/dimensionality_reduction_mds.md) (scikit-learn)
        - [Isomap](notebooks/unsupervised/dimensionality_reduction/isomap/dimensionality_reduction_isomap.md) (scikit-learn)
        - [t-SNE](notebooks/unsupervised/dimensionality_reduction/tsne/dimensionality_reduction_tsne.md) (scikit-learn)

