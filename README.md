# Python Machine Learning Snippets

_Python Machine Learning Snippets_ is my ongoing pet project where I try out different machine learning models. This project contains various machine learning examples as Jupyter notebooks with scikit-learn, statsmodel, numpy and other libraries.

> **Note:** This is an ongoing project and far away from complete.

## Getting Started

All the required Python packages can be installed with `pipenv`.

### Project Setup

First you need to install `pipenv`.

```bash
pip install --user pipenv
```

Install all the required packages

```bash
$ pipenv install --dev
```

### Run the Notebook

You can start `jupyter-lab` to play around with the Juypter notebooks.

```bash
pipenv run jupyter-lab
```

### Run the Tests (nbval)

To test the Jupyter notebooks this project uses [nbval](https://github.com/computationalmodelling/nbval), which is a `py.test`
plugin for validating Jupyter notebooks.

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

### Git LFS

Some of the files (\*.png) are stored in Git LFS. When you want to work with them locally you need to install git-lfs and check them out.

```bash
git lfs checkout
```

### CI Build (GitHub Actions)

See the GitHub Actions [build.yml](.github/workflows/build.yml) file for more details.
![CI Build](https://github.com/rueedlinger/machine-learning-snippets/workflows/CI%20Build/badge.svg)

### Export the Jupyter Notebooks to Markdown

To export the Jupyter notebooks to Markdown just run the [export-notebooks.sh](export-notebooks.sh) script.
This scrip uses `nbconvert` to convert the Jupyter notebooks.

```bash
chmod 755 export-notebooks.sh
./export-notebooks.sh
```

# The Snippets...

The following machine learning snippets are available as Jupyter Notebook.

- **Supervised learning**

  - Classification
    - [Text classification with naive bayes](notebooks/supervised/text_classification/text_classification.md) (scikit-learn)
  - Regression

    - Parametric (linear)

      - [Linear regression with sklearn (OLS)](notebooks/supervised/regression/linear/multiple_linear_regression_sklearn.ipynb) (scikit-learn)
      - [Linear regression with statsmodels (OLS)](notebooks/supervised/regression/multiple_linear_regression_statsmodels.md) (statsmodels)
      - [Regression with SVM](notebooks/supervised/regression/linear/regression_svm.ipynb) (scikit-learn)

    - Non-parametric (nonlienar)
      - [Regression with kNN](notebooks/supervised/regression/nonlienar/regression_kNN.ipynb) (scikit-learn)
      - [Regression with decision tree](notebooks/supervised/regression/nonlienar/regression_tree.ipynb) (scikit-learn)
    - Ensemble learning

      - [Regression with random forest](notebooks/supervised/regression/ensemble/regression_random_forest.ipynb) (scikit-learn)
      - [Regression with gradient boosting](notebooks/supervised/regression/ensemble/regression_xgboost.ipynb) (xgboost)

- **Unsupervised learning**
  - Examples
    - [Clustering basics and model evaluation](notebooks/unsupervised/clustering/clustering_basics_model_evaluation.ipynb) (scikit-learn)
    - [Text clustering basics](notebooks/unsupervised/clustering/clustering_text.ipynb) (scikit-learn)
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
    - [MDS](notebooks/unsupervised/dimensionality_reduction/mds/dimensionality_reduction_mds.md) (scikit-learn)
    - [Isomap](notebooks/unsupervised/dimensionality_reduction/isomap/dimensionality_reduction_isomap.ipynb) (scikit-learn)
    - [t-SNE](notebooks/unsupervised/dimensionality_reduction/tsne/dimensionality_reduction_tsne.ipynb) (scikit-learn)
