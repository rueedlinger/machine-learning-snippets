# Python Machine Learning Snippets

_Python Machine Learning Snippets_ is my ongoing pet project where I try out different machine learning models. This project contains various machine learning examples as Jupyter notebooks with scikit-learn, statsmodel, numpy and other libraries.

> **Note:** This is an ongoing project and far away from complete.

## Getting Started

### Project Setup

All the required Python packages can be installed with `pipenv`.

```bash
pip install --user pipenv
```

Install all the required packages

```bash
$ pipenv install --dev
```

> **Note:** To run the tests, export the notebooks or more details see [BUILD.md](BUILD.md)

### Run the Notebook

You can start `jupyter-lab` to play around with the Juypter notebooks.

```bash
pipenv run jupyter-lab
```

# The Snippets...

The following machine learning snippets are available as Jupyter Notebook.

## Basics

- [Statistical analysis](notebooks/basics/statistical_analysis.md)
- [Feature scaling](notebooks/basics/feature_scaling.md)

## Classification

### Text

- [Text classification with naive bayes](notebooks/supervised/text_classification/text_classification.md) (scikit-learn)

### Linear

- [Classification with logistic regression](notebooks/supervised/classification/linear/classification_logistic_regression.md) (scikit-learn)
- [Classification with ridge regression](notebooks/supervised/classification/linear/classification_ridge.md) (scikit-learn)
- [Classification with stochastic gradient descent (SGD)](notebooks/supervised/classification/linear/classification_sdg.md) (scikit-learn)

### SVM

- [Classification with SVM](notebooks/supervised/classification/svm/classification_svm.md) (scikit-learn)

### Non-parametric (nonlinear)

- [Classification with k-NN](notebooks/supervised/classification/nonlinear/classification_kNN.md) (scikit-learn)
- [Classification with decision trees](notebooks/supervised/classification/nonlinear/classification_decision_trees.md) (scikit-learn)

### Ensemble learning

- [Classification with random forest](notebooks/supervised/classification/ensemble/classification_random_forest.md) (scikit-learn)
- [Classification with extra-trees](notebooks/supervised/classification/ensemble/classification_extra_trees.md) (scikit-learn)
- [Classification with bagging](notebooks/supervised/classification/ensemble/classification_bagging.md) (scikit-learn)
- [Classification with AdaBoost (boosting)](notebooks/supervised/classification/ensemble/classification_adaboost.md) (scikit-learn)
- [Classification with gradient boosting](notebooks/supervised/classification/ensemble/classification_xgboost.md) (xgboost)

### Neural network

- [Classification with a neural network](notebooks/supervised/neural_net/classification_neural_net.md) (tensorflow / keras)

## Regression

### Linear

- [Linear regression with sklearn (OLS)](notebooks/supervised/regression/linear/multiple_linear_regression_sklearn.md) (scikit-learn)
- [Linear regression with statsmodels (OLS)](notebooks/supervised/regression/linear/multiple_linear_regression_statsmodels.md) (statsmodels)
- [Lasso Regression](notebooks/supervised/regression/linear/regression_lasso.md) (scikit-learn)
- [Ridge Regression](notebooks/supervised/regression/linear/regression_ridge.md) (scikit-learn)
- [Regression with stochastic gradient descent](notebooks/supervised/regression/linear/regression_sgd.md) (scikit-learn)

### SVM

- [Regression with SVM](notebooks/supervised/regression/svm/regression_svm.md) (scikit-learn)

### Non-parametric (nonlinear)

- [Regression with k-NN](notebooks/supervised/regression/nonlinear/regression_kNN.md) (scikit-learn)
- [Regression with decision tree](notebooks/supervised/regression/nonlinear/regression_tree.md) (scikit-learn)

### Ensemble learning

- [Regression with random forest](notebooks/supervised/regression/ensemble/regression_random_forest.md) (scikit-learn)
- [Regression with extra-trees](notebooks/supervised/regression/ensemble/regression_extra_trees.md) (scikit-learn)
- [Regression with bagging](notebooks/supervised/regression/ensemble/regression_bagging.md) (scikit-learn)
- [Regression with AdaBoost (boosting)](notebooks/supervised/regression/ensemble/regression_adaboost.md) (scikit-learn)
- [Regression with gradient boosting](notebooks/supervised/regression/ensemble/regression_xgboost.md) (xgboost)

### Neural network

- [Regression with a neural network](notebooks/supervised/neural_net/regression_neural_net.md) (tensorflow / keras)

## Clustering

### Text & model evaluation

- [Text clustering basics](notebooks/unsupervised/clustering/clustering_text.md) (scikit-learn)
- [Clustering basics and model evaluation](notebooks/unsupervised/clustering/clustering_basics_model_evaluation.md) (scikit-learn)

### Centroid-based clustering

- [K-means](notebooks/unsupervised/clustering/kmeans/clustering_kmeans.md) (scikit-learn)

### Density-based clustering

- [MeanShift](notebooks/unsupervised/clustering/meanshift/clustering_meanshift.md) (scikit-learn)
- [DBSCAN](notebooks/unsupervised/clustering/dbscan/clustering_dbscan.md) (scikit-learn)

### Connectivity based clustering

- [Agglomerative Clustering (Hierarchical Clustering)](notebooks/unsupervised/clustering/agglomerative/clustering_agglomerative.md) (scikit-learn)
- [Hierarchical Clustering](notebooks/unsupervised/clustering/hclust/clustering_hclust.md) (SciPy)

### Distribution-based clustering

- [Gaussian Mixture Model](notebooks/unsupervised/clustering/gaussian_mixture/clustering_gaussian_mixture.md) (scikit-learn)

## Dimension reduction

### Linear

- [PCA with SVD](notebooks/unsupervised/dimensionality_reduction/pca/dimensionality_reduction_pca.md) (scikit-learn)
- [PCA with Eigenvector and Correlation Matrix](notebooks/unsupervised/dimensionality_reduction/eigen/dimensionality_reduction_eigen.md) (numpy)

### Nonlinear (Manifold learning)

- [MDS](notebooks/unsupervised/dimensionality_reduction/mds/dimensionality_reduction_mds.md) (scikit-learn)
- [Isomap](notebooks/unsupervised/dimensionality_reduction/isomap/dimensionality_reduction_isomap.md) (scikit-learn)
- [t-SNE](notebooks/unsupervised/dimensionality_reduction/tsne/dimensionality_reduction_tsne.md) (scikit-learn)

## Hyperparameter optimization

- [Hyperparameter optimization with GridSearch](notebooks/hyperparameter/hyperparameter_gridsearch.md) (scikit-learn)

## AutoML

### Classification

- [Classification with AutoML](notebooks/automl/classification_with_automl.md) (auto-sklearn)

### Regression

- [Regression with AutoML](notebooks/automl/regression_with_automl.md) (auto-sklearn)

## Autoencoder

- [Anomaly detection with an Autoencoder](notebooks/unsupervised/neural_net/anomaly_detection_with_autoencoder.md) (tensorflow / keras)
