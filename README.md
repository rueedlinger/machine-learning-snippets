#  Python Machine Learning Snippets (pymls)
_Python Machine Learning Snippets (pymls)_ is an ongoing project. This project contains various machine learning 
examples as Jupyter Notebooks with scikit-learn, statsmodel, numpy and other libraries.
The examples are tested with Python 3.6.x.

> __Note:__ This is an ongoing project and is far away from complete.

## Get started...
To get started you can choose one of the these approaches:

- create a virtual environment with _virtualenv_ and install the required packages with _pip_
- create a new conda environment and install the packges with _conda install_ 
- use the docker image _rueedlinger/pyml_ (https://github.com/rueedlinger/docker-pyml) which has all 
required packages already installed.

First you should get a copy of this project. To do this just use the git _clone_ or _fork_ command.

```bash
git clone https://github.com/rueedlinger/machine-learning-snippets.git
```

### Virtualenv
The next example shows how to create an environment with "virtualenv" (https://virtualenv.pypa.io/) 
and install the required packages.

```bash
python3 -m py36-ml

source py36-ml/bin/activate

pip install -r requirements.txt
``` 

### Conda
Another aproach is to create an environment with _conda_ 
(http://conda.pydata.org/) and the required packages.

```bash
conda create -n py36-ml python=3.6

source activate py36-ml

conda install --yes --file requirements.txt
``` 

### Docker

First we create a Docker image with all the required Python libraries to run 
the machine learning snippets.

```bash
docker build -t pymls .
```


After you have created the image you can just start the Docker image with the following command.

```bash
docker run -v ${PWD}/notebooks:/notebooks -p8888:8888 -it pymls
```

> _${PWD}_ gives you the current working directory. With the _-v_ flag you can specify where the volume is mounted on your local machine. This should 
> point to the location where the notebooks are stored. The Juypter Notebook is running on port 8888. 
> To change the port mapping to the container you can us the -p flag. 


    
Next you shoud see the following output in the command line.

        Copy/paste this URL into your browser when you connect for the first time,
        to login with a token:
            http://localhost:8888/?token=e00b3199838bcc3f15a3227fd52752eec4992ad8111d1b57

To connect to the Jupyter Notebook you have to copy/paste this URL into your browser.

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