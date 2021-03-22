>**Note**: This is a generated markdown export from the Jupyter notebook file [clustering_hclust.ipynb](clustering_hclust.ipynb).
>You can also view the notebook with the [nbviewer](https://nbviewer.jupyter.org/github/rueedlinger/machine-learning-snippets/blob/master/notebooks/unsupervised/clustering/hclust/clustering_hclust.ipynb) from Jupyter. 

# Hierarchical Clustering (SciPy)


```python
%matplotlib inline
from matplotlib import pyplot as plt
from sklearn import datasets

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import dendrogram, linkage
```


```python
max_samples = 20
labels = range(1, max_samples+1)

data, labels_true = datasets.make_blobs(n_samples=max_samples, centers=[[1,1],[0,5],[2,8]], cluster_std=0.7,
                            random_state=0)

df = pd.DataFrame(data, columns=['X', 'Y'])

plt.scatter(df.X, df.Y)


for label, x, y in zip(labels, df.X, df.Y):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')

plt.show()
```


    
![png](clustering_hclust_files/clustering_hclust_2_0.png)
    



```python
linked = linkage(df, method='single', metric='euclidean')
dendrogram(linked, labels=labels, orientation='top', distance_sort='descending', show_leaf_counts=True)

plt.show()
```


    
![png](clustering_hclust_files/clustering_hclust_3_0.png)
    



```python
linked = linkage(df, method='ward', metric='euclidean')
dendrogram(linked, labels=labels, orientation='top', distance_sort='descending', show_leaf_counts=True)

plt.show()
```


    
![png](clustering_hclust_files/clustering_hclust_4_0.png)
    



```python
linked = linkage(df, method='complete', metric='euclidean')
dendrogram(linked, labels=labels, orientation='top', distance_sort='descending', show_leaf_counts=True)

plt.show()
```


    
![png](clustering_hclust_files/clustering_hclust_5_0.png)
    



```python
linked = linkage(df, method='average', metric='euclidean')
dendrogram(linked, labels=labels, orientation='top', distance_sort='descending', show_leaf_counts=True)

plt.show()
```


    
![png](clustering_hclust_files/clustering_hclust_6_0.png)
    



```python
linked = linkage(df, method='weighted', metric='euclidean')
dendrogram(linked, labels=labels, orientation='top', distance_sort='descending', show_leaf_counts=True)

plt.show()
```


    
![png](clustering_hclust_files/clustering_hclust_7_0.png)
    



```python
linked = linkage(df, method='centroid', metric='euclidean')
dendrogram(linked, labels=labels, orientation='top', distance_sort='descending', show_leaf_counts=True)

plt.show()
```


    
![png](clustering_hclust_files/clustering_hclust_8_0.png)
    
