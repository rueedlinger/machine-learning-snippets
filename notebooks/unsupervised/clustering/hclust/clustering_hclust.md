>**Note**: This is a generated markdown export from the Jupyter notebook file [clustering_hclust.ipynb](clustering_hclust.ipynb).

# Hierarchical Clustering (SciPy)


```python
%matplotlib inline
from matplotlib import pyplot as plt
from sklearn import datasets

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
```


```python
data, labels_true = datasets.make_blobs(n_samples=750, centers=[[1,1],[0,5],[2,8]], cluster_std=0.7,
                            random_state=0)


plt.scatter(data[:,0], data[:,1])

df = pd.DataFrame(data, columns=['X', 'Y'])

```


    
![png](clustering_hclust_files/clustering_hclust_2_0.png)
    



```python
Z = linkage(df, 'ward')
c, coph_dists = cophenet(Z, pdist(df, metric='euclidean'))
c
```




    0.8336974676406612




```python

```
