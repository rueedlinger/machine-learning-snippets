>**Note**: This is a generated markdown export from the Jupyter notebook file [clustering_kmeans.ipynb](clustering_kmeans.ipynb).

# Clustering with K-means 


```python
%matplotlib inline
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn import cluster
```


```python
data, labels_true = datasets.make_blobs(n_samples=750, centers=[[1,1],[0,5],[2,8]], cluster_std=0.7,
                            random_state=0)


plt.scatter(data[:,0], data[:,1])

df = pd.DataFrame(data, columns=['X', 'Y'])
```


    
![png](clustering_kmeans_files/clustering_kmeans_2_0.png)
    



```python
kmeans = cluster.KMeans(n_clusters=2)
label = kmeans.fit_predict(data)
df['label'] = label


fig = plt.figure()
fig.suptitle('K-means n=2', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)


for i in range(kmeans.n_clusters):
    plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i, color=plt.cm.jet(np.float(i) / len(np.unique(label))))

for i in kmeans.cluster_centers_:
    plt.scatter(i[0], i[1], color='black', marker='+', s=100)
    
plt.legend(bbox_to_anchor=(1.25, 1))
```

    <ipython-input-1-e08db6ac7c55>:12: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i, color=plt.cm.jet(np.float(i) / len(np.unique(label))))
    <ipython-input-1-e08db6ac7c55>:12: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i, color=plt.cm.jet(np.float(i) / len(np.unique(label))))





    <matplotlib.legend.Legend at 0x12e258c40>




    
![png](clustering_kmeans_files/clustering_kmeans_3_2.png)
    


Clustering with 3 clusters


```python
kmeans = cluster.KMeans(n_clusters=3)
label = kmeans.fit_predict(data)
df['label'] = label

fig = plt.figure()
fig.suptitle('K-means n=3', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)

for i in range(kmeans.n_clusters):
    plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i, color=plt.cm.jet(np.float(i) / len(np.unique(label))))

for i in kmeans.cluster_centers_:
    plt.scatter(i[0], i[1], color='black', marker='+', s=100)
    
plt.legend(bbox_to_anchor=(1.25, 1))
```

    <ipython-input-1-76fb897fa33c>:10: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i, color=plt.cm.jet(np.float(i) / len(np.unique(label))))
    <ipython-input-1-76fb897fa33c>:10: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i, color=plt.cm.jet(np.float(i) / len(np.unique(label))))
    <ipython-input-1-76fb897fa33c>:10: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i, color=plt.cm.jet(np.float(i) / len(np.unique(label))))





    <matplotlib.legend.Legend at 0x12e3a6130>




    
![png](clustering_kmeans_files/clustering_kmeans_5_2.png)
    


Clustering with 4 clusters


```python
kmeans = cluster.KMeans(n_clusters=4)
label = kmeans.fit_predict(data)
df['label'] = label


fig = plt.figure()
fig.suptitle('K-means n=4', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)


for i in range(kmeans.n_clusters):
    plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i, color=plt.cm.jet(np.float(i) / len(np.unique(label))))
    
for i in kmeans.cluster_centers_:
    plt.scatter(i[0], i[1], color='black', marker='+', s=100)

plt.legend(bbox_to_anchor=(1.25, 1))
```

    <ipython-input-1-dae570f48ffe>:12: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i, color=plt.cm.jet(np.float(i) / len(np.unique(label))))
    <ipython-input-1-dae570f48ffe>:12: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i, color=plt.cm.jet(np.float(i) / len(np.unique(label))))
    <ipython-input-1-dae570f48ffe>:12: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i, color=plt.cm.jet(np.float(i) / len(np.unique(label))))
    <ipython-input-1-dae570f48ffe>:12: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i, color=plt.cm.jet(np.float(i) / len(np.unique(label))))





    <matplotlib.legend.Legend at 0x12e4e5910>




    
![png](clustering_kmeans_files/clustering_kmeans_7_2.png)
    



```python
kmeans = cluster.KMeans(n_clusters=5)
label = kmeans.fit_predict(data)
df['label'] = label


fig = plt.figure()
fig.suptitle('K-means n=5', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)


for i in range(kmeans.n_clusters):
    plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i, color=plt.cm.jet(np.float(i) / len(np.unique(label))))
    
for i in kmeans.cluster_centers_:
    plt.scatter(i[0], i[1], color='black', marker='+', s=100)

plt.legend(bbox_to_anchor=(1.25, 1))
```

    <ipython-input-1-0aff14253d88>:12: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i, color=plt.cm.jet(np.float(i) / len(np.unique(label))))
    <ipython-input-1-0aff14253d88>:12: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i, color=plt.cm.jet(np.float(i) / len(np.unique(label))))
    <ipython-input-1-0aff14253d88>:12: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i, color=plt.cm.jet(np.float(i) / len(np.unique(label))))
    <ipython-input-1-0aff14253d88>:12: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i, color=plt.cm.jet(np.float(i) / len(np.unique(label))))
    <ipython-input-1-0aff14253d88>:12: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i, color=plt.cm.jet(np.float(i) / len(np.unique(label))))





    <matplotlib.legend.Legend at 0x12e615820>




    
![png](clustering_kmeans_files/clustering_kmeans_8_2.png)
    



```python
data, t = datasets.make_swiss_roll(n_samples=200, noise=0.1, random_state=0)
df = pd.DataFrame(data, columns=['X', 'Y', 'Z'])

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)

ax.scatter(df.X, df.Y, df.Z, 'o')
```




    <mpl_toolkits.mplot3d.art3d.Path3DCollection at 0x12e757460>




    
![png](clustering_kmeans_files/clustering_kmeans_9_1.png)
    



```python

kmeans = cluster.KMeans(n_clusters=2)
label = kmeans.fit_predict(data, )

df['label'] = label

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
fig.suptitle('K-means n=2', fontsize=14, fontweight='bold')

for l in np.unique(label):
    
    ax.scatter(df[df.label == l].X, df[df.label == l].Y, df[df.label == l].Z, 
               'o', color=plt.cm.jet(np.float(l) / len(np.unique(label))))

```

    <ipython-input-1-cb9c11940bdc>:14: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      'o', color=plt.cm.jet(np.float(l) / len(np.unique(label))))
    <ipython-input-1-cb9c11940bdc>:14: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      'o', color=plt.cm.jet(np.float(l) / len(np.unique(label))))



    
![png](clustering_kmeans_files/clustering_kmeans_10_1.png)
    



```python

kmeans = cluster.KMeans(n_clusters=3)
label = kmeans.fit_predict(data, )

df['label'] = label

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
fig.suptitle('K-means n=3', fontsize=14, fontweight='bold')

for l in np.unique(label):
    
    ax.scatter(df[df.label == l].X, df[df.label == l].Y, df[df.label == l].Z, 
               'o', color=plt.cm.jet(np.float(l) / len(np.unique(label))))

```

    <ipython-input-1-a87e53ad7267>:14: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      'o', color=plt.cm.jet(np.float(l) / len(np.unique(label))))
    <ipython-input-1-a87e53ad7267>:14: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      'o', color=plt.cm.jet(np.float(l) / len(np.unique(label))))
    <ipython-input-1-a87e53ad7267>:14: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      'o', color=plt.cm.jet(np.float(l) / len(np.unique(label))))



    
![png](clustering_kmeans_files/clustering_kmeans_11_1.png)
    



```python

kmeans = cluster.KMeans(n_clusters=4)
label = kmeans.fit_predict(data, )

df['label'] = label

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
fig.suptitle('K-means n=4', fontsize=14, fontweight='bold')

for l in np.unique(label):
    
    ax.scatter(df[df.label == l].X, df[df.label == l].Y, df[df.label == l].Z, 
               'o', color=plt.cm.jet(np.float(l) / len(np.unique(label))))

```

    <ipython-input-1-4ae8f4bce688>:14: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      'o', color=plt.cm.jet(np.float(l) / len(np.unique(label))))
    <ipython-input-1-4ae8f4bce688>:14: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      'o', color=plt.cm.jet(np.float(l) / len(np.unique(label))))
    <ipython-input-1-4ae8f4bce688>:14: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      'o', color=plt.cm.jet(np.float(l) / len(np.unique(label))))
    <ipython-input-1-4ae8f4bce688>:14: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      'o', color=plt.cm.jet(np.float(l) / len(np.unique(label))))



    
![png](clustering_kmeans_files/clustering_kmeans_12_1.png)
    



```python

kmeans = cluster.KMeans(n_clusters=5)
label = kmeans.fit_predict(data, )

df['label'] = label

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
fig.suptitle('K-means n=5', fontsize=14, fontweight='bold')

for l in np.unique(label):
    
    ax.scatter(df[df.label == l].X, df[df.label == l].Y, df[df.label == l].Z, 
               'o', color=plt.cm.jet(np.float(l) / len(np.unique(label))))

```

    <ipython-input-1-cf01c1d1d2ad>:14: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      'o', color=plt.cm.jet(np.float(l) / len(np.unique(label))))
    <ipython-input-1-cf01c1d1d2ad>:14: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      'o', color=plt.cm.jet(np.float(l) / len(np.unique(label))))
    <ipython-input-1-cf01c1d1d2ad>:14: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      'o', color=plt.cm.jet(np.float(l) / len(np.unique(label))))
    <ipython-input-1-cf01c1d1d2ad>:14: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      'o', color=plt.cm.jet(np.float(l) / len(np.unique(label))))
    <ipython-input-1-cf01c1d1d2ad>:14: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      'o', color=plt.cm.jet(np.float(l) / len(np.unique(label))))



    
![png](clustering_kmeans_files/clustering_kmeans_13_1.png)
    
