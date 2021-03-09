>**Note**: This is a generated markdown export from the Jupyter notebook file [dimensionality_reduction_eigen.ipynb](dimensionality_reduction_eigen.ipynb).

# Dimensionality Reduction with Eigenvector / Eigenvalues and Correlation Matrix (PCA)

inspired by http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html#eigendecomposition---computing-eigenvectors-and-eigenvalues


```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from numpy import linalg as LA

from sklearn import datasets
```


```python
iris = datasets.load_iris()
```

First we need the correlation matrix


```python
df = pd.DataFrame(iris.data, columns=iris.feature_names)
corr = df.corr()
df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sepal length (cm)</th>
      <td>1.000000</td>
      <td>-0.117570</td>
      <td>0.871754</td>
      <td>0.817941</td>
    </tr>
    <tr>
      <th>sepal width (cm)</th>
      <td>-0.117570</td>
      <td>1.000000</td>
      <td>-0.428440</td>
      <td>-0.366126</td>
    </tr>
    <tr>
      <th>petal length (cm)</th>
      <td>0.871754</td>
      <td>-0.428440</td>
      <td>1.000000</td>
      <td>0.962865</td>
    </tr>
    <tr>
      <th>petal width (cm)</th>
      <td>0.817941</td>
      <td>-0.366126</td>
      <td>0.962865</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(corr)
```




    <AxesSubplot:>




    
![png](dimensionality_reduction_eigen_files/dimensionality_reduction_eigen_6_1.png)
    



```python
eig_vals, eig_vecs = LA.eig(corr)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs.sort(key=lambda x: x[0], reverse=True)

```

Eigenvalues


```python
pd.DataFrame([eig_vals])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.918498</td>
      <td>0.91403</td>
      <td>0.146757</td>
      <td>0.020715</td>
    </tr>
  </tbody>
</table>
</div>



Eigenvector as Principal component


```python
pd.DataFrame(eig_vecs)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.521066</td>
      <td>-0.377418</td>
      <td>-0.719566</td>
      <td>0.261286</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.269347</td>
      <td>-0.923296</td>
      <td>0.244382</td>
      <td>-0.123510</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.580413</td>
      <td>-0.024492</td>
      <td>0.142126</td>
      <td>-0.801449</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.564857</td>
      <td>-0.066942</td>
      <td>0.634273</td>
      <td>0.523597</td>
    </tr>
  </tbody>
</table>
</div>



Create the projection matrix for a new two dimensional space


```python
matrix_w = np.hstack((eig_pairs[0][1].reshape(len(corr),1),
                      eig_pairs[1][1].reshape(len(corr),1)))


pd.DataFrame(matrix_w, columns=['PC1', 'PC2'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.521066</td>
      <td>-0.377418</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.269347</td>
      <td>-0.923296</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.580413</td>
      <td>-0.024492</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.564857</td>
      <td>-0.066942</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_dim = np.dot(np.array(iris.data), matrix_w)

df = pd.DataFrame(new_dim, columns=['X', 'Y'])
df['label'] = iris.target
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.640270</td>
      <td>-5.204041</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.670730</td>
      <td>-4.666910</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.454606</td>
      <td>-4.773636</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.545517</td>
      <td>-4.648463</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.561228</td>
      <td>-5.258629</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = plt.figure()
fig.suptitle('PCA with Eigenvector', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)

plt.scatter(df[df.label == 0].X, df[df.label == 0].Y, color='red', label=iris.target_names[0])
plt.scatter(df[df.label == 1].X, df[df.label == 1].Y, color='blue', label=iris.target_names[1])
plt.scatter(df[df.label == 2].X, df[df.label == 2].Y, color='green', label=iris.target_names[2])

plt.legend(bbox_to_anchor=(1.25, 1))

```




    <matplotlib.legend.Legend at 0x130400100>




    
![png](dimensionality_reduction_eigen_files/dimensionality_reduction_eigen_15_1.png)
    
