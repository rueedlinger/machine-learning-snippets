>**Note**: This is a generated markdown export from the Jupyter notebook file [dimensionality_reduction_tsne.ipynb](dimensionality_reduction_tsne.ipynb).

# Dimensionality Reduction with t-SNE


```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn import manifold, datasets
from matplotlib.colors import ListedColormap

```


```python
iris = datasets.load_iris()
tsne = manifold.TSNE(n_components=2, learning_rate=100)
new_dim = tsne.fit_transform(iris.data)
```


```python
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
      <td>-7.131333</td>
      <td>22.109650</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-6.575503</td>
      <td>19.641376</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-7.623905</td>
      <td>19.355234</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-7.402305</td>
      <td>19.086344</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-6.765808</td>
      <td>22.209557</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = plt.figure()
fig.suptitle('t-SNE', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)

plt.scatter(df[df.label == 0].X, df[df.label == 0].Y, color='red', label=iris.target_names[0])
plt.scatter(df[df.label == 1].X, df[df.label == 1].Y, color='blue', label=iris.target_names[1])
plt.scatter(df[df.label == 2].X, df[df.label == 2].Y, color='green', label=iris.target_names[2])

plt.legend(bbox_to_anchor=(1.25, 1))
```




    <matplotlib.legend.Legend at 0x129a647c0>




    
![png](dimensionality_reduction_tsne_files/dimensionality_reduction_tsne_4_1.png)
    