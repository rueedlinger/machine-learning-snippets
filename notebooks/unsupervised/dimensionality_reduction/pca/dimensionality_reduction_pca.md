>**Note**: This is a generated output from the Jupyter notebook file [dimensionality_reduction_pca.ipynb](dimensionality_reduction_pca.ipynb).

# Dimensionality Reduction with PCA (SVD)


```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn import decomposition, datasets
from matplotlib.colors import ListedColormap


```


```python
iris = datasets.load_iris()
pca = decomposition.PCA(n_components=2)
new_dim = pca.fit_transform(iris.data)
```


```python
df = pd.DataFrame(new_dim, columns=['X', 'Y'])
df['label'] = iris.target
df.head()
```




<div>
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
      <td>-2.684207</td>
      <td>0.326607</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.715391</td>
      <td>-0.169557</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.889820</td>
      <td>-0.137346</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.746437</td>
      <td>-0.311124</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.728593</td>
      <td>0.333925</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python

fig = plt.figure()
fig.suptitle('PCA (SVD)', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)

plt.scatter(df[df.label == 0].X, df[df.label == 0].Y, color='red', label=iris.target_names[0])
plt.scatter(df[df.label == 1].X, df[df.label == 1].Y, color='blue', label=iris.target_names[1])
plt.scatter(df[df.label == 2].X, df[df.label == 2].Y, color='green', label=iris.target_names[2])

plt.legend(bbox_to_anchor=(1.25, 1))
```




    <matplotlib.legend.Legend at 0x10b807e48>




    
![png](dimensionality_reduction_pca_files/dimensionality_reduction_pca_4_1.png)
    
