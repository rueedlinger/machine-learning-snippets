>**Note**: This is a generated markdown export from the Jupyter notebook file [dimensionality_reduction_tsne.ipynb](dimensionality_reduction_tsne.ipynb).
>You can also view the notebook with the [nbviewer](https://nbviewer.jupyter.org/github/rueedlinger/machine-learning-snippets/blob/master/notebooks/unsupervised/dimensionality_reduction/tsne/dimensionality_reduction_tsne.ipynb) from Jupyter. 

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
      <td>-21.419571</td>
      <td>-9.280364</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-18.810642</td>
      <td>-10.166967</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-18.943447</td>
      <td>-8.992502</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-18.503231</td>
      <td>-9.370867</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-21.380730</td>
      <td>-8.861400</td>
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

_ = plt.legend(bbox_to_anchor=(1.25, 1))
```


    
![png](dimensionality_reduction_tsne_files/dimensionality_reduction_tsne_4_0.png)
    
