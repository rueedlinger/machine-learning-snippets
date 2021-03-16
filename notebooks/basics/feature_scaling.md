>**Note**: This is a generated markdown export from the Jupyter notebook file [feature_scaling.ipynb](feature_scaling.ipynb).

# Feature scaling


```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn import preprocessing
```


```python
size = 1000

n_skew = np.random.beta(a=20, b=2, size=size)
p_skew = np.random.beta(a=3, b=30, size=size)
norm = np.random.normal(loc=500, scale=16, size=size)
outlier = np.random.normal(loc=500, scale=16, size=size-2)
outlier = np.concatenate((outlier, [700, 800]), axis=0)

X = pd.DataFrame(np.column_stack((p_skew, norm, n_skew, outlier)), columns=['p_skew', 'normal', 'n_skew', 'outlier'])

X.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>p_skew</th>
      <th>normal</th>
      <th>n_skew</th>
      <th>outlier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.089754</td>
      <td>500.666476</td>
      <td>0.908593</td>
      <td>501.301288</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.049710</td>
      <td>15.793393</td>
      <td>0.058397</td>
      <td>19.455593</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.004194</td>
      <td>455.478980</td>
      <td>0.658840</td>
      <td>450.522825</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.054549</td>
      <td>490.384003</td>
      <td>0.876700</td>
      <td>490.182717</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.081001</td>
      <td>500.759498</td>
      <td>0.919990</td>
      <td>500.493809</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.115693</td>
      <td>511.051551</td>
      <td>0.952271</td>
      <td>510.776513</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.326618</td>
      <td>542.010190</td>
      <td>0.998339</td>
      <td>800.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

_ = sns.histplot(ax=axes[0,0], data=X['normal'], kde=True)
_ = sns.boxplot(ax=axes[1,0], data=X['normal'], orient='h')
_ = sns.histplot(ax=axes[0,1], data=X['n_skew'], kde=True)
_ = sns.boxplot(ax=axes[1,1], data=X['n_skew'], orient='h')
_ = sns.histplot(ax=axes[0,2], data=X['p_skew'], kde=True,)
_ = sns.boxplot(ax=axes[1,2], data=X['p_skew'], orient='h')
_ = sns.histplot(ax=axes[0,3], data=X['outlier'], kde=True,)
_ = sns.boxplot(ax=axes[1,3], data=X['outlier'], orient='h')
```


    
![png](feature_scaling_files/feature_scaling_3_0.png)
    


## Min / Max


```python
scaler = preprocessing.MinMaxScaler()

X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
X_scaled.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>p_skew</th>
      <th>normal</th>
      <th>n_skew</th>
      <th>outlier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.265365</td>
      <td>0.522210</td>
      <td>0.735651</td>
      <td>0.145298</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.154175</td>
      <td>0.182517</td>
      <td>0.172009</td>
      <td>0.055671</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.156175</td>
      <td>0.403381</td>
      <td>0.641711</td>
      <td>0.113483</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.238217</td>
      <td>0.523285</td>
      <td>0.769221</td>
      <td>0.142988</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.345814</td>
      <td>0.642226</td>
      <td>0.864307</td>
      <td>0.172411</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

_ = sns.histplot(ax=axes[0,0], data=X_scaled['normal'], kde=True)
_ = sns.boxplot(ax=axes[1,0], data=X_scaled['normal'], orient='h')
_ = sns.histplot(ax=axes[0,1], data=X_scaled['n_skew'], kde=True)
_ = sns.boxplot(ax=axes[1,1], data=X_scaled['n_skew'], orient='h')
_ = sns.histplot(ax=axes[0,2], data=X_scaled['p_skew'], kde=True,)
_ = sns.boxplot(ax=axes[1,2], data=X_scaled['p_skew'], orient='h')
_ = sns.histplot(ax=axes[0,3], data=X_scaled['outlier'], kde=True,)
_ = sns.boxplot(ax=axes[1,3], data=X_scaled['outlier'], orient='h')
```


    
![png](feature_scaling_files/feature_scaling_6_0.png)
    


# MaxAbsScaler


```python
scaler = preprocessing.MaxAbsScaler()

X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
X_scaled.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>p_skew</th>
      <th>normal</th>
      <th>n_skew</th>
      <th>outlier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.274799</td>
      <td>0.923722</td>
      <td>0.910105</td>
      <td>0.626627</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.152195</td>
      <td>0.029139</td>
      <td>0.058494</td>
      <td>0.024319</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.012842</td>
      <td>0.840351</td>
      <td>0.659936</td>
      <td>0.563154</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.167011</td>
      <td>0.904751</td>
      <td>0.878159</td>
      <td>0.612728</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.247999</td>
      <td>0.923893</td>
      <td>0.921521</td>
      <td>0.625617</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.354215</td>
      <td>0.942882</td>
      <td>0.953856</td>
      <td>0.638471</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

_ = sns.histplot(ax=axes[0,0], data=X_scaled['normal'], kde=True)
_ = sns.boxplot(ax=axes[1,0], data=X_scaled['normal'], orient='h')
_ = sns.histplot(ax=axes[0,1], data=X_scaled['n_skew'], kde=True)
_ = sns.boxplot(ax=axes[1,1], data=X_scaled['n_skew'], orient='h')
_ = sns.histplot(ax=axes[0,2], data=X_scaled['p_skew'], kde=True,)
_ = sns.boxplot(ax=axes[1,2], data=X_scaled['p_skew'], orient='h')
_ = sns.histplot(ax=axes[0,3], data=X_scaled['outlier'], kde=True,)
_ = sns.boxplot(ax=axes[1,3], data=X_scaled['outlier'], orient='h')
```


    
![png](feature_scaling_files/feature_scaling_9_0.png)
    


## RobustScaler


```python
scaler = preprocessing.RobustScaler()

X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
X_scaled.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>p_skew</th>
      <th>normal</th>
      <th>n_skew</th>
      <th>outlier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.143160</td>
      <td>-0.004501</td>
      <td>-0.150811</td>
      <td>0.039210</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.812994</td>
      <td>0.764164</td>
      <td>0.772741</td>
      <td>0.944731</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.256159</td>
      <td>-2.190899</td>
      <td>-3.455682</td>
      <td>-2.426507</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.432619</td>
      <td>-0.502019</td>
      <td>-0.572835</td>
      <td>-0.500689</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.567381</td>
      <td>0.497981</td>
      <td>0.427165</td>
      <td>0.499311</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.017020</td>
      <td>1.995916</td>
      <td>1.036759</td>
      <td>14.543516</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

_ = sns.histplot(ax=axes[0,0], data=X_scaled['normal'], kde=True)
_ = sns.boxplot(ax=axes[1,0], data=X_scaled['normal'], orient='h')
_ = sns.histplot(ax=axes[0,1], data=X_scaled['n_skew'], kde=True)
_ = sns.boxplot(ax=axes[1,1], data=X_scaled['n_skew'], orient='h')
_ = sns.histplot(ax=axes[0,2], data=X_scaled['p_skew'], kde=True,)
_ = sns.boxplot(ax=axes[1,2], data=X_scaled['p_skew'], orient='h')
_ = sns.histplot(ax=axes[0,3], data=X_scaled['outlier'], kde=True,)
_ = sns.boxplot(ax=axes[1,3], data=X_scaled['outlier'], orient='h')
```


    
![png](feature_scaling_files/feature_scaling_12_0.png)
    


## PowerTransformer
Apply a power transform featurewise to make data more Gaussian-like.


```python
scaler = preprocessing.PowerTransformer()

X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
X_scaled.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>p_skew</th>
      <th>normal</th>
      <th>n_skew</th>
      <th>outlier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.000000e+03</td>
      <td>1.000000e+03</td>
      <td>1.000000e+03</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.767209e-15</td>
      <td>2.024692e-14</td>
      <td>-4.666489e-15</td>
      <td>-0.732310</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000500e+00</td>
      <td>1.000500e+00</td>
      <td>1.000500e+00</td>
      <td>0.681313</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.499771e+00</td>
      <td>-2.825568e+00</td>
      <td>-2.453256e+00</td>
      <td>-3.536454</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-7.011547e-01</td>
      <td>-6.544778e-01</td>
      <td>-7.380343e-01</td>
      <td>-1.147796</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-1.023066e-02</td>
      <td>7.748112e-04</td>
      <td>2.973444e-02</td>
      <td>-0.697984</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.058112e-01</td>
      <td>6.551258e-01</td>
      <td>7.628591e-01</td>
      <td>-0.294704</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.672358e+00</td>
      <td>2.649060e+00</td>
      <td>2.112185e+00</td>
      <td>3.140929</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

_ = sns.histplot(ax=axes[0,0], data=X_scaled['normal'], kde=True)
_ = sns.boxplot(ax=axes[1,0], data=X_scaled['normal'], orient='h')
_ = sns.histplot(ax=axes[0,1], data=X_scaled['n_skew'], kde=True)
_ = sns.boxplot(ax=axes[1,1], data=X_scaled['n_skew'], orient='h')
_ = sns.histplot(ax=axes[0,2], data=X_scaled['p_skew'], kde=True,)
_ = sns.boxplot(ax=axes[1,2], data=X_scaled['p_skew'], orient='h')
_ = sns.histplot(ax=axes[0,3], data=X_scaled['outlier'], kde=True,)
_ = sns.boxplot(ax=axes[1,3], data=X_scaled['outlier'], orient='h')
```


    
![png](feature_scaling_files/feature_scaling_15_0.png)
    


## Standardization
Standardize features by removing the mean and scaling to unit variance


```python
scaler = preprocessing.StandardScaler()

X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
X_scaled.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>p_skew</th>
      <th>normal</th>
      <th>n_skew</th>
      <th>outlier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.000000e+03</td>
      <td>1.000000e+03</td>
      <td>1.000000e+03</td>
      <td>1.000000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-1.489475e-15</td>
      <td>-1.223199e-14</td>
      <td>-1.296740e-14</td>
      <td>4.902745e-16</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000500e+00</td>
      <td>1.000500e+00</td>
      <td>1.000500e+00</td>
      <td>1.000500e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.722054e+00</td>
      <td>-2.862596e+00</td>
      <td>-4.278956e+00</td>
      <td>-2.611273e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-7.085745e-01</td>
      <td>-6.513875e-01</td>
      <td>-5.464115e-01</td>
      <td>-5.717705e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-1.761775e-01</td>
      <td>5.892894e-03</td>
      <td>1.952619e-01</td>
      <td>-4.152449e-02</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.220628e-01</td>
      <td>6.578872e-01</td>
      <td>7.483307e-01</td>
      <td>4.872618e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.767317e+00</td>
      <td>2.619095e+00</td>
      <td>1.537598e+00</td>
      <td>1.536053e+01</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

_ = sns.histplot(ax=axes[0,0], data=X_scaled['normal'], kde=True)
_ = sns.boxplot(ax=axes[1,0], data=X_scaled['normal'], orient='h')
_ = sns.histplot(ax=axes[0,1], data=X_scaled['n_skew'], kde=True)
_ = sns.boxplot(ax=axes[1,1], data=X_scaled['n_skew'], orient='h')
_ = sns.histplot(ax=axes[0,2], data=X_scaled['p_skew'], kde=True,)
_ = sns.boxplot(ax=axes[1,2], data=X_scaled['p_skew'], orient='h')
_ = sns.histplot(ax=axes[0,3], data=X_scaled['outlier'], kde=True,)
_ = sns.boxplot(ax=axes[1,3], data=X_scaled['outlier'], orient='h')
```


    
![png](feature_scaling_files/feature_scaling_18_0.png)
    
