>**Note**: This is a generated markdown export from the Jupyter notebook file [regression_neural_net.ipynb](regression_neural_net.ipynb).
>You can also view the notebook with the [nbviewer](https://nbviewer.jupyter.org/github/rueedlinger/machine-learning-snippets/blob/master/notebooks/supervised/neural_net/regression_neural_net.ipynb) from Jupyter. 

# Regression with a neural network


```python
%matplotlib inline
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets, metrics, model_selection

```


```python
boston = datasets.load_boston()
print(boston.DESCR)
```

    .. _boston_dataset:
    
    Boston house prices dataset
    ---------------------------
    
    **Data Set Characteristics:**  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    .. topic:: References
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
    



```python
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target
```


```python
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7)

print('train samples:', len(X_train))
print('test samples', len(X_test))
```

    train samples: 354
    test samples 152



```python
df_train = pd.DataFrame(y_train, columns=['target'])
df_train['type'] = 'train'

df_test = pd.DataFrame(y_test, columns=['target'])
df_test['type'] = 'test'

df_set = df_train.append(df_test)

_ = sns.displot(df_set, x="target" ,hue="type", kind="kde", log_scale=False)
```


    
![png](regression_neural_net_files/regression_neural_net_5_0.png)
    



```python
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(X_train))

model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(units=1)
])

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    normalization (Normalization (None, 13)                27        
    _________________________________________________________________
    dense (Dense)                (None, 1)                 14        
    =================================================================
    Total params: 41
    Trainable params: 14
    Non-trainable params: 27
    _________________________________________________________________



```python
%%time
history = model.fit(X_train, y_train, epochs=100, validation_split = 0.2, verbose=0)
```

    CPU times: user 5.18 s, sys: 412 ms, total: 5.59 s
    Wall time: 5.08 s



```python
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loss</th>
      <th>val_loss</th>
      <th>epoch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>95</th>
      <td>3.102436</td>
      <td>3.449647</td>
      <td>95</td>
    </tr>
    <tr>
      <th>96</th>
      <td>3.124186</td>
      <td>3.501871</td>
      <td>96</td>
    </tr>
    <tr>
      <th>97</th>
      <td>3.085092</td>
      <td>3.450218</td>
      <td>97</td>
    </tr>
    <tr>
      <th>98</th>
      <td>3.132863</td>
      <td>3.440988</td>
      <td>98</td>
    </tr>
    <tr>
      <th>99</th>
      <td>3.093181</td>
      <td>3.529952</td>
      <td>99</td>
    </tr>
  </tbody>
</table>
</div>




```python
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)

plot_loss(history)
```


    
![png](regression_neural_net_files/regression_neural_net_9_0.png)
    



```python
predicted = [x[0] for x in model.predict(X_test)]

fig, ax = plt.subplots()
ax.scatter(y_test, predicted)

ax.set_xlabel('True Values')
ax.set_ylabel('Predicted')
_ = ax.plot([0, y.max()], [0, y.max()], ls='-', color='red')
```


    
![png](regression_neural_net_files/regression_neural_net_10_0.png)
    



```python
residual = y_test - predicted

fig, ax = plt.subplots()
ax.scatter(y_test, residual)
ax.set_xlabel('y')
ax.set_ylabel('residual')

_ = plt.axhline(0, color='red', ls='--')
```


    
![png](regression_neural_net_files/regression_neural_net_11_0.png)
    



```python
_ = sns.displot(residual, kind="kde");
```


    
![png](regression_neural_net_files/regression_neural_net_12_0.png)
    



```python
print("r2 score: {}".format(metrics.r2_score(y_test, predicted)))
print("mse: {}".format(metrics.mean_squared_error(y_test, predicted)))
print("rmse: {}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
print("mae: {}".format(metrics.mean_absolute_error(y_test, predicted)))
```

    r2 score: 0.7146802114175688
    mse: 23.324526681358662
    rmse: 4.829547254283642
    mae: 3.0605333278053686
