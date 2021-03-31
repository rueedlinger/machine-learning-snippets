>**Note**: This is a generated markdown export from the Jupyter notebook file [anomaly_detection_with_autoencoder.ipynb](anomaly_detection_with_autoencoder.ipynb).
>You can also view the notebook with the [nbviewer](https://nbviewer.jupyter.org/github/rueedlinger/machine-learning-snippets/blob/master/notebooks/unsupervised/neural_net/anomaly_detection_with_autoencoder.ipynb) from Jupyter. 

# Anomaly detection with an Autoencoder


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
digits = datasets.load_digits()

fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r)
    ax.set_title('%i' % label)
```


    
![png](anomaly_detection_with_autoencoder_files/anomaly_detection_with_autoencoder_2_0.png)
    



```python
target = digits.target
data = digits.images

print("min value: {}".format(np.amin(data)))
print("max value: {}".format(np.amax(data)))
print("shape: {}".format(np.shape(data)))
```

    min value: 0.0
    max value: 16.0
    shape: (1797, 8, 8)



```python
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    data, target, test_size=0.5)


X_train = X_train.astype('float32') / 16.
X_test = X_test.astype('float32') / 16.


df_train = pd.DataFrame(y_train, columns=['target'])
df_train['type'] = 'train'

df_test = pd.DataFrame(y_test, columns=['target'])
df_test['type'] = 'test'

df_set = df_train.append(df_test)

_ = sns.countplot(x='target', hue='type', data=df_set)     

print('train samples:', len(X_train))
print('test samples', len(X_test))
```

    train samples: 898
    test samples 899



    
![png](anomaly_detection_with_autoencoder_files/anomaly_detection_with_autoencoder_4_1.png)
    



```python
class Autoencoder(tf.keras.models.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(64, activation='sigmoid'),
            tf.keras.layers.Reshape((8, 8))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder()


autoencoder.compile(optimizer='adam', loss='mse')



```


```python
%%time
history = autoencoder.fit(X_train, X_train,
            epochs=100,
            validation_split = 0.2,
            validation_data=(X_test, X_test),
            verbose=0)

```

    CPU times: user 7.87 s, sys: 1.23 s, total: 9.09 s
    Wall time: 6.92 s



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
      <td>0.023893</td>
      <td>0.023797</td>
      <td>95</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.023811</td>
      <td>0.023788</td>
      <td>96</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.023782</td>
      <td>0.023722</td>
      <td>97</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.023789</td>
      <td>0.023617</td>
      <td>98</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.023696</td>
      <td>0.023929</td>
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


    
![png](anomaly_detection_with_autoencoder_files/anomaly_detection_with_autoencoder_8_0.png)
    



```python
reconstructions = autoencoder.predict(digits.images)


fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(10, 3))
for ax, image, label in zip(axes, reconstructions, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r)
    ax.set_title('%i' % label)
```


    
![png](anomaly_detection_with_autoencoder_files/anomaly_detection_with_autoencoder_9_0.png)
    



```python
reconstruction_error_train = np.mean(tf.keras.losses.mae(autoencoder.predict(X_train), X_train), axis=-1)
reconstruction_error_test = np.mean(tf.keras.losses.mae(autoencoder.predict(X_test), X_test), axis=-1)

df_train = pd.DataFrame(reconstruction_error_train, columns=['reconstruction_error'])
df_train['type'] = 'train'

df_test = pd.DataFrame(reconstruction_error_test, columns=['reconstruction_error'])
df_test['type'] = 'test'

df_set = df_train.append(df_test)


fig, axs = plt.subplots(nrows=2, figsize=(10, 5))
fig.suptitle('Reconstruction error', fontsize=16)

p_threshold = 99
threshold = np.percentile(reconstruction_error_test, p_threshold)

x_max = np.max(reconstruction_error_test) + np.std(reconstruction_error_test)


axs[0].axvline(threshold, color='r', ls='--')
axs[0].set(xlim=(0, x_max))

axs[0].text(0.85, 0.2, 'threshold {:.3f}
(percentile: {})'.format(threshold, p_threshold), 
            horizontalalignment='left', verticalalignment='center', transform=axs[0].transAxes)


axs[1].axvline(threshold, color='r', ls='--')
axs[1].set(xlim=(0, x_max))


_ = sns.kdeplot(data=df_set, x='reconstruction_error' ,hue='type', ax=axs[0])
_ = sns.boxplot(data=df_set, x='reconstruction_error', y='type', orient='h', ax=axs[1])


```


    
![png](anomaly_detection_with_autoencoder_files/anomaly_detection_with_autoencoder_10_0.png)
    



```python
anomalies_index = np.argwhere(reconstruction_error_test > threshold).flatten()

anomalies_x = np.array(X_test)[anomalies_index] 
anomalies_y = np.array(y_test)[anomalies_index] 


fig, axes = plt.subplots(nrows=1, ncols=len(anomalies_x), figsize=(10, 3))
fig.suptitle('Samples with reconstruction error > {:.3f} (percentile: {})'.format(threshold, p_threshold), fontsize=16)

for ax, image, label, in zip(axes, anomalies_x, anomalies_y):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r)
    ax.set_title('%i' % label)

```


    
![png](anomaly_detection_with_autoencoder_files/anomaly_detection_with_autoencoder_11_0.png)
    



```python
_ = sns.countplot(x=anomalies_y).set_title('Reconstruction error by target')     
```


    
![png](anomaly_detection_with_autoencoder_files/anomaly_detection_with_autoencoder_12_0.png)
    



```python
flipped_images = np.array([np.transpose(x) for x in digits.images[0:10]])
flipped_images = flipped_images / 16.
flipped_images

reconstruction_error_flipped_images = np.mean(tf.keras.losses.mae(autoencoder.predict(flipped_images), flipped_images), axis=-1) 
is_anomaly = reconstruction_error_flipped_images > threshold
```


```python
fig, axes = plt.subplots(nrows=1, ncols=len(flipped_images), figsize=(10, 2))
fig.suptitle('Flipped images'.format(threshold, p_threshold), fontsize=16)
for ax, image, anomaly in zip(axes, flipped_images, is_anomaly):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r)
    if anomaly:
        ax.set_title('anomaly')
```


    
![png](anomaly_detection_with_autoencoder_files/anomaly_detection_with_autoencoder_14_0.png)
    



```python
pd.DataFrame(reconstruction_error_flipped_images, columns=['reconstruction_error'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reconstruction_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.293820</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.219904</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.339496</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.316413</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.251248</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.287674</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.305904</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.311851</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.344047</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.310956</td>
    </tr>
  </tbody>
</table>
</div>
