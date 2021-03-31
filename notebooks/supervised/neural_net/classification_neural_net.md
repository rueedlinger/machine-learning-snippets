>**Note**: This is a generated markdown export from the Jupyter notebook file [classification_neural_net.ipynb](classification_neural_net.ipynb).
>You can also view the notebook with the [nbviewer](https://nbviewer.jupyter.org/github/rueedlinger/machine-learning-snippets/blob/master/notebooks/supervised/neural_net/classification_neural_net.ipynb) from Jupyter. 

# Classification with a neural network


```python
%matplotlib inline
import tensorflow as tf

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


    
![png](classification_neural_net_files/classification_neural_net_2_0.png)
    



```python
plt.figure()
plt.imshow(digits.images[0], cmap=plt.cm.gray_r)
plt.colorbar()
plt.grid(False)
plt.show()
```


    
![png](classification_neural_net_files/classification_neural_net_3_0.png)
    



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



    
![png](classification_neural_net_files/classification_neural_net_5_1.png)
    



```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(8, 8)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])




model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            (None, 64)                0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               8320      
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 9,610
    Trainable params: 9,610
    Non-trainable params: 0
    _________________________________________________________________



```python
%%time
history = model.fit(X_train, y_train, epochs=100, validation_split = 0.2, verbose=0)
```

    CPU times: user 6.55 s, sys: 822 ms, total: 7.37 s
    Wall time: 5.77 s



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
      <th>accuracy</th>
      <th>val_loss</th>
      <th>val_accuracy</th>
      <th>epoch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>95</th>
      <td>0.007926</td>
      <td>1.0</td>
      <td>0.097244</td>
      <td>0.977778</td>
      <td>95</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.007718</td>
      <td>1.0</td>
      <td>0.097566</td>
      <td>0.972222</td>
      <td>96</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.007589</td>
      <td>1.0</td>
      <td>0.098171</td>
      <td>0.977778</td>
      <td>97</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.007448</td>
      <td>1.0</td>
      <td>0.098217</td>
      <td>0.977778</td>
      <td>98</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.007221</td>
      <td>1.0</td>
      <td>0.098246</td>
      <td>0.977778</td>
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


    
![png](classification_neural_net_files/classification_neural_net_9_0.png)
    



```python
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=0)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
```

    Test loss: 0.11751686781644821
    Test accuracy: 0.9688543081283569



```python
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

```


```python
predicted = [np.argmax(x) for x in probability_model.predict(X_test)]

confusion_matrix = pd.DataFrame(metrics.confusion_matrix(y_test, predicted))
confusion_matrix
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>83</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>78</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>92</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>105</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>90</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>86</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>83</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>79</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>95</td>
    </tr>
  </tbody>
</table>
</div>




```python
_ = sns.heatmap(confusion_matrix, annot=True, cmap="Blues")
```


    
![png](classification_neural_net_files/classification_neural_net_13_0.png)
    



```python
print("accuracy: {:.3f}".format(metrics.accuracy_score(y_test, predicted)))
print("precision: {:.3f}".format(metrics.precision_score(y_test, predicted, average='weighted')))
print("recall: {:.3f}".format(metrics.recall_score(y_test, predicted, average='weighted')))
print("f1 score: {:.3f}".format(metrics.f1_score(y_test, predicted, average='weighted')))
```

    accuracy: 0.969
    precision: 0.970
    recall: 0.969
    f1 score: 0.969
