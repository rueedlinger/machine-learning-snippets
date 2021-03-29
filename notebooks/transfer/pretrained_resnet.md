>**Note**: This is a generated markdown export from the Jupyter notebook file [pretrained_resnet.ipynb](pretrained_resnet.ipynb).
>You can also view the notebook with the [nbviewer](https://nbviewer.jupyter.org/github/rueedlinger/machine-learning-snippets/blob/master/notebooks/transfer/pretrained_resnet.ipynb) from Jupyter. 

# Pre-trained model ResNet


```python
%matplotlib inline

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
img_path = 'test.jpg'
img = image.load_img(img_path, target_size=(224, 224))

_ = plt.imshow(img)
```


    
![png](pretrained_resnet_files/pretrained_resnet_2_0.png)
    



```python
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
np.shape(x)
```




    (1, 224, 224, 3)




```python
model = ResNet50(weights='imagenet')
preds = model.predict(x)

for e in decode_predictions(preds, top=3)[0]:
    print('{} with probability: {:.5f}'.format(e[1], e[2]))
```

    ice_bear with probability: 0.99990
    weasel with probability: 0.00007
    Arctic_fox with probability: 0.00001
