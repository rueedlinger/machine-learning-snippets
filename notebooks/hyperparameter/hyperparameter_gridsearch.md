>**Note**: This is a generated markdown export from the Jupyter notebook file [hyperparameter_gridsearch.ipynb](hyperparameter_gridsearch.ipynb).
>You can also view the notebook with the [nbviewer](https://nbviewer.jupyter.org/github/rueedlinger/machine-learning-snippets/blob/master/notebooks/hyperparameter/hyperparameter_gridsearch.ipynb) from Jupyter. 

# Hyperparameter optimization with GridSearch


```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn import svm, datasets, metrics, model_selection, preprocessing, pipeline
```

Load the data set


```python
wine = datasets.load_wine()
print(wine.DESCR)
```

    .. _wine_dataset:
    
    Wine recognition dataset
    ------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 178 (50 in each of three classes)
        :Number of Attributes: 13 numeric, predictive attributes and the class
        :Attribute Information:
     		- Alcohol
     		- Malic acid
     		- Ash
    		- Alcalinity of ash  
     		- Magnesium
    		- Total phenols
     		- Flavanoids
     		- Nonflavanoid phenols
     		- Proanthocyanins
    		- Color intensity
     		- Hue
     		- OD280/OD315 of diluted wines
     		- Proline
    
        - class:
                - class_0
                - class_1
                - class_2
    		
        :Summary Statistics:
        
        ============================= ==== ===== ======= =====
                                       Min   Max   Mean     SD
        ============================= ==== ===== ======= =====
        Alcohol:                      11.0  14.8    13.0   0.8
        Malic Acid:                   0.74  5.80    2.34  1.12
        Ash:                          1.36  3.23    2.36  0.27
        Alcalinity of Ash:            10.6  30.0    19.5   3.3
        Magnesium:                    70.0 162.0    99.7  14.3
        Total Phenols:                0.98  3.88    2.29  0.63
        Flavanoids:                   0.34  5.08    2.03  1.00
        Nonflavanoid Phenols:         0.13  0.66    0.36  0.12
        Proanthocyanins:              0.41  3.58    1.59  0.57
        Colour Intensity:              1.3  13.0     5.1   2.3
        Hue:                          0.48  1.71    0.96  0.23
        OD280/OD315 of diluted wines: 1.27  4.00    2.61  0.71
        Proline:                       278  1680     746   315
        ============================= ==== ===== ======= =====
    
        :Missing Attribute Values: None
        :Class Distribution: class_0 (59), class_1 (71), class_2 (48)
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988
    
    This is a copy of UCI ML Wine recognition datasets.
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
    
    The data is the results of a chemical analysis of wines grown in the same
    region in Italy by three different cultivators. There are thirteen different
    measurements taken for different constituents found in the three types of
    wine.
    
    Original Owners: 
    
    Forina, M. et al, PARVUS - 
    An Extendible Package for Data Exploration, Classification and Correlation. 
    Institute of Pharmaceutical and Food Analysis and Technologies,
    Via Brigata Salerno, 16147 Genoa, Italy.
    
    Citation:
    
    Lichman, M. (2013). UCI Machine Learning Repository
    [https://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
    School of Information and Computer Science. 
    
    .. topic:: References
    
      (1) S. Aeberhard, D. Coomans and O. de Vel, 
      Comparison of Classifiers in High Dimensional Settings, 
      Tech. Rep. no. 92-02, (1992), Dept. of Computer Science and Dept. of  
      Mathematics and Statistics, James Cook University of North Queensland. 
      (Also submitted to Technometrics). 
    
      The data was used with many others for comparing various 
      classifiers. The classes are separable, though only RDA 
      has achieved 100% correct classification. 
      (RDA : 100%, QDA 99.4%, LDA 98.9%, 1NN 96.1% (z-transformed data)) 
      (All results using the leave-one-out technique) 
    
      (2) S. Aeberhard, D. Coomans and O. de Vel, 
      "THE CLASSIFICATION PERFORMANCE OF RDA" 
      Tech. Rep. no. 92-01, (1992), Dept. of Computer Science and Dept. of 
      Mathematics and Statistics, James Cook University of North Queensland. 
      (Also submitted to Journal of Chemometrics).
    



```python
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target
```

Stratify the data by the target label


```python
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.5, stratify=y)

df_train = pd.DataFrame(y_train, columns=['target'])
df_train['type'] = 'train'

df_test = pd.DataFrame(y_test, columns=['target'])
df_test['type'] = 'test'

df_set = df_train.append(df_test)

_ = sns.countplot(x='target', hue='type', data=df_set)     

print('train samples:', len(X_train))
print('test samples', len(X_test))
```

    train samples: 89
    test samples 89



    
![png](hyperparameter_gridsearch_files/hyperparameter_gridsearch_6_1.png)
    



```python
parameters = {
            'kernel':('linear', 'rbf', 'sigmoid'), 
            'C':[1, 10], 'degree': [3,4], 
            'decision_function_shape': ['ovo', 'ovr']
        }

estimator = svm.SVC()

model = model_selection.GridSearchCV(estimator, parameters)
model.fit(X_train, y_train)
```




    GridSearchCV(estimator=SVC(),
                 param_grid={'C': [1, 10],
                             'decision_function_shape': ['ovo', 'ovr'],
                             'degree': [3, 4],
                             'kernel': ('linear', 'rbf', 'sigmoid')})



What was the best estimator?


```python
model.best_estimator_
```




    SVC(C=1, decision_function_shape='ovo', kernel='linear')



Let's print some more deatils


```python
results_df = pd.DataFrame(model.cv_results_)
results_df = results_df.sort_values(by=['rank_test_score'])
results_df = (
    results_df
    .set_index(results_df["params"].apply(
        lambda x: "_".join(str(val) for val in x.values()))
    )
    .rename_axis('model')
)
results_df[
    ['params', 'rank_test_score', 'mean_test_score', 'std_test_score']
]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>params</th>
      <th>rank_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1_ovo_3_linear</th>
      <td>{'C': 1, 'decision_function_shape': 'ovo', 'de...</td>
      <td>1</td>
      <td>0.920261</td>
      <td>0.059782</td>
    </tr>
    <tr>
      <th>10_ovr_4_linear</th>
      <td>{'C': 10, 'decision_function_shape': 'ovr', 'd...</td>
      <td>1</td>
      <td>0.920261</td>
      <td>0.059782</td>
    </tr>
    <tr>
      <th>1_ovo_4_linear</th>
      <td>{'C': 1, 'decision_function_shape': 'ovo', 'de...</td>
      <td>1</td>
      <td>0.920261</td>
      <td>0.059782</td>
    </tr>
    <tr>
      <th>1_ovr_3_linear</th>
      <td>{'C': 1, 'decision_function_shape': 'ovr', 'de...</td>
      <td>1</td>
      <td>0.920261</td>
      <td>0.059782</td>
    </tr>
    <tr>
      <th>10_ovr_3_linear</th>
      <td>{'C': 10, 'decision_function_shape': 'ovr', 'd...</td>
      <td>1</td>
      <td>0.920261</td>
      <td>0.059782</td>
    </tr>
    <tr>
      <th>1_ovr_4_linear</th>
      <td>{'C': 1, 'decision_function_shape': 'ovr', 'de...</td>
      <td>1</td>
      <td>0.920261</td>
      <td>0.059782</td>
    </tr>
    <tr>
      <th>10_ovo_3_linear</th>
      <td>{'C': 10, 'decision_function_shape': 'ovo', 'd...</td>
      <td>1</td>
      <td>0.920261</td>
      <td>0.059782</td>
    </tr>
    <tr>
      <th>10_ovo_4_linear</th>
      <td>{'C': 10, 'decision_function_shape': 'ovo', 'd...</td>
      <td>1</td>
      <td>0.920261</td>
      <td>0.059782</td>
    </tr>
    <tr>
      <th>10_ovr_3_rbf</th>
      <td>{'C': 10, 'decision_function_shape': 'ovr', 'd...</td>
      <td>9</td>
      <td>0.684314</td>
      <td>0.072387</td>
    </tr>
    <tr>
      <th>10_ovr_4_rbf</th>
      <td>{'C': 10, 'decision_function_shape': 'ovr', 'd...</td>
      <td>9</td>
      <td>0.684314</td>
      <td>0.072387</td>
    </tr>
    <tr>
      <th>10_ovo_3_rbf</th>
      <td>{'C': 10, 'decision_function_shape': 'ovo', 'd...</td>
      <td>9</td>
      <td>0.684314</td>
      <td>0.072387</td>
    </tr>
    <tr>
      <th>10_ovo_4_rbf</th>
      <td>{'C': 10, 'decision_function_shape': 'ovo', 'd...</td>
      <td>9</td>
      <td>0.684314</td>
      <td>0.072387</td>
    </tr>
    <tr>
      <th>1_ovo_3_rbf</th>
      <td>{'C': 1, 'decision_function_shape': 'ovo', 'de...</td>
      <td>13</td>
      <td>0.650327</td>
      <td>0.073162</td>
    </tr>
    <tr>
      <th>1_ovo_4_rbf</th>
      <td>{'C': 1, 'decision_function_shape': 'ovo', 'de...</td>
      <td>13</td>
      <td>0.650327</td>
      <td>0.073162</td>
    </tr>
    <tr>
      <th>1_ovr_3_rbf</th>
      <td>{'C': 1, 'decision_function_shape': 'ovr', 'de...</td>
      <td>13</td>
      <td>0.650327</td>
      <td>0.073162</td>
    </tr>
    <tr>
      <th>1_ovr_4_rbf</th>
      <td>{'C': 1, 'decision_function_shape': 'ovr', 'de...</td>
      <td>13</td>
      <td>0.650327</td>
      <td>0.073162</td>
    </tr>
    <tr>
      <th>1_ovr_4_sigmoid</th>
      <td>{'C': 1, 'decision_function_shape': 'ovr', 'de...</td>
      <td>17</td>
      <td>0.393464</td>
      <td>0.009150</td>
    </tr>
    <tr>
      <th>1_ovr_3_sigmoid</th>
      <td>{'C': 1, 'decision_function_shape': 'ovr', 'de...</td>
      <td>17</td>
      <td>0.393464</td>
      <td>0.009150</td>
    </tr>
    <tr>
      <th>1_ovo_4_sigmoid</th>
      <td>{'C': 1, 'decision_function_shape': 'ovo', 'de...</td>
      <td>17</td>
      <td>0.393464</td>
      <td>0.009150</td>
    </tr>
    <tr>
      <th>1_ovo_3_sigmoid</th>
      <td>{'C': 1, 'decision_function_shape': 'ovo', 'de...</td>
      <td>17</td>
      <td>0.393464</td>
      <td>0.009150</td>
    </tr>
    <tr>
      <th>10_ovo_4_sigmoid</th>
      <td>{'C': 10, 'decision_function_shape': 'ovo', 'd...</td>
      <td>21</td>
      <td>0.147059</td>
      <td>0.069846</td>
    </tr>
    <tr>
      <th>10_ovr_3_sigmoid</th>
      <td>{'C': 10, 'decision_function_shape': 'ovr', 'd...</td>
      <td>21</td>
      <td>0.147059</td>
      <td>0.069846</td>
    </tr>
    <tr>
      <th>10_ovo_3_sigmoid</th>
      <td>{'C': 10, 'decision_function_shape': 'ovo', 'd...</td>
      <td>21</td>
      <td>0.147059</td>
      <td>0.069846</td>
    </tr>
    <tr>
      <th>10_ovr_4_sigmoid</th>
      <td>{'C': 10, 'decision_function_shape': 'ovr', 'd...</td>
      <td>21</td>
      <td>0.147059</td>
      <td>0.069846</td>
    </tr>
  </tbody>
</table>
</div>




```python
predicted = model.predict(X_test)

truth_table = pd.DataFrame(predicted, columns=['target_predicted'])
truth_table['target_truth'] = y_test

truth_table = truth_table.groupby(['target_predicted', 'target_truth']).size().unstack().fillna(0)

truth_table
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>target_truth</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
    <tr>
      <th>target_predicted</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>32.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
_ = sns.heatmap(truth_table, annot=True, cmap="Blues")
```


    
![png](hyperparameter_gridsearch_files/hyperparameter_gridsearch_13_0.png)
    



```python
print("accuracy: {:.3f}".format(metrics.accuracy_score(y_test, predicted)))
print("precision: {:.3f}".format(metrics.precision_score(y_test, predicted, average='weighted')))
print("recall: {:.3f}".format(metrics.recall_score(y_test, predicted, average='weighted')))
print("f1 score: {:.3f}".format(metrics.f1_score(y_test, predicted, average='weighted')))
```

    accuracy: 0.944
    precision: 0.945
    recall: 0.944
    f1 score: 0.943
