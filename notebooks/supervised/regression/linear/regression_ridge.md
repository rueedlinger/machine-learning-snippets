> **Note**: This is a generated markdown export from the Jupyter notebook file [regression_ridge.ipynb](regression_ridge.ipynb).

## Ridge Regression (regularized linear regression model)

```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn import linear_model, datasets, metrics, model_selection, preprocessing, pipeline
```

Load the data set

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
model = linear_model.Ridge(alpha=.5)
model.fit(X_train, y_train)
```

    Ridge(alpha=0.5)

```python
predicted = model.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, predicted)

ax.set_xlabel('True Values')
ax.set_ylabel('Predicted')
_ = ax.plot([0, y.max()], [0, y.max()], ls='-', color='red')
```

![png](regression_ridge_files/regression_ridge_7_0.png)

```python
residual = y_test - predicted

fig, ax = plt.subplots()
ax.scatter(y_test, residual)
ax.set_xlabel('y')
ax.set_ylabel('residual')

_ = plt.axhline(0, color='red', ls='--')
```

![png](regression_ridge_files/regression_ridge_8_0.png)

```python
sns.displot(residual, kind="kde");
```

    <seaborn.axisgrid.FacetGrid at 0x123884370>

![png](regression_ridge_files/regression_ridge_9_1.png)

```python
print("r2 score: {}".format(metrics.r2_score(y_test, predicted)))
print("mse: {}".format(metrics.mean_squared_error(y_test, predicted)))
print("rmse: {}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
print("mae: {}".format(metrics.mean_absolute_error(y_test, predicted)))
```

    r2 score: 0.6943114047110797
    mse: 25.553430027270117
    rmse: 5.055040061885773
    mae: 3.412194557705958

```python

```