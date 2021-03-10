> **Note**: This is a generated markdown export from the Jupyter notebook file [statistical_analysis.ipynb](statistical_analysis.ipynb).

## Statistical analysis

In this notebook we use _pandas_ and the _stats_ module from _scipy_ for some basic statistical analysis.

```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

import pandas as pd

from matplotlib import pyplot as plt
plt.style.use("ggplot")

```

First we need some data. Let'use pandas to load the _'adult'_ data set from the _UC Irvine Machine Learning Repository_ in our dataframe.

```python
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"])

# some data cleaning remove leading and trailing spaces
df['Sex'] = df['Sex'].str.strip()


df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Workclass</th>
      <th>fnlwgt</th>
      <th>Education</th>
      <th>Education-Num</th>
      <th>Martial Status</th>
      <th>Occupation</th>
      <th>Relationship</th>
      <th>Race</th>
      <th>Sex</th>
      <th>Capital Gain</th>
      <th>Capital Loss</th>
      <th>Hours per week</th>
      <th>Country</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>

### Descriptive statistics

Let's have a first look at the shape of our dataframe.

```python
df.shape
```

    (32561, 15)

What are the column names.

```python
df.columns
```

    Index(['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-Num',
           'Martial Status', 'Occupation', 'Relationship', 'Race', 'Sex',
           'Capital Gain', 'Capital Loss', 'Hours per week', 'Country', 'Target'],
          dtype='object')

We can calculate the mean, median, standard error of the mean (sem), variance, standard deviation (std) and the quantiles for every column in the dataframe

```python
df.mean()
```

    Age                   38.581647
    fnlwgt            189778.366512
    Education-Num         10.080679
    Capital Gain        1077.648844
    Capital Loss          87.303830
    Hours per week        40.437456
    dtype: float64

```python
df.median()
```

    Age                   37.0
    fnlwgt            178356.0
    Education-Num         10.0
    Capital Gain           0.0
    Capital Loss           0.0
    Hours per week        40.0
    dtype: float64

```python
df.sem()
```

    Age                 0.075593
    fnlwgt            584.937250
    Education-Num       0.014258
    Capital Gain       40.927838
    Capital Loss        2.233126
    Hours per week      0.068427
    dtype: float64

```python
df.var()
```

    Age               1.860614e+02
    fnlwgt            1.114080e+10
    Education-Num     6.618890e+00
    Capital Gain      5.454254e+07
    Capital Loss      1.623769e+05
    Hours per week    1.524590e+02
    dtype: float64

```python
df.std()
```

    Age                   13.640433
    fnlwgt            105549.977697
    Education-Num          2.572720
    Capital Gain        7385.292085
    Capital Loss         402.960219
    Hours per week        12.347429
    dtype: float64

```python
df.quantile(q=0.5)
```

    Age                   37.0
    fnlwgt            178356.0
    Education-Num         10.0
    Capital Gain           0.0
    Capital Loss           0.0
    Hours per week        40.0
    Name: 0.5, dtype: float64

```python
df.quantile(q=[0.05, 0.95])
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>fnlwgt</th>
      <th>Education-Num</th>
      <th>Capital Gain</th>
      <th>Capital Loss</th>
      <th>Hours per week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.05</th>
      <td>19.0</td>
      <td>39460.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>63.0</td>
      <td>379682.0</td>
      <td>14.0</td>
      <td>5013.0</td>
      <td>0.0</td>
      <td>60.0</td>
    </tr>
  </tbody>
</table>
</div>

In the next sample we replace a value with _None_ so that we can show how to hanlde missing values in a dataframe.

## Basic visualization

First let's create a pair plot

```python
_ = sns.pairplot(df, hue="Target")
```

![png](statistical_analysis_files/statistical_analysis_21_0.png)

```python
_ = sns.displot(df, x="Age" ,hue="Sex", label="male", kind="kde", log_scale=False)
```

![png](statistical_analysis_files/statistical_analysis_22_0.png)

## Inferential statistics

```python
female = df[df.Sex == 'Female']
male = df[df.Sex == 'Male']
```

T-Test

```python
t, p = stats.ttest_ind(female['Age'], male['Age'])
print("test statistic: {}".format(t))
print("p-value: {}".format(p))
```

    test statistic: -16.092517011911756
    p-value: 4.8239930687799265e-58

Wilcoxon rank-sum test

```python
z, p = stats.ranksums(female['Age'], male['Age'])
print("test statistic: {}".format(z))
print("p-value: {}".format(p))
```

    test statistic: -18.107256874221704
    p-value: 2.79324734147619e-73
