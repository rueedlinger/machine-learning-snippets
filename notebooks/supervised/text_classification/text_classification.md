>**Note**: This is a generated markdown export from the Jupyter notebook file [text_classification.ipynb](text_classification.ipynb).

# Text Classification (scikit-learn) with Naive Bayes

In this __Machine Learning Snippet__ we use scikit-learn (http://scikit-learn.org/) and ebooks from Project Gutenberg (https://www.gutenberg.org/) to create a text classifier, which can classify German, French, Dutch and English documents.

We need one document per language and split the document into smaller chuncks to train the classifier.

For our snippet we use the following ebooks:
- _'A Christmas Carol'_ by Charles Dickens (English), https://www.gutenberg.org/ebooks/46
- _'Der Weihnachtsabend'_ by Charles Dickens (German), https://www.gutenberg.org/ebooks/22465
- _'Cantique de Noël'_ by Charles Dickens (French), https://www.gutenberg.org/ebooks/16021
- _'Een Kerstlied in Proza'_ by Charles Dickens (Dutch), https://www.gutenberg.org/ebooks/28560


__Note:__
The ebooks are for the use of anyone anywhere at no cost and with
almost no restrictions whatsoever.  You may copy it, give it away or
re-use it under the terms of the Project Gutenberg License included
with this eBook or online at www.gutenberg.org




### Gathering data
First let's extract the text without the header and footer from the ebooks and split the text by whitespace in tokens.


```python
import re
import urllib.request

'''
with urllib.request.urlopen('http://www.gutenberg.org/cache/epub/22465/pg22465.txt') as response:
   txt_german = response.read().decode('utf-8') 

with urllib.request.urlopen('https://www.gutenberg.org/files/46/46-0.txt') as response:
   txt_english = response.read().decode('utf-8') 

with urllib.request.urlopen('http://www.gutenberg.org/cache/epub/16021/pg16021.txt') as response:
   txt_french = response.read().decode('utf-8') 

with urllib.request.urlopen('http://www.gutenberg.org/cache/epub/28560/pg28560.txt') as response:
   txt_dutch = response.read().decode('utf-8') 
'''

with open('data/pg22465.txt', 'r') as reader:
    txt_german = reader.read()

with open('data/46-0.txt', 'r') as reader:
    txt_english = reader.read()

with open('data/pg16021.txt', 'r') as reader:
    txt_french = reader.read()

with open('data/pg28560.txt', 'r') as reader:
    txt_dutch = reader.read()


def get_markers(txt, begin_pattern, end_pattern):
    iter = re.finditer(begin_pattern, txt)
    index_headers = [m.start(0) for m in iter]
    
    iter = re.finditer(end_pattern, txt)
    index_footers = [m.start(0) for m in iter]    
    
    # return first match
    return index_headers[0] + len(begin_pattern.replace('\','')), index_footers[0]

def extract_text_tokens(txt, 
                        begin_pattern='\*\*\* START OF THIS PROJECT GUTENBERG EBOOK', 
                        end_pattern='\*\*\* END OF THIS PROJECT GUTENBERG EBOOK'):
    header, footer = get_markers(txt, begin_pattern, end_pattern)
    return txt[header: footer].split()


tokens_german = extract_text_tokens(txt_german)
tokens_english = extract_text_tokens(txt_english)
tokens_french = extract_text_tokens(txt_french)
tokens_dutch = extract_text_tokens(txt_dutch)

print('tokens (german)', len(tokens_german))
print('tokens (english)', len(tokens_english))
print('tokens (french)', len(tokens_french))
print('tokens (dutch)', len(tokens_dutch))
```

    tokens (german) 27218
    tokens (english) 28562
    tokens (french) 32758
    tokens (dutch) 31506


## Data Preparation
Next we do some data cleaning. This means we remove special characters and numbers.


```python
import re

def remove_special_chars(x):
    
    # remove special characters
    chars = ['_', '(', ')', '*', '"', '[', ']', '?', '!', ',', '.', '»', '«', ':', ';']
    for c in chars:
        x = x.replace(c, '')
    
    # remove numbers
    x = re.sub('\d', '', x)
    
    return x

def clean_data(featurs): 
    # strip, remove sepcial characters and numbers
    tokens = [remove_special_chars(x.strip()) for x in featurs]
    
    cleaned = []
    
    # only use words with length > 1
    for t in tokens:
        if len(t) > 1:
            cleaned.append(t)
            
    return cleaned

cleaned_tokens_english = clean_data(tokens_english)
cleaned_tokens_german = clean_data(tokens_german)
cleaned_tokens_french = clean_data(tokens_french)
cleaned_tokens_dutch = clean_data(tokens_dutch)


print('cleaned tokens (german)', len(cleaned_tokens_german))
print('cleaned tokens (french)', len(cleaned_tokens_french))
print('cleaned tokens (dutch)', len(cleaned_tokens_dutch))
print('cleaned tokens (english)', len(cleaned_tokens_english))

```

    cleaned tokens (german) 27181
    cleaned tokens (french) 31995
    cleaned tokens (dutch) 31405
    cleaned tokens (english) 27527


Now we create for every language 1300 text samples with 20 tokens (words). These samples will later be 
used to train and test our model.


```python
from sklearn.utils import resample

max_tokens = 20
max_samples = 1300


def create_text_sample(x):
    
    data = []
    text = []
    for i, f in enumerate(x):
        text.append(f)
        if i % max_tokens == 0 and i != 0:
            data.append(' '.join(text))
            text = []
    return data
    

sample_german = resample(create_text_sample(cleaned_tokens_german), replace=False, n_samples=max_samples)
sample_french = resample(create_text_sample(cleaned_tokens_french), replace=False, n_samples=max_samples)
sample_dutch = resample(create_text_sample(cleaned_tokens_dutch), replace=False, n_samples=max_samples)
sample_english = resample(create_text_sample(cleaned_tokens_english), replace=False, n_samples=max_samples)

print('samples (german)', len(sample_german))
print('samples (french)', len(sample_french))
print('samples (dutch)', len(sample_dutch))
print('samples (english)', len(sample_english))

```

    samples (german) 1300
    samples (french) 1300
    samples (dutch) 1300
    samples (english) 1300


A text sample looks like this.


```python
print('English sample:
------------------')
print(sample_english[0])
print('------------------')

```

    English sample:
    ------------------
    dress trimmed with summer flowers But the strangest thing about it was that from the crown of its head there
    ------------------


### Choosing a model
As classifier we use the MultinomialNB classifier with the TfidfVectorizer. 

First we create the data structure which we will use to train the model. 

```
{
    samples: {
        text:[], 
        target: []
    }
    labels: [] 
}
 
```



```python
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def create_data_structure(**kwargs):
    data = dotdict({'labels':[]})
    data.samples = dotdict({'text': [], 'target': []})
    
    label = 0
    for name, value in kwargs.items():
        data.labels.append(name)
        for i in value:
            data.samples.text.append(i)
            data.samples.target.append(label)
        label += 1
            
    
    return data

data = create_data_structure(de = sample_german, en = sample_english, 
                             fr = sample_french, nl = sample_dutch)

print('labels: ', data.labels)
print('target (labels encoded): ', set(data.samples.target))
print('samples: ', len(data.samples.text))
```

    labels:  ['de', 'en', 'fr', 'nl']
    target (labels encoded):  {0, 1, 2, 3}
    samples:  5200


## Training
It's importan that we shuffle and split the data into training (70%) and test set (30%)


```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data.samples.text, data.samples.target, test_size=0.40)

print('train size (x, y): ', len(x_train),  len(y_train))
print('test size (x, y): ', len(x_test), len(y_test))

```

    train size (x, y):  3120 3120
    test size (x, y):  2080 2080


We connect all our parts (classifier, etc.) to our _Machine Learning Pipeline_. So it’s easier and faster to go trough all processing steps to build a model.

The TfidfVectorizer will use the the word analyzer, min document frequency of 10  and convert the text to lowercase. I know we already did a lowercase conversion in the previous step. We also provide some stop words which should be ignored in our model. 

The MultinomialNB classifier wil use the default alpha value 1.0.

Here you can play around with the settings. In the next section you see how to evaluate your model.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

stopwords = ['scrooge', 'scrooges', 'bob']

pipeline = Pipeline([('vect', TfidfVectorizer(analyzer='word', 
                            min_df=10, lowercase=True, stop_words=stopwords)),
                      ('clf', MultinomialNB(alpha=1.0))])

```

### Evaluation

In this step we will evaluate the performance of our classifier. So we do the following evaluation:
- Evaluate the model with k-fold on the training set
- Evaluate the final model with the test set

First let's evaluate our model with a k-fold cross validation. 


```python
from sklearn.model_selection import KFold
from sklearn import model_selection

folds = 5

for scoring in ['f1_weighted', 'accuracy']:

    scores = model_selection.cross_val_score(pipeline, X=x_train, y=y_train, 
                                         cv=folds, scoring=scoring)
    print(scoring)
    print('scores: %s' % scores )
    print(scoring + ': %0.6f (+/- %0.4f)' % (scores.mean(), scores.std() * 2))
    print()

```

    f1_weighted
    scores: [1.         1.         1.         0.99839742 1.        ]
    f1_weighted: 0.999679 (+/- 0.0013)
    
    accuracy
    scores: [1.         1.         1.         0.99839744 1.        ]
    accuracy: 0.999679 (+/- 0.0013)
    


Next we build the model and evaluate the result against our test set.


```python
from sklearn import metrics

text_clf = pipeline.fit(x_train, y_train)

predicted = text_clf.predict(x_test)

print(metrics.classification_report(y_test, predicted, digits=4))
```

                  precision    recall  f1-score   support
    
               0     1.0000    1.0000    1.0000       526
               1     0.9981    1.0000    0.9990       525
               2     1.0000    1.0000    1.0000       504
               3     1.0000    0.9981    0.9990       525
    
        accuracy                         0.9995      2080
       macro avg     0.9995    0.9995    0.9995      2080
    weighted avg     0.9995    0.9995    0.9995      2080
    


## Examine the features of the model

Let's see what are the most informative features


```python
import numpy as np

# show most informative features
def show_top10(classifier, vectorizer, categories):

    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))


show_top10(text_clf.named_steps['clf'], text_clf.named_steps['vect'], data.labels)
```

    de: ein ich das sie zu es er die der und
    en: was in that his he it of to and the
    fr: qu une que les un il la et le de
    nl: op dat te zijn hij van een de het en


    /Users/mru/.local/share/virtualenvs/machine-learning-snippets-mLikUPnf/lib/python3.8/site-packages/sklearn/utils/deprecation.py:101: FutureWarning: Attribute coef_ was deprecated in version 0.24 and will be removed in 1.1 (renaming of 0.26).
      warnings.warn(msg, category=FutureWarning)


Let's see which and how many features our model has.


```python
feature_names = np.asarray(text_clf.named_steps['vect'].get_feature_names())

print('number of features: %d' % len(feature_names))
print('first features: %s'% feature_names[0:10])
print('last features: %s' % feature_names[-10:])
```

    number of features: 809
    first features: ['aan' 'aber' 'about' 'after' 'again' 'ah' 'ai' 'air' 'al' 'all']
    last features: ['zu' 'zum' 'zurück' 'zwei' 'écria' 'étaient' 'était' 'été' 'être' 'über']


### New data
Let's try out the classifier with the new data.


```python
new_data = ['Hallo mein Name ist Hugo.', 
            'Hi my name is Hugo.', 
            'Bonjour mon nom est Hugo.',
            'Hallo mijn naam is Hugo.',
            'Eins, zwei und drei.',
            'One, two and three.',
            'Un, deux et trois.',
            'Een, twee en drie.'
           ]

predicted = text_clf.predict(new_data)
probs = text_clf.predict_proba(new_data)
for i, p in enumerate(predicted):
    print(new_data[i], ' --> ', data.labels[p], ', prob:' , probs[i][p])
    
```

    Hallo mein Name ist Hugo.  -->  de , prob: 0.8425895444632882
    Hi my name is Hugo.  -->  en , prob: 0.8303842632344532
    Bonjour mon nom est Hugo.  -->  fr , prob: 0.9516803698057394
    Hallo mijn naam is Hugo.  -->  nl , prob: 0.743320138749105
    Eins, zwei und drei.  -->  de , prob: 0.9065403234994696
    One, two and three.  -->  en , prob: 0.9785308510746763
    Un, deux et trois.  -->  fr , prob: 0.9858276336351973
    Een, twee en drie.  -->  nl , prob: 0.9643584844428632
