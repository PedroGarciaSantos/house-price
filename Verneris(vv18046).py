
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

from sklearn import datasets
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_regression
import warnings
import seaborn as sns

get_ipython().magic(u'matplotlib inline')
import scipy.stats as stats
from sklearn.model_selection import KFold
from IPython.display import HTML, display
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 100

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']

train.head()


# In[4]:


train.SalePrice.describe()


# In[5]:


print("Skew is:", train.SalePrice.skew())


# In[6]:


target = np.log(train.SalePrice)
print("Skew is:", target.skew())


# In[9]:


numeric_features = train.select_dtypes(include=[np.number])
corr = numeric_features.corr()

f, ax = plt.subplots(figsize=(17, 15))
sns.heatmap(corr, vmax=.8, square=True)


# In[10]:


corr["SalePrice"].sort_values(ascending=False)


# In[11]:


y = np.log(train.SalePrice)
for i in quantitative:
    var = i
    data = pd.concat([train['SalePrice'], train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[16]:


for c in qualitative:
    train[c] = train[c].astype('category')
    if train[c].isnull().any():
        train[c] = train[c].cat.add_categories(['MISSING'])
        train[c] = train[c].fillna('MISSING')

def encode(frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()
    
    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, "Enc"+feature] = o
    
qual_encoded = []
for q in qualitative:  
    encode(train, q)
    qual_encoded.append("Enc" + q)
print(qual_encoded)


# In[18]:


def pairplot(x, y, **kwargs):
    ax = plt.gca()
    ts = pd.DataFrame({'time': x, 'val': y})
    ts = ts.groupby('time').mean()
    ts.plot(ax=ax)
    plt.xticks(rotation=90)
    
f = pd.melt(train, id_vars=['SalePrice'], value_vars=qual_encoded)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=8)
g = g.map(pairplot, "value", "SalePrice")


# In[19]:


def pairplot(x, y, **kwargs):
    ax = plt.gca()
    ts = pd.DataFrame({'time': x, 'val': y})
    ts = ts.groupby('time').mean()
    ts.plot(ax=ax)
    plt.xticks(rotation=90)
    
f = pd.melt(train, id_vars=['SalePrice'], value_vars=quantitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=7)
g = g.map(pairplot, "value", "SalePrice")


# In[23]:


data = train.drop(["LotFrontage", "BsmtFinSF1", "BsmtUnfSF", "BsmtFinSF2", "YearRemodAdd", "YearBuilt", "PoolArea", "BedroomAbvGr", "EncLotShape", "EncLandSlope", "EncRoofMatl", "EncFoundation", "EncBsmtCond", "EncGarageType", "EncPoolQC", "EncSaleType"], axis=1)


# In[24]:


testQuantitative = [f for f in test.columns if test.dtypes[f] != 'object']
testQuantitative.remove('Id')
testQualitative = [f for f in test.columns if test.dtypes[f] == 'object']


# In[25]:


for c in testQualitative:
    test[c] = test[c].astype('category')
    if test[c].isnull().any():
        test[c] = test[c].cat.add_categories(['MISSING'])
        test[c] = test[c].fillna('MISSING')

def encode(frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    #ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    #ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()
    
    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, "Enc"+feature] = o
    
testQual_encoded = []
for q in testQualitative:  
    encode(test, q)
    testQual_encoded.append("Enc" + q)
testQual_encoded

