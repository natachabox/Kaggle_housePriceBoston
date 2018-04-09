# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:45:14 2018
# =============================================================================
# Natacha Cahorel
# Formation developpeur DATA IA Microsoft
# =============================================================================

@author: natacha
"""


print(__doc__)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


train = pd.read_csv('train.csv')

#retraitement des données :

train['LotFrontage'] = train['LotFrontage'].fillna(-1)

# NA mean No alley Access acording to data_description
train['Alley'] = train['Alley'].fillna('NoAlleyAccess')


train['MasVnrType'] = train['MasVnrType'].fillna('NoInfo')

train['MasVnrArea'] = train['MasVnrArea'].fillna(-1)

# NA mean No Basement
train['BsmtQual'] = train['BsmtQual'].fillna('NB')
train['BsmtCond'] = train['BsmtCond'].fillna('NB')
train['BsmtExposure'] = train['BsmtExposure'].fillna('NB')
train['BsmtFinType1'] = train['BsmtFinType1'].fillna('NB')
train['BsmtFinType2'] = train['BsmtFinType2'].fillna('NB')

# 1 missing value
train['Electrical'] = train['Electrical'].fillna('NoInfo')


# NA mean No Fireplace
train['FireplaceQu'] = train['FireplaceQu'].fillna('NF')

# NA mean No Garage
train['GarageType'] = train['GarageType'].fillna('NG')
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(-1)
train['GarageFinish'] = train['GarageFinish'].fillna('NG')
train['GarageQual'] = train['GarageQual'].fillna('NG')
train['GarageCond'] = train['GarageCond'].fillna('NG')

# NA mean No pool, no fence, No misc feature
train['PoolQC'] = train['PoolQC'].fillna('NP')
train['Fence'] = train['Fence'].fillna('NF')
train['MiscFeature'] = train['MiscFeature'].fillna('NMF')


columns_to_encode = list(train.loc[:, train.dtypes == object])

for column in columns_to_encode:
    le = preprocessing.LabelEncoder()
    le.fit(train[column])
    train[column] = le.transform(train[column])
    #train = pd.get_dummies(train, columns=[column])

plt.figure()
sns.heatmap(train.corr(),annot=False)
plt.title('Correlation')

    
y = train['SalePrice']
X = train.drop(['SalePrice', 'Id'], 1)


# flatten y into a 1-D array
y = np.ravel(y)

X_embedded = TSNE(n_components=2, perplexity=50).fit_transform(train)
n_samples = train.shape[0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_embedded[:,0], X_embedded[:,1], np.zeros(n_samples), c=y)

plt.show()


plt.figure()
plt.scatter(X_embedded[:,0], X_embedded[:,1], s=50, c=y, alpha=0.5)
plt.show()

    
    
    
#plt.figure()
#sns.heatmap(train.corr(),annot=True)
#plt.title('Correlation')

# this is our test set

y = train['SalePrice']
X = train.drop(['SalePrice', 'Id'], 1)


# flatten y into a 1-D array
y = np.ravel(y)

# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


############################################
#Linear Regression
############################################

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print(regr.score(X_train, y_train))


predicted_regr = regr.predict(X_test)
print(regr.score(X_test, y_test))

# The coefficients
#print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, predicted_regr))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, predicted_regr))





