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
    train = pd.get_dummies(train, columns=[column])
    
#plt.figure()
#sns.heatmap(train.corr(),annot=True)
#plt.title('Correlation')

# this is our test set

y = train['SalePrice']
X = train.drop(['SalePrice'], 1)


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


##########################################
#Feature importances with forests of trees
##########################################

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest.fit(X_train, y_train)

predicted_ETC = forest.predict(X_test)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center", tick_label=X_train.columns)
plt.xticks(range(X_train.shape[1]), X_train.columns)
plt.xlim([-1, X_train.shape[1]])
plt.show()



#plot the confusion matrix

class_names = list(train)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predicted_ETC)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
