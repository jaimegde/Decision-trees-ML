#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 12:33:09 2021

@author: jaimegde
"""

from sklearn.ensemble import RandomForestClassifier
from plotnine import *
from plotnine.data import diamonds
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 

diamonds.head()
X = diamonds.copy()

le1 = preprocessing.LabelEncoder()
le1.fit(diamonds.clarity)
X.clarity = le1.transform(X.clarity)

le2 = preprocessing.LabelEncoder()
le2.fit(diamonds.color)
X.color = le2.transform(X.color)
X.head()


Y = X.loc[:,["cut"]]
X = X.loc[:, X.columns != "cut"]
X.shape
Y.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)

X_train.shape 
X_test.shape 
Y_train.shape 
Y_test.shape 

dt_model = RandomForestClassifier(max_depth = 4, n_estimators = 50, class_weight="balanced")
dt_model.fit(X_train, Y_train)
dt_model.feature_importances_ 

imp = pd.DataFrame()
imp["Variable"] = X_train.columns
imp["imp"] = dt_model.feature_importances_
ggplot(aes(x="Variable", y="imp", fill="Variable"), imp) + geom_bar(stat="identity")

prediction = dt_model.predict(X_test)
prob = dt_model.predict_proba(X_test)
dt_model.score(X_test, Y_test)

dt_model.classes_


prob = pd.DataFrame(prob)
prob.columns = dt_model.classes_
prob.index = X_test.index

probabilities = dt_model.predict_proba(X_test)
probabilities = pd.DataFrame(probabilities)
probabilities.head()
dt_model.classes_
probabilities.columns = dt_model.classes_
probabilities.head()

Y_test = pd.DataFrame(Y_test)
Y_test.head()
probabilities.head()
probabilities.reset_index(drop=True, inplace=True)
Y_test.reset_index(drop=True, inplace=True)

probabilities = pd.concat([probabilities, Y_test], axis=1)
probabilities.head()

probabilities = probabilities.sort_values(by="Premium", ascending=False)
probabilities.head()

limit = int(0.05*probabilities.shape[0])
top4 = probabilities.head(431)
top4.cut.value_counts()
pd.crosstab(top4.cut.columns="Number of cases", normalize=True)


probabilities = dt_model.predict_proba(X_test)
probabilities = pd.DataFrame(probabilities)
probabilities.columns = "Prob_" + dt_model.classes_
probabilities.head(3)
probabilities = pd.concat([probabilities, Y_test], axis=1)

Y_test = pd.DataFrame(Y_test)
probabilities = probabilities.sort_values(by="Prob_Premium", ascending=False)
probabilities["q"] = pd.cut(range(probabilities.shape[0]), 100, labels = range(100))
probabilities.q.value_counts()
probabilities.head(20)
ggplot(aes(x="q", fill="cut"), probabilities) + geom_bar(position="fill")

ggplot(aes(x="q", fill="cut"), probabilities.loc[probabilities.cut == "Premium",:]) + geom_bar()










