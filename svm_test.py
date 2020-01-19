'''
editor: Jones
date: 20190920
content: 
1. 找特徵
2. 4個特徵：
3. X軸，Y軸的變異數，
4. 平均壓力值，
5. 找重心，加半徑畫圓，圓内點數/總點數
6. 將特徵放到SVM去訓練，用來區分坐姿和睡姿
'''

from sklearn import datasets
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import LeaveOneOut


data = pd.read_csv('input/1217_bingyu_feature_test_data.csv')

X = data[['average value', 'vertical variance', 'horizonal variance', 'ratio as 2']]
y = data[['target']]

print(X[:10])
sc = StandardScaler()
sc.fit(X)
X_test_std = sc.transform(X)

print(X_test_std[:10])

# print(X_test_std[:10])

# # read model
clf= joblib.load('model/1216svm.pki')

# print(clf.predict(X_test_std))


# 預測
predict_result = clf.predict(X_test_std)
really_result = y['target'].values

print('predict_result =', predict_result)
print('really_result =', really_result)

# 錯誤統計
error = 0
for i, v in enumerate(predict_result):
	if v != really_result[i]:
		print(i)
		error += 1
print('error =',error)






