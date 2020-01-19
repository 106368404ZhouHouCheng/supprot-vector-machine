'''

editor: Jones
date: 2019/07/29
content: 用SVM去區分坐姿與睡姿

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
from sklearn.svm import SVC



# # 四個特征值：Point和 Average Pressure Value
data = pd.read_csv('input/1216_feature_train_data.csv')  
print(data.head)

X = data[['average value', 'vertical variance', 'horizonal variance', 'ratio as 2']]
y = data[['target']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


print('X_train =',len(X_train))
print('y_train =',len(y_train))

print(X_train[:10])
# Normatlization
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)



print('X_train_std[:10] =',X_train_std[:10])
sc.fit(X_test)
X_test_std = sc.transform(X_test)


svm = SVC()
clf = svm.fit(X_train_std, y_train)

# 預測
predict_result = svm.predict(X_test_std)
really_result = y_test['target'].values

print(predict_result) 
print(really_result)
print('type(predict_result) =', type(predict_result))

print('clf:', clf.score(X_test_std, y_test))

scores = cross_val_score(clf, X, y, cv=100, scoring='accuracy')
print('scores:', scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# save model
joblib.dump(clf, 'model/1216svm.pki')

# 錯誤統計
error = 0
for i, v in enumerate(predict_result):
	if v != really_result[i]:
		error += 1
print('error =',error)

# 顯示
# from mlxtend.plotting import plot_decision_regions

# plot_decision_regions(X_train_std, y_train['target'].values, clf= svm)
# plt.title('2 Sitting Positions and 4 Sleeping Positions')
# plt.xlabel('Point [Standardized]')
# plt.ylabel('Average Pressure Value [Standardized]')
# plt.legend(loc = 'upper left')
# plt.tight_layout()
# plt.show()


# # read model
# clf3 = joblib.load('save/clf.pki')

# print(clf3.predict(X[0:5]))


# open file 
# f = open('0709jones.txt', 'r',encoding = 'utf-8')

# raw_data_list = []
# for x in f:
# 	raw_data_list.append(x)
# raw_data_list = raw_data_list[1:]
# print(len(raw_data_list))
# row = 20
# col = 11
# original_list = [] 
# original_array = np.array(raw_data_list[50].split(),dtype = int)[:220]
# original_array = original_array.reshape(row,col)


# # 正躺
# list1 = raw_data_list[10:70]
# list2 = raw_data_list[110:170]

# array1 = []
# for i in list1:
# 	array0 = np.array(i.split(),dtype = int)[:220]	
# 	array1.append(array0)

# # pd_array1 = pd.DataFrame(array1)
# # print(pd_array1)

# array2 = []
# for i in list2:
# 	array0 = np.array(i.split(),dtype = int)[:220]	
# 	array2.append(array0)

# print(len(array1))
# print(len(array2))

# new_array = array1 + array2

# df = pd.DataFrame(np.array(new_array))
# df.to_csv ("output.csv" , encoding = "utf-8")



# # average_array = np.mean(new_array,axis=0,dtype = int)
# # print(average_array)
# # average_array1 = average_array.reshape(row,col)
# # average_array2 = average_array.reshape(row,col)
# # print(average_array1)

# # 顯示圖形
# fig, ax = plt.subplots()
# plt.title('Positions')
# plt.imshow(original_array[::-1], cmap = plt.cm.jet)

# plt.ylim((-0.5, 19.5))
# plt.yticks(np.arange(-0.5,19.5,1.0))
# plt.xlim((-0.5, 10.5))
# plt.xticks(np.arange(-0.5,10.5,1.0))
# plt.show()