'''

editor: Jones
date: 2019/07/29
content: 
1. txt to csv

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 門檻值
def min_thresholding(threshold_array):

	min_threshold = 50
	# print(min_threshold)
	threshold_array[threshold_array <= min_threshold] = 0

	return threshold_array


# 主程序
def main():
	# open file 
	f = open('data_txt/子暘1210_lying.txt', 'r',encoding = 'utf-8')

	raw_data_list = []
	for x in f:
		raw_data_list.append(x)
	print(len(raw_data_list))

	countA = 0
	countB = 0
	countC = 0
	countD = 0

	# 正躺
	face_up_list = []
	# 右侧躺
	face_right_list = []
	# 左侧躺
	face_left_list = []
	# 俯卧
	face_down_list = []


	for item in raw_data_list[1:]:
		if item[-2] == '0':
			countA += 1
			face_up_list.append(item)
		elif item[-2] == '1':
			countB += 1
			face_right_list.append(item)
		elif item[-2] == '2':
			countC += 1
			face_left_list.append(item)
		elif item[-2] == '3':
			countD += 1
			face_down_list.append(item)

	print(countA)
	print(countB)
	print(countC)
	print(countD)

	original_array = min_thresholding(np.array(face_up_list[140].split(),dtype = int)[:220].reshape(20,11))

	# # 向左平移1格
	# left_roll_1 = np.roll(original_array, -1, axis = 1)
	# # print(left_roll_1)

	# # # 向右平移1格
	# right_roll_1 = np.roll(original_array, 1, axis = 1)
	# # print(right_roll_1)

	# # # 向右平移2格
	# right_roll_2 = np.roll(original_array, 2, axis = 1)
	# # print(right_roll_2)

	# # # 向右平移3格
	# right_roll_3 = np.roll(original_array, 3, axis = 1)
	# # print(right_roll_3)

	# # 100 frame
	# face_up_list = face_up_list[40:140]

	# face_right_list = face_right_list[137:237]

	# face_left_list = face_left_list[100:200]

	# face_down_list = face_down_list[120:220]

	# 25 frame
	face_up_list = face_up_list[115:140]

	face_right_list = face_right_list[137:162]

	face_left_list = face_left_list[100:125]

	face_down_list = face_down_list[120:145]

	# face_up_left_roll_1 = []
	# face_up_right_roll_1 = []
	# face_up_right_roll_2 = []
	# face_up_right_roll_3 = []

	# for i in face_up_list:
	# 	j = np.array(i.split(),dtype=int)[:220]
	# 	j = min_thresholding(j.reshape(20,11))


	# 	# 向左平移1格
	# 	left_roll_1 = np.roll(j, -1, axis = 1)
	# 	# 2D to 1D
	# 	left_roll_1D = left_roll_1.ravel()
	# 	face_up_left_roll_1.append(left_roll_1D)

	# 	# 向右平移1格
	# 	right_roll_1 = np.roll(j, 1, axis = 1)
	# 	face_up_right_roll_1.append(right_roll_1.ravel())

	# 	# 向右平移2格
	# 	right_roll_2 = np.roll(j, 2, axis = 1)
	# 	face_up_right_roll_2.append(right_roll_2.ravel())

	# 	# 向右平移3格
	# 	right_roll_3 = np.roll(j, 3, axis = 1)
	# 	face_up_right_roll_3.append(right_roll_3.ravel())

	# face_up_left_roll_1.extend(face_up_right_roll_2)
	# face_up_left_roll_1.extend(face_up_right_roll_2)
	# face_up_left_roll_1.extend(face_up_right_roll_3)

	# print('len =', len(face_up_left_roll_1))

	# print(np.array(face_up_left_roll_1[-2]).reshape(20,11))


	all_list  = face_up_list + face_right_list + face_left_list + face_down_list

	# print(all_list)

	all_array = []

	for i in all_list:
		j = min_thresholding(np.array(i.split(),dtype = int))[:220]
		all_array.append(j)

	new_array = np.array(all_array)

	print(new_array[-1])

	columns = []
	i = 0
	while i < 220:
		col = 'pixel%d' %i
		i = i + 1
		columns.append(col)

	df = pd.DataFrame(new_array, columns = columns)

	# label 0, 坐姿
	# label 1, 睡姿

	label_1 = np.ones(100, dtype=int)

	df.insert(220, "label", label_1, True)


	print(df.head())

	# df.rename(columns={'pixel220':'label'}, inplace=True)

	print(df.head())

	df.to_csv ("1210ziyang_SVM_lying.csv" , encoding = "utf-8")


	# 顯示圖形
	# fig, ax = plt.subplots()

	# plt.title('Sleeping Positions')
	# plt.imshow(right_roll_3[::-1], cmap=plt.cm.jet)
	# plt.ylim((-0.5, 19.5))
	# plt.yticks(np.arange(0,19,1.0))
	# plt.colorbar()
	# plt.show()



# 程式起點
if __name__ == '__main__':
	main()



# # list7
# list7 = list1 + list2 + list3 + list4 + list5 + list6
# new_array = []
# for i in list7:
# 	j = np.array(i.split(),dtype = int)[:220]
# 	new_array.append(j)

# new_array = np.array(new_array)
# print(new_array[0])
# columns = []
# i = 0
# while i < 220:
# 	col = 'pixel%d' %i
# 	i = i + 1
# 	columns.append(col)

# # columns = ['pixel0',]

# df = pd.DataFrame(new_array, columns = columns)

# # label 0, 盘腿坐
# # label 1, 腿伸直坐
# # label 2, 正躺
# # label 3, 右侧躺
# # label 4, 左侧躺
# # label 5, 俯卧（趴睡）

# label_0 = np.zeros(300, dtype=int)
# label_1 = np.ones(300, dtype=int)
# label_2 = np.ones(300, dtype=int) * 2
# label_3 = np.ones(300, dtype=int) * 3
# label_4 = np.ones(300, dtype=int) * 4
# label_5 = np.ones(300, dtype=int) * 5

# label_list = np.concatenate((label_0, label_1, label_2, label_3, label_4, label_5),axis=0)
# print(label_list)

# df.insert(0, "label", label_list, True)
# print(df)
# df.rename(columns={'0':'pixel0'}, inplace=True)

# df.to_csv ("0731Jones_TrainingData.csv" , encoding = "utf-8")