'''

editor: Jones
date: 20200116
content: 
1. 找特徵
2. 4個特徵：
3. X軸，Y軸的變異數，
4. 平均壓力值，
5. 找重心，加半徑畫圓，圓内點數/總點數
6. 將特徵放到SVM去訓練，用來區分坐姿和睡姿

新增： 將壓力影像轉直方圖，找直方圖的變異數，將其作為一個特徵值

'''

import numpy as np
import matplotlib.pyplot as plt
# import cv2
import pandas as pd 
import math

# 門檻值
def thresholding(raw_data_array):

	threshold = 40
	raw_data_array[raw_data_array <= threshold] = 0
	return raw_data_array


# 二值化
def binarization(binarized_array):

	binarized = 0
	binarized_array[binarized_array <= binarized] = 0
	binarized_array[binarized_array > binarized] = 1023

	return binarized_array


# 非零的壓力值陣列
def nonzero_pressure_value(threshold_array):

	nonzero_array = threshold_array[np.nonzero(threshold_array > 0)]
	return nonzero_array


# 計算非零壓力值的加權平均值
def nonzero_average_pressure_value(nonzero_array):

	j = 0
	average_pres_val = 0
	size = len(nonzero_array)
	while j < size:
		average_pres_val = average_pres_val + (1/size * nonzero_array[j])
		j = j + 1
	return average_pres_val


# 計算非零壓力值的變異數
def nonzero_variance_pressure_value(nonzero_array, average_pres_val):

	variance_1D = 0
	size = len(nonzero_array)
	for k in nonzero_array:
		variance_1D = variance_1D + (1/size * (k - average_pres_val) ** 2)
	return variance_1D


# 水平軸的變異數
def variance_x_axis(my_array):

	# 垂直軸總和
	x_axis_array = np.sum(my_array, axis = 0)
	print(x_axis_array)

	nonzero_x_axis_array = nonzero_pressure_value(x_axis_array)
	average_x_axis_pres_val = nonzero_average_pressure_value(nonzero_x_axis_array)
	variance_x_axis = nonzero_variance_pressure_value(nonzero_x_axis_array, average_x_axis_pres_val)
	print(nonzero_x_axis_array)
	print(average_x_axis_pres_val)
	print(variance_x_axis)

	return variance_x_axis


# 垂直軸的變異數
def variance_y_axis(my_array):

	# 垂直軸總和
	y_axis_array = np.sum(my_array, axis = 1)
	# print(y_axis_array)

	nonzero_y_axis_array = nonzero_pressure_value(y_axis_array)
	average_y_axis_pres_val = nonzero_average_pressure_value(nonzero_y_axis_array)
	variance_y_axis = nonzero_variance_pressure_value(nonzero_y_axis_array, average_y_axis_pres_val)
	# print(nonzero_y_axis_array)
	# print(average_y_axis_pres_val)
	# print(variance_y_axis)

	return variance_y_axis


# 物體的重心for2維影像
def center_of_gravity(my_array):
	
	sum_i = 0
	sum_j = 0

	for y, i in enumerate(my_array):
		for x, j in enumerate(i):
			b = sum_i
			a = sum_j

			sum_i = y * j
			sum_i = sum_i + b

			sum_j = x * j
			sum_j = sum_j + a

	yg = sum_i/np.sum(my_array)
	xg = sum_j/np.sum(my_array)

	return xg, yg


# 以重心為圓心，半徑為1.0，以此遞增，所圍成的圓，計算圓中壓中的壓力值點數與總的壓力值點數的比例
def ratio_of_points(my_array):

	xg, yg =  center_of_gravity(my_array)
	# point_all = pressure_points(my_array)
	my_array = thresholding(my_array)

	distance_list = []
	row = 20
	col = 11
	radius_1 = 1.0

	for row_index, row_element in enumerate(my_array):
		for col_index, col_element in enumerate(row_element):
			if(col_element == 1):
				distance = np.sqrt((xg - col_index) ** 2 + (yg - row_index) ** 2)
				distance_list.append(distance)


	distance_array = np.sort(np.array(distance_list))
	# print('distance_array =', distance_array)
	ratio_max = np.max(distance_array)
	# print(ratio_max)
	ratio_list = []

	count = 1

	while count <= math.ceil(ratio_max):
		ratio = np.size(distance_array[distance_array < radius_1 * count])/np.size(distance_array)
		ratio_list.append(round(ratio, 3))
		count += 1
	# ratio_array = np.array(ratio_list)

	return ratio_list

def ratio_index(ratio_list):

	ratio0_list = []
	ratio1_list = []
	ratio2_list = []
	ratio3_list = []

	# print(ratio_list[500:600])

	ratio = 0
	count = 0
	while count < len(ratio_list):
		ratio0_list.append(ratio_list[count][0])
		ratio1_list.append(ratio_list[count][1])
		ratio2_list.append(ratio_list[count][2])
		# ratio3_list.append(ratio_list[count][3])
		count += 1 

	# print(np.mean(np.array(ratio0_list)))
	# print('variance =', np.var(np.array(ratio0_list)))
	# print(np.mean(np.array(ratio1_list)))
	# print('variance =', np.var(np.array(ratio1_list)))
	# print(np.mean(np.array(ratio2_list)))
	# print('variance =', np.var(np.array(ratio2_list)))
	# print(np.mean(np.array(ratio3_list)))
	# print('variance =', np.var(np.array(ratio3_list)))

# 主程序
def main():
	# open file 
	raw_data = pd.read_csv('data_csv/1217秉渝_test_data.csv')
	raw_data = raw_data.to_numpy()

	row = 20
	col = 11
	original_list = [] 

	# original_array = np.array(raw_data[0])[:220].reshape(row, col)
	original_array = np.array(raw_data[0])[:220].reshape(row, col)
	print(original_array)

	# threshold_array = thresholding(original_array)
	# nonzero_array = nonzero_pressure_value(threshold_array)
	# average_pres_val = nonzero_average_pressure_value(nonzero_array)
	# variance_1D = nonzero_variance_pressure_value(nonzero_array, average_pres_val)

	# print('threshold_array =', threshold_array)
	# print('nonzero_array =', nonzero_array)
	# print('average_pres_val =', average_pres_val)
	# print('variance_1D =', variance_1D)
	# var_x_axis = variance_x_axis(threshold_array)
	# var_y_axis = variance_y_axis(threshold_array)

	# 非零壓力值的加權平均值list
	average_pres_val_list = []
	# 非零壓力值變異數list
	variance_1D_list = []
	# 水平軸變異數 list
	variance_x_axis_list = []
	# 垂直軸變異數 list
	variance_y_axis_list = []


	xg_list = []
	yg_list = []
	ratio_list = []

	for item in raw_data:
		item_array = np.array(item)[:220]
		item_array = item_array.reshape(row, col)
		threshold_item_array = thresholding(item_array)
		nonzero_item_array = nonzero_pressure_value(threshold_item_array)
		average_pres_val = nonzero_average_pressure_value(nonzero_item_array)
		variance_1D = nonzero_variance_pressure_value(nonzero_item_array, average_pres_val)
		var_x_axis = variance_x_axis(threshold_item_array)
		var_y_axis = variance_y_axis(threshold_item_array)

		# ratio = ratio_of_points(item_array)
		# print(ratio)
		# xg, yg = center_of_gravity(item_array)
		# xg = int(round(xg))
		# yg = int(round(yg))

		average_pres_val_list.append(average_pres_val)
		variance_1D_list.append(variance_1D)
		variance_x_axis_list.append(var_x_axis)
		variance_y_axis_list.append(var_y_axis)

	print('average_pres_val_list =', average_pres_val_list)
	print('variance_1D_list =', variance_1D_list)
	print('variance_x_axis_list =', variance_x_axis_list)
	# print('variance_y_axis_list =', variance_y_axis_list)
		# ratio_list.append(ratio)
		# xg_list.append(xg)
		# yg_list.append(yg)

	# ratio_index(ratio_list)

	# # 取半径为2
	# index_list = []
	# for index in ratio_list:
	# 	index_list.append(index[2])


	# print(ratio_list[0:100])

	# print(type(ratio_list[0][0]))

	# print('average_value_list =', average_value_list)
	# print('vertical_var_list =', vertical_var_list)
	# print('horizonal_var_list =', horizonal_var_list)
	# print('ratio_list =', ratio_list)
	# print('index_list =', index_list)


	# target_array = np.hstack((np.zeros(1800), np.ones(1800)))
	# target_array = np.hstack((np.zeros(100), np.ones(100)))

	# target_array = np.hstack(np.zeros(1800))

	# d = {'average value': np.array(average_value_list), 'horizonal variance': np.array(horizonal_var_list), 'vertical variance': np.array(vertical_var_list), 
	# 		'ratio as 2': np.array(index_list), 'target': target_array}

	# df = pd.DataFrame(data = d)
	# df = pd.DataFrame(data = d_ratio)
	# print(df)
	# df.to_csv ("input/1217_bingyu_feature_test_data.csv" , encoding = "utf-8")

	# r_1 = 1.0
	# r_2 = 2.0
	# r_3 = 3.0

	# # 方法一：参数方程
	# theta = np.arange(0, 2*np.pi, 0.01)
	# x1 = xg1 + r_1 * np.cos(theta)
	# y1 = yg1 + r_1 * np.sin(theta)

	# x2 = xg1 + r_2 * np.cos(theta)
	# y2 = yg1 + r_2 * np.sin(theta)


	# x3 = xg1 + r_3 * np.cos(theta)
	# y3 = yg1 + r_3 * np.sin(theta)

	# 顯示圖形
	# fig, ax = plt.subplots()
	# plt.title('lying Posture')
	# plt.imshow(original_array,cmap = plt.cm.jet)
	# plt.xticks(np.arange(0.0,11.0,1.0))
	# plt.yticks(np.arange(0.0,20.0,1.0))
	# plt.scatter([xg1,],[yg1,], marker = 'o', color = 'red', s = 50)
	# # plt.plot(x1, y1, color = 'red')
	# # plt.plot(x2, y2, color = 'red')
	# plt.plot(x3, y3, color = 'red')
	# plt.show()

	# fig, ax = plt.subplots()
	# plt.title("Simple Plot")
	# plt.xticks(np.arange(-1,19,1.0))
	# plt.yticks(np.arange(-0.1,1.1,0.05))
	# for i, element in enumerate(ratio_list):
	# 	plt.plot(range(len(element)), element,  marker = '.')
	# plt.legend()
	# plt.show()



# 程式起點
if __name__ == '__main__':
    main()  





