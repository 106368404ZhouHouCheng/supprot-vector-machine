'''

editor: Jones
date: 20200116
content: 
1. 找特徵
2. 4個特徵：
3. Y軸的變異數，
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


# 水平軸的變異數
def variance_x_axis(my_array):

	# 垂直軸總和
	x_axis_array = np.sum(my_array, axis = 0)
	# print(x_axis_array)
	weight_x_axis_sum = 0
	for x, x_axis_val in enumerate(x_axis_array):
		weight_x_axis_sum = weight_x_axis_sum + x * x_axis_val

	xg = weight_x_axis_sum/np.sum(x_axis_array)

	var_x_axis = 0
	for x, x_axis_val in enumerate(x_axis_array):
		var_x_axis = var_x_axis + (x_axis_val / np.sum(x_axis_array) * (x - xg) ** 2)
	return var_x_axis


# 垂直軸的變異數
def variance_y_axis(my_array):

	# 水平軸總和
	y_axis_array = np.sum(my_array, axis = 1)
	# print(y_axis_array)
	weight_y_axis_sum = 0
	for y, y_axis_val in enumerate(y_axis_array):
		weight_y_axis_sum = weight_y_axis_sum + y * y_axis_val

	yg = weight_y_axis_sum/np.sum(y_axis_array)

	var_y_axis = 0
	for y, y_axis_val in enumerate(y_axis_array):
		var_y_axis = var_y_axis + (y_axis_val / np.sum(y_axis_array) * (y - yg) ** 2)
	return var_y_axis


# 計算垂直軸的加權平均值
def average_pressure_value_y_axis(my_array):

	# 水平軸總和
	y_axis_array = np.sum(my_array, axis = 1)
	print(y_axis_array)

	average_y_axis = 0
	size = len(y_axis_array)
	for i in y_axis_array:
		average_y_axis = average_y_axis + (1/size * i)
	print('average_y_axis =', average_y_axis)
	return average_y_axis


# 以重心為圓心，半徑為1.0，以此遞增，所圍成的圓，計算圓中壓中的壓力值點數與總的壓力值點數的比例
def ratio_of_points(my_array):

	xg, yg =  center_of_gravity(my_array)
	# point_all = pressure_points(my_array)
	binary_array = binarization(my_array)
	print(binary_array)

	distance_list = []
	row = 20
	col = 11
	radius_1 = 1.0

	for row_index, row_element in enumerate(binary_array):
		for col_index, col_element in enumerate(row_element):
			if(col_element == 1023):
				distance = np.sqrt((xg - col_index) ** 2 + (yg - row_index) ** 2)
				print('distance =', distance)
				distance_list.append(distance)

	distance_array = np.sort(np.array(distance_list))
	print('distance_array =', distance_array)
	ratio_max = np.max(distance_array)
	print(ratio_max)
	ratio_list = []
	count = 0

	while count <= math.ceil(ratio_max):
		ratio = np.size(distance_array[distance_array < radius_1 * count])/np.size(distance_array)
		print('ratio =', ratio)
		ratio_list.append(round(ratio, 3))
		count += 1
	# ratio_array = np.array(ratio_list)
	print(ratio_list)

	return ratio_list


# 以重心為圓心，半徑為1.0，以此遞增，所圍成的圓，計算圓中壓中的壓力值*個數與壓力總和的比例，找合適半徑
def ratio_of_pressure_value(my_array):

	xg, yg =  center_of_gravity(my_array)
	# point_all = pressure_points(my_array)
	# print(xg)
	# print(yg)

	distance_list = []
	pressure_value_list = []
	row = 20
	col = 11
	radius_1 = 1.0
	count = 0

	for row_index, row_element in enumerate(my_array):
		for col_index, col_element in enumerate(row_element):
			if(col_element > 0):
				distance = np.sqrt((xg - col_index) ** 2 + (yg - row_index) ** 2)
				distance_list.append(distance)
				pressure_value_list.append(col_element)

	# print(distance_list)
	pres_val_ratio_list = []
	pres_val_ratio = 0
	# print(np.sum(my_array))

	while count < (max(distance_list) + 1):
		# print('count =', count)
		i = 0
		while i < len(distance_list):
			# print(distance_list[i])
			if (distance_list[i] > radius_1 * (count-1)) and (distance_list[i] < radius_1 * count):
				# print('distance =', distance_list[i])
				# print('pressure value =', pressure_value_list[i])
				pres_val_ratio = pressure_value_list[i] + pres_val_ratio
				# print('pres_val_ratio =', pres_val_ratio)
			i = i+1
		# print('pres_val_ratio =', pres_val_ratio)
		value_ratio = pres_val_ratio / np.sum(my_array)
		# print('value_ratio =', value_ratio)
		pres_val_ratio_list.append(round(value_ratio, 3))
		count = count + 1

	# print(pres_val_ratio_list)
	return pres_val_ratio_list



	# return ratio_list

# Local binary patterns
def local_binary_patterns(my_array):

	xg, yg =  center_of_gravity(my_array)


def center_of_gravity_pressure_value(my_array):

	xg, yg =  center_of_gravity(my_array)
	xg_bottom = int(xg)
	xg_top = math.ceil(xg)
	yg_bottom = int(yg)
	yg_top = math.ceil(yg)

	print('xg_bottom =', xg_bottom)
	print('xg_top =', xg_top)
	print('yg_bottom =', yg_bottom)
	print('yg_top =', yg_top)

	f_Q11 = my_array[yg_bottom][xg_bottom]
	f_Q12 = my_array[yg_bottom][xg_top]
	f_Q21 = my_array[yg_top][xg_bottom]
	f_Q22 = my_array[yg_top][xg_top]
	print(f_Q11)
	print(f_Q12)
	print(f_Q21)
	print(f_Q22)


	A = np.array([xg_top - xg, xg - xg_bottom])
	B = np.array([[f_Q11, f_Q12],
				[f_Q21, f_Q22]])
	C = np.array([[yg_top - yg],
				[yg - yg_bottom]])
	print(A)
	print(B)
	print(C)

	# center_of_gravity_pressure_value
	cg_pres_val = A.dot(B)
	print(cg_pres_val)
	cg_pres_val = cg_pres_val.dot(C)
	print(cg_pres_val)
	cg_pres_val = round(cg_pres_val[0])

	return cg_pres_val





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

# 以重心為圓心，畫橢圓，定半徑



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

	xg, yg = center_of_gravity(original_array)

	# 非零壓力值的加權平均值list
	average_pres_val_list = []
	# 非零壓力值變異數list
	variance_1D_list = []
	# 垂直軸變異數 list
	average_pres_val_y_axis_list = []
	# 垂直軸加權平均值 list
	variance_y_axis_list = []
	# 定圓心，半徑內壓力總和/整張影像壓力總和 list
	ratio_of_pressure_value_list = []


	xg_list = []
	yg_list = []
	ratio_list = []

	for item in raw_data:
		item_array = np.array(item)[:220]
		item_array = item_array.reshape(row, col)
		nonzero_item_array = nonzero_pressure_value(item_array)
		average_pres_val = nonzero_average_pressure_value(nonzero_item_array)
		variance_1D = nonzero_variance_pressure_value(nonzero_item_array, average_pres_val)
		aver_pres_val_y_axis = average_pressure_value_y_axis(item_array)
		var_y_axis = variance_y_axis(item_array)
		pres_val_ratio_list = ratio_of_pressure_value(item_array)

	# 	# ratio = ratio_of_points(item_array)
	# 	# print(ratio)
	# 	# xg, yg = center_of_gravity(item_array)
	# 	# xg = int(round(xg))
	# 	# yg = int(round(yg))


		average_pres_val_list.append(average_pres_val)
		variance_1D_list.append(variance_1D)
		average_pres_val_y_axis_list.append(aver_pres_val_y_axis)
		variance_y_axis_list.append(var_y_axis)
		# ratio_of_pressure_value_list.append(pres_val_ratio_list)

	# print(ratio_of_pressure_value_list[0:5])

	# df = pd.DataFrame(ratio_of_pressure_value_list)
	# print(df.head)
	# df.to_csv('test.csv')



	print('average_pres_val_list =', average_pres_val_list)
	print('variance_1D_list =', variance_1D_list)
	print('variance_y_axis_list =', variance_y_axis_list)
	print('average_pres_val_y_axis_list =', average_pres_val_y_axis_list)
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





