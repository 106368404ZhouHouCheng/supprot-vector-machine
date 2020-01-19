'''

editor: Jones
date: 2019/07/29
content: 
1. txt to csv

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def thresholding(threshold_array):
	min_threshold = 50
	threshold_array[threshold_array < min_threshold] = 0
	return threshold_array


def variance(my_array):

	# 垂直軸變異數
	vertical_var = int(np.var(np.sum(my_array, axis = 0)))
	# 水平軸變異數
	horizonal_var = int(np.var(np.sum(my_array, axis = 1)))

	return vertical_var, horizonal_var

# 主程序
def main():
	# open file 
	f = open('data_txt/sitting data/秉渝1217_sitting.txt', 'r',encoding = 'utf-8')

	raw_data_list = []
	for x in f:
		raw_data_list.append(x)
	print(len(raw_data_list))

	countA = 0
	# 坐姿
	sitting_list = []

	for item in raw_data_list[1:]:
		if item[-2] == '5':
			countA += 1
			sitting_list.append(item)

	print('len =', len(sitting_list)) 

	original_array = np.array(sitting_list[-1].split(),dtype = int)[:220].reshape(20,11)

	# # 100 frame 子暘
	# sitting_list = sitting_list[100:200]

	# 100 frame 育誠
	# sitting_list = raw_data_list[200:300]

	# 100 frame 佳芸
	# sitting_list = raw_data_list[100:200]

	# 100 frame 阿杜
	# sitting_list = sitting_list[:100]

	# 100 frame 建宇
	# sitting_list = sitting_list[-100:]

	# 100 frame 思琪
	# sitting_list = sitting_list[-100:]

	# 100 frame 竑量
	# sitting_list = sitting_list[-100:]

	# 100 frame 培健
	# sitting_list = sitting_list[-100:]

	# 100 frame 皓菘
	# sitting_list = sitting_list[-100:]

	# 100 frame 鵬翔
	# sitting_list = sitting_list[-100:]

	# 100 frame 沛忱
	# sitting_list = sitting_list[-100:]

	# 100 frame 嘉宏
	# sitting_list = sitting_list[-100:]

	# 100 frame 燕鴻
	# sitting_list = sitting_list[-100:]

	# 100 frame 顯郡
	# sitting_list = sitting_list[-100:]

	# 100 frame 沛臻
	# sitting_list = sitting_list[-100:]

	# 100 frame 翊嘉
	# sitting_list = sitting_list[-100:]

	# 100 frame 郅博
	# sitting_list = sitting_list[-100:]

	# 100 frame Jones
	# sitting_list = raw_data_list[100:200]

	# 100 frame 秉渝

	sitting_list = sitting_list[-100:]

	all_array = []

	for i in sitting_list:
		j = np.array(i.split(),dtype = int)[:220]
		all_array.append(j)

	new_array = np.array(all_array)

	print(len(new_array))

	columns = []
	i = 0
	while i < 220:
		col = 'pixel%d' %i
		i = i + 1
		columns.append(col)


	df = pd.DataFrame(new_array, columns = columns)

	# label 0, 坐姿
	# label 1, 睡姿

	label_0 = np.zeros(100, dtype=int)
	# print(label_0)
	df.insert(220, "label", label_0, True)
	print(df.head())

	# df.rename(columns={'pixel220':'label'}, inplace=True)

	df.to_csv("1217秉渝_sitting.csv" , encoding = "utf-8")


	# 顯示圖形
	fig, ax = plt.subplots()
	plt.title('Sitting Positions')
	plt.imshow(original_array[::-1], cmap=plt.cm.jet)
	plt.ylim((-0.5, 19.5))
	plt.yticks(np.arange(0,19,1.0))
	plt.colorbar()
	plt.show()

# 程式起點
if __name__ == '__main__':
	main()
