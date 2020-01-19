'''

editor: Jones
date: 2019/07/29
content: 
1. txt to csv

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import cv2

# 門檻值
def min_thresholding(threshold_array):

	min_threshold = 50
	# print(min_threshold)
	threshold_array[threshold_array <= min_threshold] = 0

	return threshold_array


# 主程序
def main():
	# open file 
	f = open('data_txt/lying data/秉渝1217_lying.txt', 'r',encoding = 'utf-8')

	raw_data_list = []
	for x in f:
		raw_data_list.append(x)
	print(len(raw_data_list))

	# 正躺
	face_up_list = []
	# # 右侧躺
	face_right_list = []
	# # 左侧躺
	face_left_list = []
	# # 俯卧
	face_down_list = []

	for item in raw_data_list[1:]:
		if item[-2] == '0':
			face_up_list.append(item)
		elif item[-2] == '1':
			face_right_list.append(item)
		elif item[-2] == '2':
			face_left_list.append(item)
		elif item[-2] == '3':
			face_down_list.append(item)


	print('countA =',len(face_up_list)) 
	print('countB =',len(face_right_list))
	print('countC =',len(face_left_list)) 
	print('countD =',len(face_down_list))

	original_array = np.array(face_down_list[-1].split(),dtype = int)[:220].reshape(20,11)

	# img_200 = cv2.resize(original_array/4,(220,400), interpolation=cv2.INTER_CUBIC)

	split_index = -25

	# sitting_list = raw_data_list[50:150]

	# lying_list = raw_data_list[-375:-350] + raw_data_list[-275:-250] + raw_data_list[-100:-75] + raw_data_list[split_index:]
	# print('lying =',len(lying_list))

	# # 100 frame 子暘
	# lying_list = face_up_list[split_index:] + face_right_list[split_index:] + face_left_list[split_index:] + face_down_list[split_index:]
	# print('lying =',len(lying_list))

	# 100 frame 育誠
	# lying_list = face_up_list[split_index:] + face_right_list[100:125] + face_right_list[275:300] + face_right_list[split_index:]
	# print('lying =',len(lying_list))

	# 100 frame 佳芸
	# lying_list = face_up_list[split_index:] + face_right_list[split_index:] + face_left_list[split_index:] + face_down_list[split_index:]
	# print('lying =',len(lying_list))

	# 100 frame 阿杜
	# lying_list = face_up_list[175:200] + face_right_list[split_index:] + face_left_list[split_index:] + face_down_list[100:125]
	# print('lying =',len(lying_list))

	# 100 frame 建宇
	# lying_list = face_up_list[split_index:] + face_right_list[split_index:] + face_left_list[split_index:] + face_down_list[split_index:]
	# print('lying =',len(lying_list))

	# 100 frame 思琪
	# lying_list = face_up_list[split_index:] + face_right_list[split_index:] + face_left_list[split_index:] + face_down_list[75:100]
	# print('lying =',len(lying_list))

	# 100 frame 竑量
	# lying_list = face_up_list[split_index:] + face_right_list[split_index:] + face_left_list[split_index:] + face_down_list[split_index:]
	# print('lying =',len(lying_list))

	# 100 frame 培健
	# lying_list = face_up_list[split_index:] + face_right_list[split_index:] + face_left_list[split_index:] + face_down_list[split_index:]
	# print('lying =',len(lying_list))

	# 100 frame 皓菘
	# lying_list = face_up_list[split_index:] + face_right_list[split_index:] + face_left_list[split_index:] + face_down_list[150:175]
	# print('lying =',len(lying_list))

	# 100 frame 鵬翔
	# lying_list = face_up_list[250:275] + face_up_list[split_index:] + face_left_list[split_index:] + face_down_list[split_index:]
	# print('lying =',len(lying_list))

	# 100 frame 沛忱
	# lying_list = face_up_list[split_index:] + face_right_list[split_index:] + face_left_list[split_index:] + face_down_list[split_index:]
	# print('lying =',len(lying_list))

	# 100 frame 嘉宏
	# lying_list = face_up_list[250:275] + face_right_list[split_index:] + face_left_list[split_index:] + face_down_list[100:125]
	# print('lying =',len(lying_list))

	# 100 frame 燕鴻
	# lying_list = face_up_list[150:175] + face_right_list[split_index:] + face_left_list[split_index:] + face_down_list[50:75]
	# print('lying =',len(lying_list))

	# 100 frame 顯郡
	# lying_list = face_up_list[split_index:] + face_right_list[split_index:] + face_left_list[200:225] + face_down_list[200:225]
	# print('lying =',len(lying_list))

	# 100 frame 沛臻
	# lying_list = face_up_list[split_index:] + face_right_list[split_index:] + face_left_list[split_index:] + face_down_list[split_index:]
	# print('lying =',len(lying_list))

	# 100 frame 翊嘉
	# lying_list = face_up_list[split_index:] + face_right_list[split_index:] + face_left_list[200:225] + face_down_list[split_index:]
	# print('lying =',len(lying_list))


	# 100 frame 郅博
	# lying_list = face_up_list[split_index:] + face_right_list[split_index:] + face_left_list[175:200] + face_down_list[split_index:]
	# print('lying =',len(lying_list))

	# 100 frame Jones
	# lying_list = raw_data_list[1400:1425] + raw_data_list[1600:1625] + raw_data_list[2000:2025] + raw_data_list[2400:2425]
	# print('lying =',len(lying_list))

	# 100 frame 秉渝
	lying_list = face_up_list[split_index:] + face_right_list[split_index:] + face_left_list[175:200] + face_down_list[split_index:]
	print('lying =',len(lying_list))

	lying_array = []

	for i in lying_list:
		j = np.array(i.split(),dtype = int)[:220]
		lying_array.append(j)

	lying_array = np.array(lying_array)

	print(len(lying_array))

	columns = []
	i = 0
	while i < 220:
		col = 'pixel%d' %i
		i = i + 1
		columns.append(col)

	df = pd.DataFrame(lying_array, columns = columns)

	# # label 0, 坐姿
	# label 1, 睡姿

	label_1 = np.ones(100, dtype=int)
	df.insert(220, "label", label_1, True)
	print(df)
	df.to_csv("data_csv/1216lyingData/1217秉渝_lying.csv" , encoding = "utf-8")


	# 顯示圖形
	fig, ax = plt.subplots()

	plt.title('Sleeping Positions')
	plt.imshow(original_array[::-1], cmap=plt.cm.jet)
	plt.ylim((-0.5, 19.5))
	plt.yticks(np.arange(0,19,1.0))
	plt.colorbar()
	plt.show()


	# fig, ax = plt.subplots()
	# plt.title('resize')
	# plt.imshow(img_200, cmap = plt.cm.jet)
	# # plt.colorbar()
	# plt.show()





# 程式起點
if __name__ == '__main__':
	main()
