# supprot-vector-machine
使用支撐向量機區分坐臥姿態

1.壓力影像（Bedsheet Sensor Raw Pressure Map）20 * 11 = 220 
2.前處理 Part 1（Pre-Processing Part1）:定門檻值(Threshold)，將低於門檻值的壓力值變為0，等於或大於的則保持其原來的壓力值，則得到新的壓力影像（New Pressure Image）
3.特徵提取（Feature Extraction）,對壓力影像提取特徵（提取特徵時，對壓力影像為0的壓力值不做計算，New Pressure Image --> New Nonzero Pressure Image）
	3.1：壓力平均值
	3.2：將壓力影像轉直方圖，計算其直方圖的變異數
	3.3：將X軸的壓力值疊加，只看Y軸的壓力值，則得到一個一維包含20個壓力值的陣列（y axis array）,計算y axis array的變異數（variance）
	3.4：將X軸的壓力值疊加，只看Y軸的壓力值，則得到一個一維包含20個壓力值的陣列（y axis array）,計算y axis array的加權平均值(mean)
	3.5：找壓力影像的重心，加半徑畫圓，圓内點數/總點數，找合適半徑
	3.6：找壓力影像的重心，加半徑畫圓，圓内壓力值總和/整張影像壓力總和，找合適半徑

4.坐臥姿態區分（sitting or Lying Classification）：將上述提取的特徵值放進支撐向量機（Support Vector Machine, SVM）做訓練，將生成的模型（sitting_lying_diff_model）用作在整晚的睡眠品質分析中區分受測者是坐姿還是臥姿
