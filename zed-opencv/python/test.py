import numpy as np
from PIL import Image
import csv
# def export_list_csv(export_list, csv_dir):
# 
#     with open(csv_dir, "w") as f:
#         writer = csv.writer(f, lineterminator='/n',delimiter=' ',)
# 
#         if isinstance(export_list[0], list): #多次元の場合
#             writer.writerows(export_list)
# 
#         else:
#             writer.writerow(export_list)
# export_list_csv([1,2,2],'aa.txt')
# print(round(0.12344989797199, 10))
fn='C:/00_work/05_src/zed2/zed-opencv/python/data_1012/pos1/reconstruction-000001.depth-ZED_22378008.png'
depth = np.array(Image.open(fn))
k=0