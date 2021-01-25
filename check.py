import cv2
import pandas as pd

f='./zoom.csv'
f=pd.read_csv(f)
n1='lop'

for i in f.iterrows():
	n,x1,y1,x2,y2,c=i[1]
	img=cv2.imread(n)
	cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
	cv2.imshow('lkoi',img)
	cv2.waitKey()
