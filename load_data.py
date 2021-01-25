import cv2,os,glob
import numpy as np
import pandas as pd

csv='./d<2.csv'
im='./images'

f=pd.read_csv(csv)

img_name1='none'
count=0
for i in f.iterrows():
	img_name,x1,y1,x2,y2,clas=i[1]
	x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
	if img_name!=img_name1:
		if count==1:
			cv2.imshow('l',img)
			cv2.waitKey()
		img=cv2.imread(img_name)
		if clas==1:
			cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
		elif clas==2:
			cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
		elif clas==3:
			cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
		elif clas==4:
			cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255),2)
		elif clas==5:
			cv2.rectangle(img,(x1,y1),(x2,y2),(100,100,100),2)
		else:
			cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0),2)
		
		#cv2.imshow('p',img)
		#cv2.waitKey()
		img_name1=img_name
		#print(2)
	else:
		count=1
		#print(3)
		if clas==1:
			cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
		elif clas==2:
			cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
		elif clas==3:
			cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
		elif clas==4:
			cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255),2)
		elif clas==5:
			cv2.rectangle(img,(x1,y1),(x2,y2),(100,100,100),2)
		else:
			cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0),2)
		
