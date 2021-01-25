import pandas as pd
import cv2

f=pd.read_csv('./final.csv')

n=open('./csv_to_tfrecord_ready.csv','w')
n.write('name,x_center,y_center,width,height,class\n')

for i in f.iterrows():
	n,x1,y1,x2,y2,c=i[1]
	a=cv2.imread(n)
	w,h=x2-x1,y2-y1
	if w<2 or h<2:	
		continue
	x_center,y_center=x1+w/2,y1+h/2
	y,x,channels=a.shape
	x_center,y_center,width,height=x_center/x,y_center/y,w/x,h/y
	n.write('{},{},{},{},{},{}\n'.format(n,x_center,y_center,width,height,c))
