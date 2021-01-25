import cv2
import numpy as np
import pandas as pd
import glob,os,random

out=open('./zoom.csv','w')
out.write('name,x1,y1,x2,y2,class\n')
def zoom_out(img,zoom,random_value,b,ac,name):
    a=img.copy()
    x,y,w,h=b
    w=w-x
    h=h-y
    if h==0:
         h=1
    if w==0:
        w=1
    val=w/h
    qw=(zoom/100)*random_value
    qwx=int(qw*img.shape[1])
    qwy=int(qw*img.shape[0])
    #print(qwx,qwy,a.shape)
    if qwx==0:
        qwx=1
    if qwy==0:
        qwy=1
    img2=cv2.resize(a,(a.shape[1]-qwx*2,a.shape[0]-qwy*2))
    img3=np.ones(img.shape,dtype=np.uint8)
    img3[qwy:-qwy,qwx:-qwx,:]=img2
    x=int((x/a.shape[1])*img2.shape[1])
    y=int((y/a.shape[0])*img2.shape[0])
    w=int((w/a.shape[1])*img2.shape[1])
    h=int((h/a.shape[0])*img2.shape[0])
    #print((qwx+x,qwy+y),(qwx+x+w,qwy+y+h),(a.shape[1]-qwx*2,a.shape[0]-qwy*2))
    #print(img2.shape,x,y,w,h)
    #cv2.rectangle(img3,(qwx+x,qwy+y),(qwx+x+w,qwy+y+h),(0,255,0),2)
    #cv2.rectangle(a,(b[0],b[1]),(b[2],b[3]),(0,255,0),2)
    #cv2.imshow('p',img3)
    #cv2.imshow('po',a)
    #cv2.waitKey(1)
    if ac==1:
        name='./zoom_out_images/'+name.split('/')[-1]
        print(name)
        cv2.imwrite(name,img3)
    cd=[qwx+x,qwy+y,qwx+x+w,qwy+y+h]
    return cd

f=pd.read_csv('./d<2.csv')
name1='none'
ac=0
for i in f.iterrows():
	ac=0
	name,x1,y1,x2,y2,cl=i[1]
	
	if name1!=name:
		rv=random.random()
		name1=name
		img=cv2.imread(name)
		ac=1	
	b=[x1,y1,x2,y2]
	zoom=25
	cd=zoom_out(img,zoom,rv,b,ac,name)
	name=name='./zoom_out_images/'+name.split('/')[-1]
	out.write('{},{},{},{},{},{}\n'.format(name,cd[0],cd[1],cd[2],cd[3],cl))





