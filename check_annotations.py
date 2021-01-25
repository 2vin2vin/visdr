import cv2,glob,os
import numpy as np
import pandas as pd

an='./annotations'
im='./images'

'''3-car,van 1-pedestrian,people 2-bicycle,tricycle,motor 4-truck 5-bus 6-others
'''

out=open('./d<2.csv','w')
out.write('img_name,x1,y1,x2,y2,class\n')

for i in glob.glob(an+'/*txt'):
	f=((open(i,'r').read()).split('\n'))[:-1]
	img_name=im+'/'+(i.split('/')[-1])[:-3]+'jpg'
	j=0
	print(i,f[-1])
	#input()
	while j<(len(f)):
		print(f[j],j)
		if f[j][-1]==',':
			f[j]=f[j][:-1]
		x,y,w,h,_,c,_,d=f[j].split(',')
		x,y,w,h,clas,d=int(x),int(y),int(w),int(h),int(c),int(d)
		x1,y1,x2,y2=x,y,x+w,y+h	
		j+=1	
		if d>=2:
			continue
		if clas<3:
			clas=1
		elif clas==3:
			clas=2
		elif clas==4:
			clas=3
		elif clas==5:
			clas=3
		elif clas==6:
			clas=4
		elif clas==7:
			clas=2
		elif clas==8:
			clas=2
		elif clas==9:
			clas=5
		elif clas==10:
			clas=2
		else:
			clas=6
		out.write('{},{},{},{},{},{}\n'.format(img_name,x1,y1,x2,y2,clas))
		#j+=1


'''
	
	f=pd.read_csv(i)
	img_name=im+'/'+(i.split('/')[-1])[:-3]+'jpg'
	img=cv2.imread(img_name)
	for j in f.iterrows():
		#print(j)
		cv2.rectangle(img,(j[1][0],j[1][1]),(j[1][0]+j[1][2],j[1][1]+j[1][3]),(0,255,0),2)
	img1=cv2.resize(img.copy(),(640,480))
	cv2.imshow('l',img1)
	cv2.waitKey()
'''
'''
		x1,y1,x2,y2=j[1][0],j[1][1],j[1][0]+j[1][2],j[1][1]+j[1][3]
		clas=j[1][5]
		#if j[1][-1]>0:
		#	continue
		if clas<3:
			clas=1
		elif clas==3:
			clas=2
		elif clas==4:
			clas=3
		elif clas==5:
			clas=3
		elif clas==6:
			clas=4
		elif clas==7:
			clas=2
		elif clas==8:
			clas=2
		elif clas==9:
			clas=5
		elif clas==10:
			clas=2
		else:
			clas=6
		out.write('{},{},{},{},{},{}\n'.format(img_name,x1,y1,x2,y2,clas))
'''
