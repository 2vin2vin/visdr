import cv2
import numpy as np
import pandas as pd
import glob,os,random,imutils,math


def up_to_down(img,random_value,b):
    bore=1#vertical flip
    '''if random number >0.5 then flip'''
    if random_value>0.5:
        a=img.copy()
        img3=a[::-1,:,:]
        img3=img3.copy()
        x,y,w,h=b
        w,h=w-x,h-y
        y=img.shape[0]-y
        coord=[x,y,x+w,y-h]
        return img3,coord
    else:
        return img,b

def left_to_right(img,random_value,b):
    bore=1#horizontal flip
    '''if random number >0.5 then flip'''
    if random_value>0.5:
        a=img.copy()
        img2 = a[:,::-1,:]
        img3=img2.copy()
        x,y,w,h=b
        w,h=w-x,h-y
        x=img.shape[1]-x
        coord=[x-w,y,x,y+h]
        return img3,coord
    else:
        return img,b

def left_or_right(img,translate,random_value,b):
    #translate
    '''if random number >0.5 then left or else right'''
    a=img.copy()
    img3=np.zeros(img.shape,dtype=np.uint8)
    #random_value=random.random()
    per=translate/100*random_value
    xm=int(per*img.shape[1])
    if xm==0:
        xm=1
    x,y,w,h=b
    w,h=w-x,h-y
    if random_value>0.5:
        img3[:,:-xm,:]=a[:,xm:,:]
        x=x-xm
        if (x+w)<0:
            m=0
        else:
            m=x+w
        if x<0:
            x=0
        coord=[x,y,m,y+h]
        return img3,coord
    else:
        img3[:,xm:,:]=a[:,:-xm,:]
        x=x+xm
        if x>a.shape[1]:
            x=a.shape[1]
        if (x+w)>a.shape[1]:
            m=a.shape[1]
        else:
            m=x+w
        coord=[x,y,m,y+h]
        return img3,coord


def up_or_down(img,translate,random_value,b):
    #translate
    '''if random number >0.5 then down or else up'''
    a=img.copy()
    img3=np.zeros(img.shape,dtype=np.uint8)
    per=translate/100*random_value
    xm=int(per*img.shape[1])
    if xm==0:
        xm=1
    x,y,w,h=b
    w,h=w-x,h-y
    if random_value>0.5:
        img3[xm:,:,:]=a[:-xm,:,:]
        y=y+xm
        if (y+h)>a.shape[0]:
            m=a.shape[0]
        else:
            m=y+h
        if y<0:
            y=0
        coord=[x,y,x+w,m]
        return img3,coord
    else:
        img3[:-xm,:,:]=a[xm:,:,:]
        y=y-xm
        if (y+h)<0:
            m=0
        else:
            m=y+h
        if y<0:
            y=0
        coord=[x,y,x+w,m]
        return img3,coord

def shear(img,sheared,random_value,b):
    a=img.copy()
    img3=np.zeros(a.shape,dtype=np.uint8)
    per=sheared/100*random_value
    xm=int(per*a.shape[1])
    if xm==0:
        sm=1
    i1=a.shape[0]-1
    alpha=xm/a.shape[0]
    beta=0
    j=0
    while i1>0:
        b1=a[i1,:,:]
        c=np.zeros(b1.shape,dtype=np.uint8)
        d=int(alpha*j)
        if d==0:
            d=1
        c[d:,:]=b1[:-d,:]
        img3[i1,:,:]=c
        i1-=1
        j+=1
    x,y,w,h=b
    w,h=w-x,h-y
    d1=int(alpha*(a.shape[0]-y))
    d2=int(alpha*(a.shape[0]-(y+h)))
    coord=[d2+x,y,d1+x+w,y+h]
    return img3,coord

def rotate(img,angle,random_value,b):
    a=img.copy()
    a1=a.copy()
    theta=angle*random_value
    p1=theta/180*22/7
    img3=imutils.rotate(a1,angle=theta)
    x,y,w,h=b
    w,h=w-x,h-y
    #######for bounding box rotation
    xcenter=int(x+w/2)
    ycenter=int(y+h/2)
    lt=np.array([[(x-xcenter),(-y+ycenter)]])
    rt=np.array([[(x+w-xcenter),(-y+ycenter)]])
    lb=np.array([[(x-xcenter),(-y-h+ycenter)]])
    rb=np.array([[(x+w-xcenter),(-y-h+ycenter)]])
    rotmat=np.array([[(math.cos(p1),-math.sin(p1)),(math.sin(p1),math.cos(p1))]])
    lt1=np.dot(lt,rotmat)
    rt1=np.dot(rt,rotmat)
    lb1=np.dot(lb,rotmat)
    rb1=np.dot(rb,rotmat)
    ttt=(lt1,rt1,lb1,rb1)
    smallx=lt1[0][0][0]
    smally=lt1[0][0][1]
    for i1 in ttt:
        if i1[0][0][0]<smallx:
            smallx=i1[0][0][0]
        if i1[0][0][1]<smally:
            smally=i1[0][0][1]
    yc,xc,cc=np.multiply(a.shape,0.5)
    xr=int(((xcenter-xc)*math.cos(p1))-(math.sin(p1)*(-ycenter+yc)))
    yr=int(((xcenter-xc)*math.sin(p1))+(math.cos(p1)*(-ycenter+yc)))
    if xr<0:
        pt1=xc+xr
    else:
        pt1=xr+xc
    if yr<0:
        pt2=yc-yr
    else:
        pt2=yc-yr
    coord=[int(pt1+smallx),int(pt2+smally),int(pt1-smallx),int(pt2-smally)]
    return img3,coord

def zoom_in(img,zoom,random_value,b):
    a=img.copy()
    a1=a.copy()
    x,y,w,h=b
    w,h=w-x,h-y
    qw=zoom*random_value/100
    imgnew=a1[int(qw*a.shape[0]):int(a.shape[0]-qw*a.shape[0]),
             int(qw*a.shape[1]):int(a.shape[1]-qw*a.shape[1])]
    x=int(x-qw*a.shape[1])
    y=int(y-qw*a.shape[0])
    w=x+w
    h=y+h
    img3=cv2.resize(imgnew,(a.shape[1],a.shape[0]))
    x1=int(x/imgnew.shape[1]*a.shape[1])
    y1=int(y/imgnew.shape[0]*a.shape[0])
    w1=int(w/imgnew.shape[1]*a.shape[1])
    h1=int(h/imgnew.shape[0]*a.shape[0])
    coord=[x1,y1,w1,h1]
    return img3,coord

def display(img3,b):
	cv2.rectangle(img3,(b[0],b[1]),(b[2],b[3]),(0,255,0),2)
	cv2.resize(img3,(640,480))
	cv2.imshow('kl',img3)

	cv2.waitKey()


out=open('./final.csv','w')
out.write('name,x1,y1,x2,y2,class\n')
f=pd.read_csv('./zoom.csv')
name1='none'
ac=0
for i in f.iterrows():
	ac=0
	name,x1,y1,x2,y2,cl=i[1]
	
	if name1!=name:
		rv=random.random()
		rv1=random.random()
		rv2=random.random()
		rv3=random.random()
		rv4=random.random()
		rv5=random.random()
		rv6=random.random()
		name1=name
		img=cv2.imread(name)
		ac=1	
	b=[x1,y1,x2,y2]
	#zoom=30
	#cd=zoom_out(img,zoom,rv,b,ac,name)
	translate,sheared,angle,zoom=20,20,60,10
	img3,b=up_to_down(img,rv1,b)
	#display(img3,b)
	img3,b=left_to_right(img3,rv2,b)
	#display(img3,b)
	img3,b=up_or_down(img3,translate,rv3,b)
	#display(img3,b)
	img3,b=left_or_right(img3,translate,rv4,b)
	#display(img3,b)
	img3,b=shear(img3,sheared,rv5,b)
	#display(img3,b)
	img3,b=rotate(img3,angle,rv6,b)
	#display(img3,b)
	#print(rv,rv1,rv2,rv3,rv4,rv5,rv6)
	#img3,b=zoom_in(img3,zoom,rv,b)
	cd=b
	if name1!=name:
		name='./final_images/'+name.split('/')[-1]
		cv2.imwrite(name,img3)
	name='./final_images/'+name.split('/')[-1]
	out.write('{},{},{},{},{},{}\n'.format(name,cd[0],cd[1],cd[2],cd[3],cl))
	#break
