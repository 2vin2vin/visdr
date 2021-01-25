import cv2,glob,os
import pandas as pd
import numpy as np

import tensorflow as tf

csv='./d<2.csv'

f=pd.read_csv(csv)

def create(feature,label):
	tf_ex=tf.train.Example(features=tf.train.Features(feature={
	'Name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[features[0].encode('utf-8')])),
	'x1': tf.train.Feature(float_list=tf.train.FloatList(value=[features[1]])),
	'y1': tf.train.Feature(float_list=tf.train.FloatList(value=[features[2]])),
	'x2': tf.train.Feature(float_list=tf.train.FloatList(value=[features[3]])),
	'y2': tf.train.Feature(float_list=tf.train.FloatList(value=[features[4]])),
	'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode('utf-8')])),
	}))
	return tf_ex

#3-car,van 1-pedestrian,people 2-bicycle,tricycle,motor 4-truck 5-bus 6-others
with tf.io.TFRecordWriter('./d<2.tfrecords') as writer:
	for i in f.iterrows():
		print(i[1][0])
		img=cv2.imread(i[1][0])
		ht,wt,c=img.shape
		x1=i[1][1]/wt
		y1=i[1][2]/ht
		x2=i[1][3]/wt
		y2=i[1][4]/ht
		if i[1][5]==1:
			label='People'
		elif i[1][5]==2:
			label='Two-wheeler'
		elif i[1][5]==3:
			label='Four-wheeler'
		elif i[1][5]==4:
			label='Truck'
		elif i[1][5]==5:
			label='Bus'
		elif i[1][5]==6:
			label='Others'
		features=[i[1][0],x1,y1,x2,y2]
		example=create(features,label)
		writer.write(example.SerializeToString())
		break
writer.close()
