#  Copyright (c) 2020.
#  Airodyn Systems Private Limited

# ______________________________________
#   File Name:          
#   Author:             Aviral Sharma
# ______________________________________
import Object_Detectors.config.dataset_config as config
from AirodynCV.utils.tfannotation import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import os
import pandas as pd


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


CLASSES={'Pedestrian':1,'Twowheeler':2,'Fourwheeler':3,'Truck':4,'Bus':5,'Other':6}
def main(_):
    f = open(config.CLASSES_FILE, "w")

    for (k, v) in CLASSES.items():
        item = ("item {\n"
                "\tid: " + str(v) + "\n"
                "\tname: '" + k + "'\n"
                "}\n")
        f.write(item)
    f.close()

    # initialize data dictionary to map each image filename to all bounding boxes associated with image,
    # load contents of annotation file
    D = {}
    rows = pd.read_csv('./augment.csv')

    for row in rows.iterrows():
        row = row[1]
        imagePath, width, height,startX, startY, endX, endY,label = row
        (startX, startY) = (float(startX), float(startY))
        (endX, endY) = (float(endX), float(endY))

        if label not in config.CLASSES:
            continue

        # building path to input image, grabbing other bounding boxes and labels associated with image
        # path, labels and bounding box lists respectively
        #imagePath='/images'+'/'+imagePath
        p = os.path.sep.join(['/home/ubuntu/vd', imagePath])
        p=imagePath
        b = D.get(p, [])

        b.append((label,width,height, (startX, startY, endX, endY)))
        D[p] = b

    (trainKeys, testKeys) = train_test_split(list(D.keys()), test_size=0.2, random_state=42)

    datasets = [
        ("train", trainKeys, './records/train.tfrecord'),
        ("test", testKeys, './records/test.tfrecord')]

    for (dType, keys, outputPath) in datasets:
        print("[UPDATE] processing '{}'...".format(dType))
        writer = tf.io.TFRecordWriter(outputPath)
        total = 0
        for k in keys:

            encoded = tf.io.gfile.GFile(k).read()
            #encoded = encoded.tobytes()
            
            pilImage = Image.open(io.BytesIO(encoded))
            (w, h) = pilImage.size[:2]
            
            filename = k.split(os.path.sep)[-1]
            encoding = filename[filename.rfind(".") + 1:]
            
            #tfAnnot = TFAnnotation()
            image = bytes_feature(encoded.encode('utf8')
            encoding = bytes_feature(encoding)
            filename = bytes_feature(filename)
            width = w
            height = h

            for (label, w,h,(startX, startY, endX, endY)) in D[k]:

                xMin = startX / w
                xMax = endX / w
                yMin = startY / h
                yMax = endY / h

                xMins.append(xMin)
                xMaxs.append(xMax)
                yMins.append(yMin)
                yMaxs.append(yMax)
                textLabels.append(label.encode("utf8"))
                classes.append(CLASSES[label])
                #tfAnnot.difficult.append(0)

                total += 1

            #features = tf.train.Features(feature=tfAnnot.build())
            example = tf.train.Example(tf.train.Features(feature={
           'image/height': int64_feature(height),
           'image/width': int64_feature(width),
           'image/filename': bytes_feature(filename),
           'image/source_id': bytes_feature(filename),
           'image/encoded': bytes_feature(encoded_jpg),
           'image/format': bytes_feature(image_format),
           'image/object/bbox/xmin': float_list_feature(xMins),
           'image/object/bbox/xmax': float_list_feature(xMaxs),
           'image/object/bbox/ymin': float_list_feature(yMins),
           'image/object/bbox/ymax': float_list_feature(yMaxs),
           'image/object/class/text': bytes_list_feature(textLabels),
           'image/object/class/label': int64_list_feature(classes), }))

            writer.write(example.SerializeToString())

        writer.close()
        print("[UPDATE] {} examples saved for '{}'".format(total, dType))


if __name__ == "__main__":
    tf.compat.v1.app.run()
