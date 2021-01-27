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


def main(_):
    f = open(config.CLASSES_FILE, "w")

    for (k, v) in config.CLASSES.items():
        item = ("item {\n"
                "\tid: " + str(v) + "\n"
                "\tname: '" + k + "'\n"
                "}\n")
        f.write(item)
    f.close()

    # initialize data dictionary to map each image filename to all bounding boxes associated with image,
    # load contents of annotation file
    D = {}
    rows = open(config.ANNOT_PATH).read().strip().split("\n")

    for row in rows[1:]:
        row = row.split(",")
        (imagePath, width, height, label, startX, startY, endX, endY) = row
        (startX, startY) = (float(startX), float(startY))
        (endX, endY) = (float(endX), float(endY))

        if label not in config.CLASSES:
            continue

        # building path to input image, grabbing other bounding boxes and labels associated with image
        # path, labels and bounding box lists respectively
        #imagePath='/images'+'/'+imagePath
        p = os.path.sep.join([config.BASE_PATH, imagePath])
        p=imagePath
        b = D.get(p, [])

        b.append((label,width,height, (startX, startY, endX, endY)))
        D[p] = b

    (trainKeys, testKeys) = train_test_split(list(D.keys()), test_size=config.TEST_SIZE, random_state=42)

    datasets = [
        ("train", trainKeys, config.TRAIN_RECORD),
        ("test", testKeys, config.TEST_RECORD)]

    for (dType, keys, outputPath) in datasets:
        print("[UPDATE] processing '{}'...".format(dType))
        writer = tf.io.TFRecordWriter(outputPath)
        total = 0
        for k in keys:

            encoded = tf.io.gfile.GFile(k.split("/")[-1], "rb").read()
            encoded = bytes(encoded)
            
            pilImage = Image.open(k)
            (w, h) = pilImage.size[:2]
            
            filename = k.split(os.path.sep)[-1]
            encoding = filename[filename.rfind(".") + 1:]
            
            tfAnnot = TFAnnotation()
            tfAnnot.image = encoded
            tfAnnot.encoding = encoding
            tfAnnot.filename = filename
            tfAnnot.width = w
            tfAnnot.height = h

            for (label, w,h,(startX, startY, endX, endY)) in D[k]:

                xMin = startX / w
                xMax = endX / w
                yMin = startY / h
                yMax = endY / h

                tfAnnot.xMins.append(xMin)
                tfAnnot.xMaxs.append(xMax)
                tfAnnot.yMins.append(yMin)
                tfAnnot.yMaxs.append(yMax)
                tfAnnot.textLabels.append(label.encode("utf8"))
                tfAnnot.classes.append(config.CLASSES[label])
                tfAnnot.difficult.append(0)

                total += 1

            features = tf.train.Features(feature=tfAnnot.build())
            example = tf.train.Example(features=features)

            writer.write(example.SerializeToString())

        writer.close()
        print("[UPDATE] {} examples saved for '{}'".format(total, dType))


if __name__ == "__main__":
    tf.compat.v1.app.run()
