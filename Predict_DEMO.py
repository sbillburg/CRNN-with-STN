import os
import cv2
import numpy as np
from keras.models import load_model
from keras import backend as bknd
import string
from STN.spatial_transformer import SpatialTransformer

width = 200
height = 31
label_len = 16
characters = '0123456789'+string.ascii_lowercase+'-'

imgReadPath = 'PATH_TO_IMAGE_FOLDER'
fileList = os.listdir(imgReadPath)
img_list = np.zeros((len(fileList), width, height, 3), dtype=np.uint8)


def loss_mean_squared(y_true, y_pred):
    return bknd.mean(bknd.square(y_pred - y_true), axis=-1)/25


ii = 0
for i in fileList:

    img_path = imgReadPath + '/' + i
    img = cv2.imread(img_path)
    img_size = img.shape
    if (img_size[1] / img_size[0] * 1.0) < 6:
        img_reshape = cv2.resize(img, (int(31.0 / img_size[0] * img_size[1]), height))

        mat_ori = np.zeros((height, width - int(31.0 / img_size[0] * img_size[1]), 3), dtype=np.uint8)
        out_img = np.concatenate([img_reshape, mat_ori], axis=1).transpose([1, 0, 2])
    else:
        out_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        out_img = np.asarray(out_img).transpose([1, 0, 2])

    img_list[ii] = np.asarray(out_img)
    ii += 1

model = load_model('PATH_TO_WEIGHT_FILE')
''' if you want to load model with STN, please use
model = load_model('PATH_TO_WEIGHT_FILE', custom_objects={'SpatialTransformer': SpatialTransformer})'''

y_pred = model.predict(img_list)
shape = y_pred[:, 2:, :].shape
ctc_decode = bknd.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
out = bknd.get_value(ctc_decode)[:, :label_len]

out_list = []
for m in range(len(fileList)):
    result_str = ''.join([characters[k] for k in out[m]])
    out_list.append(result_str)

print(out_list)
