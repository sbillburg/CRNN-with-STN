import cv2
import numpy as np
import linecache

from config import width, height, label_len, lexicon_dic_path, file_list, file_list_val, img_folder, \
    characters
    
# load train data
file_list_full = file_list.readlines()
file_list_len = len(file_list_full)


# load validation data
file_list_val_full = file_list_val.readlines()
file_list_val_len = len(file_list_val_full)


def img_gen(batch_size=50, input_shape=None):
    x = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
    y = np.zeros((batch_size, label_len), dtype=np.uint8)
    # y = np.zeros((batch_size, ), dtype=np.uint8)

    while True:
        for ii in range(batch_size):

            while True:  # abandon the lexicon which is longer than 16 characters
                pick_index = np.random.randint(0, file_list_len - 1)
                file_list_full_split = [m for m in file_list_full[pick_index].split()]
                lexicon = linecache.getline(lexicon_dic_path, int(file_list_full_split[1]) + 1).strip("\n")
                img_path = img_folder + file_list_full_split[0][1:]
                img = cv2.imread(img_path)
                # abandon the lexicon which is longer than 16 characters, because I set the label_len = 16, you can change it anyway.
                # some dataset images may be damaged during unzip
                if (img is not None) and len(lexicon) <= label_len:
                    img_size = img.shape  # (height, width, channels)
                    if img_size[1] > 2 and img_size[0] > 2:
                        break
            if (img_size[1]/img_size[0]*1.0) < 6.4:
                img_reshape = cv2.resize(img, (int(31.0/img_size[0]*img_size[1]), height))
                mat_ori = np.zeros((height, width - int(31.0/img_size[0]*img_size[1]), 3), dtype=np.uint8)
                out_img = np.concatenate([img_reshape, mat_ori], axis=1).transpose([1, 0, 2])
            else:
                out_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                out_img = np.asarray(out_img).transpose([1, 0, 2])

            # due to the explanation of ctc_loss, try to not add "-" for blank
            while len(lexicon) < label_len:
                lexicon += "-"

            x[ii] = out_img
            y[ii] = [characters.find(c) for c in lexicon]
        yield [x, y, np.ones(batch_size) * int(input_shape[1] - 2), np.ones(batch_size) * label_len], y


def img_gen_val(batch_size=1000):
    x = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
    # y = np.zeros((batch_size, label_len), dtype=np.uint8)
    y = []

    while True:
        for ii in range(batch_size):

            while True:  
                pick_index = np.random.randint(0, file_list_val_len - 1)
                file_list_full_split = [m for m in file_list_val_full[pick_index].split()]
                lexicon = linecache.getline(lexicon_dic_path, int(file_list_full_split[1]) + 1).strip("\n")
                img_path = img_folder + file_list_full_split[0][1:]
                img = cv2.imread(img_path)
                # abandon the lexicon which is longer than 16 characters, because I set the label_len = 16, you can change it anyway.
                # some dataset images may be damaged during unzip
                if (img is not None) and len(lexicon) <= label_len:
                    img_size = img.shape  # (height, width, channels)
                    if img_size[1] > 2 and img_size[0] > 2:
                        break

            if (img_size[1]/img_size[0]*1.0) < 6.4:
                img_reshape = cv2.resize(img, (int(31.0/img_size[0]*img_size[1]), height))
                mat_ori = np.zeros((height, width - int(31.0/img_size[0]*img_size[1]), 3), dtype=np.uint8)
                out_img = np.concatenate([img_reshape, mat_ori], axis=1).transpose([1, 0, 2])
            else:
                out_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                out_img = np.asarray(out_img).transpose([1, 0, 2])

            x[ii] = out_img
            y.append(lexicon)
        yield x, y
