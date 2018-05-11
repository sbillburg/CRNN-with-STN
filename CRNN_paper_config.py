import string

width = 200
height = 31
label_len = 16

characters = '0123456789'+string.ascii_lowercase+'-'
label_classes = len(characters)+1

# for batch_generator
lexicon_dic_path = '/media/junbo/DATA/OCR_datasets/max/lexicon.txt'
file_list = open('/media/junbo/DATA/OCR_datasets/max/annotation_train.txt', 'r')
file_list_val = open('/media/junbo/DATA/OCR_datasets/max/annotation_val.txt', 'r')
img_folder = '/media/junbo/DATA/OCR_datasets/max/90kDICT32px'


# for CRNN_with_STN
learning_rate = 0.0001  # 0.002 for default
load_model_path = '/home/junbo/PycharmProjects/test0_mnist/models/weights_best_STN.50-1.84.hdf5'
cp_save_path = '/home/junbo/PycharmProjects/test0_mnist/models/weights_best_STN.{epoch:02d}-{loss:.2f}.hdf5'
base_model_path = '/home/junbo/PycharmProjects/test0_mnist/models/weights_for_predict_STN.hdf5'
tb_log_dir = '/home/junbo/PycharmProjects/test0_mnist/CRC_n/paper_log'  # Optional
