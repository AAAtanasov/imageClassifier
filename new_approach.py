import cv2
import numpy as np
import os.path
import random
import math
import matplotlib.pyplot as plt
from itertools import repeat
from sklearn.svm import LinearSVC
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
import itertools
import pickle
from sklearn.neural_network import MLPClassifier as mlp
from scipy.cluster.vq import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier

"""
Array used to store all the images for 10-fold cross validation
"""
ten_fold_array = [[] for i in repeat(None, 10)]

sub_folders_list = []
X_train = []
Y_train = []
X_test = []
Y_test = []
sift = cv2.xfeatures2d.SIFT_create()
PRE_ALLOCATION_BUFFER = 1000
des_list = []
train_dictionary = {}
test_dictionary = {}
all_features_dict = {}



def iterate_class_folders(number_of_classes):
    main_dir = "101_ObjectCategories"
    # sub_folders_list = []
    all_folder_names = os.listdir(main_dir)
    temp_folders = ['anchor', 'accordion', 'crocodile']
    # all_folder_names = temp_folders
    all_folder_names.remove(".DS_Store")

    for tempIndex, _ in enumerate(range(number_of_classes)):
        choice_item_index = random.randrange(len(all_folder_names))
        choice_item = all_folder_names.pop(choice_item_index - 1)
        sub_folders_list.append(choice_item)

    for folder in sub_folders_list:
        retrieve_image_from_folder(folder)


def retrieve_image_from_folder(folder_name):
    # image path and valid extensions
    image_dir = "101_ObjectCategories/" + folder_name  # specify your path here
    image_path_list = []
    valid_image_extensions = [".jpg", ".jpeg", ".png",
                              ".tif", ".tiff"]  # specify your valid extensions here
    valid_image_extensions = [item.lower() for item in valid_image_extensions]

    for file in os.listdir(image_dir):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        image_path_list.append(os.path.join(image_dir, file))

    list_of_images = image_path_list

    def random_index_assign(incoming_list, count_of_elems_to_extract, global_ten_fold_array, array_index):
        items_to_add = extract_from_array(incoming_list, count_of_elems_to_extract)
        global_ten_fold_array[array_index].extend(items_to_add)  # Extending global list containing files



    def extract_from_array(incoming_list, number_to_extract):
        result_items = []

        for tempIndex, _ in enumerate(range(int(number_to_extract))):
            choice_item = incoming_list.pop(0)
            result_items.append(choice_item)

        return result_items

    length_of_images = len(list_of_images)  # Number of images in folder
    real_element_division = length_of_images / 10  # Iterator
    real_index = 0  # Temporal value to store the current step state
    last_start_position = 0  # Temporal value to store the last addition value

    for elementIndex, _ in enumerate(range(10)):  # Enumerate over 10-fold list
        real_index += real_element_division  # Real valued index kept for state
        temp_index = math.floor(real_index - last_start_position)  # Integer value
        last_start_position += temp_index

        if last_start_position != length_of_images and elementIndex == 9:
            temp_index += length_of_images - last_start_position

        random_index_assign(list_of_images, temp_index,
                            ten_fold_array, elementIndex)


iterate_class_folders(101)
print(sub_folders_list)


def generate_sift_features(picture_path):
    image = cv2.imread(picture_path, cv2.COLOR_BGR2GRAY)
    kp, desc = sift.detectAndCompute(image, None)
    # nfeatures = initial_desc.shape[1]
    # padding = np.zeros((2, nfeatures), dtype=numpy.float64)
    # asd = np.vstack((kp, padding))
    # temp = initial_desc.astype('float')
    # descriptors = temp[:]
    # pickle.dump([kp.T, descriptors.T], open("../pickles/sifts/start.p", "wb"))

    return kp, desc


def extract_sift_features_from_array(input_arr, feature_dictionary, y_array):
    for picture in input_arr:
        feature_label = picture.split('/')[1].split('\\')[0]
        feature_name = feature_label + '.sift'
        # add resize if needed
        kp, descriptors = generate_sift_features(picture)
        if descriptors is None:
            continue
            
        if feature_name not in feature_dictionary.keys():
            feature_dictionary[feature_name] = []

        feature_dictionary[feature_name].append(descriptors)
        y_array.append(feature_label)
        # else:
        #     # feature_dictionary[feature_name] += descriptors
        #     feature_dictionary[feature_name] = np.append(feature_dictionary[feature_name], [descriptors])


# iterate dict keys, compute histograms for each key, store it in a code book

def dict2numpy(dict):
    nkeys = len(dict)
    array = np.zeros((nkeys * PRE_ALLOCATION_BUFFER, 128))
    pivot = 0
    for key in dict.keys():
        category_arr = dict[key]
        for sub_value in category_arr:
            value = sub_value
            nelements = value.shape[0]
            while pivot + nelements > array.shape[0]:
                padding = np.zeros_like(array)
                array = np.vstack((array, padding))
            array[pivot:(pivot + nelements)] = value
            pivot += nelements
        array = np.resize(array, (pivot, 128))
    return array

def computeHistograms(codebook, descriptors):
    code, dist = vq(descriptors, codebook)
    histogram_of_words, bin_edges = np.histogram(code,
                                              bins=range(codebook.shape[0] + 1),
                                              normed=True)
    return histogram_of_words


for i in range(9):
    print('Training at {0}'.format(i + 1))
    extract_sift_features_from_array(ten_fold_array[i], train_dictionary, Y_train)

extract_sift_features_from_array(ten_fold_array[9], test_dictionary, Y_test)
print('Done with test feature extraction')

pickle.dump(Y_train, open("pickles/y_train.p", "wb"))
pickle.dump(Y_test, open("pickles/y_test.p", "wb"))
pickle.dump(train_dictionary, open("x_train_dict.p", "wb"))
pickle.dump(test_dictionary, open("x_test_dict.p", "wb"))


all_features_array = dict2numpy(train_dictionary)
all_test_features_array = dict2numpy(test_dictionary)
nfeatures = all_features_array.shape[0]
nclusters = int(np.sqrt(nfeatures))
n_test_clusters = int(np.sqrt(all_test_features_array.shape[0]))
print('Extracting codebook')
codebook, _ = kmeans(all_features_array, nclusters, thresh=0.2)

print('Extracted codebook')

train_words_histograms = []
test_words_histograms = []
test = []

def write_to_histogram(category, histogram):
    for subarr in category:
        word_histogram = computeHistograms(codebook, subarr)
        histogram.append(word_histogram)


for key in train_dictionary.keys():
    write_to_histogram(train_dictionary[key], train_words_histograms)

for key in test_dictionary.keys():
    write_to_histogram(test_dictionary[key], test_words_histograms)


def modify_histogram(nwords, histogram_array):
    data_rows = np.zeros(nwords)  # add for label
    index = -1
    for histogram in histogram_array:
        # index += 1
        if histogram.shape[0] != nwords:
            nwords = histogram.shape[0]
            data_rows = np.zeros(nwords)
            print('nclusters reduced to {0}'.format(nwords))

        # data_row = np.hstack((y[index], histogram))
        data_rows = np.vstack((data_rows, histogram))

    return data_rows

print('Transforming data')
new_x_train = modify_histogram(nclusters, np.asarray(train_words_histograms))
new_x_test = modify_histogram(n_test_clusters, np.asarray(test_words_histograms))
pickle.dump(new_x_train, open("pickles/x_train.p", "wb"))
pickle.dump(new_x_test, open("pickles/x_test.p", "wb"))



clf = SGDClassifier( n_jobs=-1)
# clf.fit(new_x_train[1:], np.asarray(Y_train))
#
# accuracy = clf.score(new_x_test[1:], np.array(Y_test))
# print(accuracy)
print('Teaching classifiers')

#clf_two = GaussianNB()
clf.fit(new_x_train[1:], np.asarray(Y_train))
second_score = clf.score(new_x_test[1:], np.array(Y_test))
print(second_score)

test = 1

# x_train = dict2numpy()



"""Fill array"""


# def gen_sift_features(gray_img):
#     kp, desc = sift.detectAndCompute(gray_img, None)
#     return kp, desc


# def split_data_labels(current_folder_files, X_array, Y_array):
#     extract_count = 0
#     for picture in current_folder_files:
#         label = picture.split('/')[1].split('\\')[0]
#         image = cv2.imread(picture, cv2.COLOR_BGR2GRAY)
#         kp, descr = gen_sift_features(image)
#
#         if descr is None:
#             continue
#
#         if extract_count % 40 == 0:
#             print('Extracted {0} features, current label: {1}'.format(extract_count, label))
#
#         extract_count += 1
#
#         X_array.append(descr)
#         Y_array.append(label)

