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

"""
Array used to store all the images for 10-fold cross validation
"""
ten_fold_array = [[] for i in repeat(None, 10)]
sub_folders_list = []


def iterate_class_folders(number_of_classes):
    main_dir = "101_ObjectCategories"
    # sub_folders_list = []
    all_folder_names = os.listdir(main_dir)
    temp_folders = ['Motorbikes', 'accordion', 'crocodile']
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

X_train = []
Y_train = []
X_test = []
Y_test = []
# sift = cv2.xfeatures2d.SIFT_create()
des_list = []

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

