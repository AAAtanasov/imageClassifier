# import the necessary packages
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

"""
Array used to store all the images for 10-fold cross validation
"""
ten_fold_array = [[] for i in repeat(None, 10)]


def iterate_class_folders(number_of_classes):
    main_dir = "101_ObjectCategories"
    sub_folders_list = []
    all_folder_names = os.listdir(main_dir)
    temp_folders = ['ant', 'accordion']
    all_folder_names = temp_folders
    # all_folder_names.remove("BACKGROUND_Google")

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
                              ".tif", ".tiff"]  # specify your vald extensions here
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
        return

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


iterate_class_folders(2)
# print(ten_fold_array)
# temp_image = cv2.imread(ten_fold_array[0][0])

"""Corner detection"""


def corner_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    cv2.goodFeaturesToTrack()
    corners = np.int0(corners)

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image, (x, y), 4, 255, -1)

    cv2.imshow("temp", image)


def edge_detection(image):
    # edges = cv2.Laplacian(image, cv2.CV_64F)    #cv2.Canny(image, 100, 100)
    edges = cv2.Canny(image, 100, 100)

    cv2.imshow("Canny", edges)


def brute_force(image1, image2):
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    image3 = cv2.drawMatches(image1, kp1, image2, kp2, matches[:20], None, flags=2)
    plt.imshow(image3)
    plt.show()

X_train = []
Y_train = []
X_test = []
Y_test = []
sift = cv2.xfeatures2d.SIFT_create()
des_list = []

"""Fill array"""
def gen_sift_features(gray_img):
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc


def split_data_labels(current_folder_files, X_array, Y_array):
    for picture in current_folder_files:
        label = picture.split('/')[1].split('\\')[0]
        image = cv2.imread(picture, cv2.COLOR_BGR2GRAY)
        kp, descr = gen_sift_features(image)

        if descr is None:
            continue

        # if label != current_label:
        #     print('Extracted features from {0}, extracting {1}'.format(current_label, label))
        #     current_label = label

        X_array.append(descr)
        Y_array.append(label)

for tempIndex, _ in enumerate(range(9)):
    currentFolderFiles = ten_fold_array[tempIndex]
    split_data_labels(currentFolderFiles, X_train, Y_train)


split_data_labels(ten_fold_array[9], X_test, Y_test)

"""Transform learning features into usable"""
descriptors_list = X_train[0]
traincount = 0
for element in X_train[1:]:
    if traincount % 50 == 0:
        print('Trained {0} examples, current label: {1}'.format(traincount, Y_train[traincount]))
    descriptors_list = np.vstack((descriptors_list, element))
    traincount += 1

k = 100
voc, variance = kmeans(descriptors_list, k, 1)

im_features = np.zeros((len(Y_train), k), "float32")
for i in range(len(Y_train)):
    words, distance = vq(X_train[i], voc)
    for w in words:
        im_features[i][w] += 1


nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
idf = np.array(np.log((1.0*len(Y_train)+1) / (1.0*nbr_occurences + 1)), 'float32')
print(nbr_occurences)

stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

clf = KNeighborsClassifier(n_neighbors=9)
clf.fit(im_features, np.array(Y_train))
print('fitted')

"""Transform test features into usable"""
descriptor_test = X_test[0]
for element in X_test[1:]:
    descriptor_test = np.vstack((descriptor_test, element))



voctest, variancetest = kmeans(descriptor_test, k, 1)


test_im_features = np.zeros((len(Y_test), k), "float32")
for i in range(len(Y_test)):
    words, distance = vq(X_test[i], voctest)
    for w in words:
        test_im_features[i][w] += 1

# Missing nbr occurances and idf

stdSlr = StandardScaler().fit(test_im_features)
test_im_features = stdSlr.transform(test_im_features)
print('transformed test')

accuracy = clf.score(test_im_features, np.array(Y_test))
predict = clf.predict(test_im_features)

print('Progress')
print(accuracy)

matrix = confusion_matrix(Y_test, predict)
print(matrix)

# temp_train = X_train[:9]
# temp_label = Y_train[:9]

# KNN.fit(temp_train, temp_label)
# confidence = KNN.score(X_test, Y_test)
# print(confidence)


# def to_gray(color_img):
#     gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
#     return gray
#
#
# def show_sift_features(gray_img, color_img, kp):
#     return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))
#
# temp_image = cv2.imread(ten_fold_array[0][0])
# temp_image_gray = to_gray(temp_image)
# temp_kp, temp_desc = gen_sift_features(temp_image_gray)
# show_sift_features(temp_image_gray, temp_image, temp_kp);

# temp_image1 = cv2.imread(ten_fold_array[0][1])
# temp_image1 = cv2.imread(ten_fold_array[0][1])
# temp_image2 = cv2.imread(ten_fold_array[0][2])
# temp_image3 = cv2.imread(ten_fold_array[0][3])

# edge_detection(temp_image)
# edge_detection(temp_image1)
# corner_detection(temp_image)

# brute_force(temp_image, temp_image1)

cv2.waitKey(0)
cv2.destroyAllWindows()
