# import the necessary packages
import cv2
import numpy as np
import os.path
import random
import math
import matplotlib.pyplot as plt
from itertools import repeat

"""
Array used to store all the images for 10-fold cross validation
"""
ten_fold_array = [[] for i in repeat(None, 10)]


def iterate_class_folders(number_of_classes):
    main_dir = "101_ObjectCategories"
    sub_folders_list = []
    all_folder_names = os.listdir(main_dir)
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


iterate_class_folders(5)
print(ten_fold_array)
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
hog_descriptor = cv2.HOGDescriptor()


def split_data_labels(current_folder_files, X_array, Y_array):
    for picture in current_folder_files:
        label = picture.split('/')[1].split('\\')[0]
        image = cv2.imread(picture)
        current_hog_feature = hog_descriptor.compute(image)
        X_array.append(current_hog_feature)
        Y_array.append(label)

for tempIndex, _ in enumerate(range(9)):
    currentFolderFiles = ten_fold_array[tempIndex]
    split_data_labels(currentFolderFiles, X_train, Y_train)


split_data_labels(ten_fold_array[9], X_test, Y_test)

# KNN = knn_classifier(n_neighbors=9)
# temp_train = X_train[:9]
# temp_label = Y_train[:9]

# KNN.fit(temp_train, temp_label)
# confidence = KNN.score(X_test, Y_test)
# print(confidence)

# temp_image = cv2.imread(ten_fold_array[0][0])
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
