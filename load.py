# import the necessary packages
import cv2
import os
import os.path
import random
from itertools import repeat

"""
Array used to store all the images for 10-fold cross validation
"""
ten_fold_array = [[] for i in repeat(None, 10)]


def iterate_class_folders(number_of_classes):
    main_dir = "101_ObjectCategories"
    sub_folders_list = []
    all_folder_names = os.listdir(main_dir)
    all_folder_names.remove("BACKGROUND_Google")

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
        global_ten_fold_array[array_index].extend(items_to_add)
        return

    def extract_from_array(incoming_list, number_to_extract):
        result_items = []

        for tempIndex, _ in enumerate(range(int(number_to_extract))):
            choice_item_index = random.randrange(len(incoming_list))
            choice_item = incoming_list.pop(choice_item_index - 1)
            result_items.append(choice_item)

        return result_items

    uneven_elements_count = len(list_of_images) % 10
    rounded_elements_count = len(list_of_images) - uneven_elements_count
    ten_percent_extractable_integer = rounded_elements_count / 10

    index = 0

    for elementIndex, value in enumerate(range(10)):
        if (index + 1) > uneven_elements_count:
            random_index_assign(list_of_images, ten_percent_extractable_integer,
                                ten_fold_array, elementIndex)
        else:
            random_index_assign(list_of_images, ten_percent_extractable_integer + 1,
                                ten_fold_array, elementIndex)
        index += 1


iterate_class_folders(100)
print(ten_fold_array)
temp_image = cv2.imread(ten_fold_array[0][0])
cv2.imshow("temp", temp_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
