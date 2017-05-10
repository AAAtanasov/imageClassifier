# import the necessary packages
import cv2
import os
import os.path
import random
from itertools import repeat

# debug info OpenCV version
print("OpenCV version: " + cv2.__version__)

param = "accordion"


def retrieve_image_from_folder(param):
    # image path and valid extensions
    image_dir = "101_ObjectCategories/" + param  # specify your path here
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

    print(random.randrange(len(list_of_images)))

    def random_index_assign(incoming_list, count_of_elems_to_extract, global_ten_fold_array, arrIndex):
        items_to_add = extract_from_array(incoming_list, count_of_elems_to_extract)
        # global_ten_fold_array.append(items_to_add)
        global_ten_fold_array[arrIndex] = items_to_add
        a = global_ten_fold_array[arrIndex]
        return

    def extract_from_array(incoming_list, number_to_extract):
        result_items = []

        for tempIndex in range(int(number_to_extract)):
            choice_item_index = random.randrange(len(incoming_list))
            choice_item = incoming_list.pop(choice_item_index - 1)
            cv2.imread(choice_item)
            result_items.append(choice_item)

        return result_items

    """
    Array used to store all the images for 10-fold cross validation
    """
    ten_fold_array = [[] for i in repeat(None, 10)]
    print(type(ten_fold_array))

    uneven_elements_count = len(list_of_images) % 10
    rounded_elements_count = len(list_of_images) - uneven_elements_count
    ten_percent_extractable_integer = rounded_elements_count / 10

    print(uneven_elements_count)
    index = 0

    for elementIndex, value in enumerate(range(10)):
        print(elementIndex)
        if (index + 1) > uneven_elements_count:
            random_index_assign(list_of_images, ten_percent_extractable_integer,
                                ten_fold_array, elementIndex)
        else:
            random_index_assign(list_of_images, ten_percent_extractable_integer + 1,
                                ten_fold_array, elementIndex)
        index += 1

    # print(len(ten_fold_array))
    # for imageNode in ten_fold_array:
    #     for image in imageNode:
    #         temp_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    #         cv2.imshow(image, temp_image)
    #         cv2.waitKey(0)

    print(ten_fold_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


retrieve_image_from_folder("ant")
