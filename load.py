# import the necessary packages
import cv2
import os
import os.path
import random
from itertools import repeat

# debug info OpenCV version
print("OpenCV version: " + cv2.__version__)

param = "accordion"


# image path and valid extensions
imageDir = "101_ObjectCategories/" + param  # specify your path here
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png",
                          ".tif", ".tiff"]  # specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]

# create a list all files in directory and
# append files with a vaild extention to image_path_list
for file in os.listdir(imageDir):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDir, file))

#   listOfImages = []
listOfImages = image_path_list

# loop through image_path_list to open each image
# for imagePath in image_path_list:
#     image = cv2.imread(imagePath)
# 
#     # display the image on screen with imshow()
#     # after checking that it loaded
#     if image is not None:
#         # cv2.imshow(imagePath, image)
#         listOfImages.append(image)
#     elif image is None:
#         print("Error loading: " + imagePath)
#         # end this loop iteration and move on to next image
#         continue

    # wait time in milliseconds
    # this is required to show the image
    # 0 = wait indefinitely
    # exit when escape key is pressed
    # key = cv2.waitKey(0)
    # if key == 27:  # escape
        # break

# close any open windows
#
#
print(random.randrange(len(listOfImages)))


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


countOfAllImages = len(listOfImages)
print(countOfAllImages)

"""
Array used to store all the images for 10-fold cross validation
"""
ten_fold_array = [[] for i in repeat(None, 10)]
print(type(ten_fold_array))

unevenElementsCount = len(listOfImages) % 10
roundedElementCount = len(listOfImages) - unevenElementsCount
tenPercentExtractableInteger = roundedElementCount / 10

print(unevenElementsCount)
index = 0

for elementIndex, value in enumerate(range(10)):
    print(elementIndex)
    if (index + 1) > unevenElementsCount:
        random_index_assign(listOfImages, tenPercentExtractableInteger,
                          ten_fold_array, elementIndex)
    else:
        random_index_assign(listOfImages, tenPercentExtractableInteger + 1,
                           ten_fold_array, elementIndex)
    index += 1


print(len(ten_fold_array))
for imageNode in ten_fold_array:
    for image in imageNode:
        temp_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        cv2.imshow(image, temp_image)
        cv2.waitKey(0)

print(ten_fold_array)
cv2.waitKey(0)
cv2.destroyAllWindows()
