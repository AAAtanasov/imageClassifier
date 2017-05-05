# import the necessary packages
import cv2
import os
import os.path
import random

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

listOfImages = []
# loop through image_path_list to open each image
for imagePath in image_path_list:
    image = cv2.imread(imagePath)

    # display the image on screen with imshow()
    # after checking that it loaded
    if image is not None:
        # cv2.imshow(imagePath, image)
        listOfImages.append(image)
    elif image is None:
        print("Error loading: " + imagePath)
        # end this loop iteration and move on to next image
        continue

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


def randomIndexAssign(incommingList, countOfElemsToExtract, globalTenFoldArray, arrIndex):
    items_to_add = extractFromArray(incommingList, countOfElemsToExtract)
    globalTenFoldArray[arrIndex] = items_to_add
    return


def extractFromArray(incommingList, numberToExtract):
    resultItems = []
    lengtOfIncommingList = len(incommingList)
    for tempIndex in range(int(numberToExtract)):
        choiceItemIndex = random.randrange(len(incommingList))
        choiceItem = incommingList.pop(choiceItemIndex - 1)
        resultItems.append(choiceItem)

    return resultItems


countOfAllImages = len(listOfImages)
print(countOfAllImages)

"""
Array used to store all the images for 10-fold cross validation
"""
tenFoldArray = []

unevenElementsCount = len(listOfImages) % 10
roundedElementCount = len(listOfImages) - unevenElementsCount
tenPercentExtractableInteger = roundedElementCount / 10

print(unevenElementsCount)
index = 0
for elementIndex in range(10):
    print(elementIndex)
    if (index + 1) < unevenElementsCount:
        randomIndexAssign(listOfImages, tenPercentExtractableInteger,
                          tenFoldArray, elementIndex)
    else:
        randomIndexAssign(listOfImages, tenPercentExtractableInteger +
                          1, tenFoldArray, elementIndex)
    index += 1



print(tenFoldArray)
cv2.waitKey(0)
cv2.destroyAllWindows()
