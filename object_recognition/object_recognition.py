# import packages here
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
print(cv2.__version__) # verify OpenCV version


import skimage.exposure
from sklearn.utils import shuffle
import matplotlib.gridspec as gridspec

# load data
def loadData(file):
  # Implement your loadData(file) here
  image_count = 10
  image_format = ".png"
  img_list = []
  
  # Reading list of images
  for i in range(1,image_count+1):
    img_file = file + str(i) + image_format
#     img_list.append(cv2.cvtColor(cv2.imread(img_file, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    img_list.append(cv2.imread(img_file, 0))
  return img_list
  
# ===== Display your first graph here =====
# create a vector of labels
# assume labels: bird = 0, human = 1

x_images = loadData('SourceImages/human_vs_birds/bird_')
x_images = x_images + loadData('SourceImages/human_vs_birds/human_')
# Birds - 0, Humans - 1
y_labels = np.ones(20)
y_labels[:10] = 0
# Shuffle the data set
shuffle_x, shuffle_y = shuffle(x_images, y_labels)

# Displaying the data set
plt.figure(figsize=(18,15))
grid =  gridspec.GridSpec(3, 10)
index = 0
for i in range(2):
  for j in range(10):
    plt.subplot(grid[i,j])
    plt.imshow(shuffle_x[index], cmap='gray')
    plt.title(shuffle_y[index])
    index = index + 1
    plt.axis('off')

plt.show()

from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

# Compute HOG features for the images
def computeHOGfeatures(shuffle_x):

    # Implement your computeHOGfeatures() here
  # HOG using cv2
  # cell size in pixels (h x w)
  cell_size = (16, 16)
  block_size_param = (2, 2)  
  nbins = 9 # Recommended
  winSize = (shuffle_x[0].shape[1] // cell_size[1] * cell_size[1], shuffle_x[0].shape[0] // cell_size[0] * cell_size[0])
  blockSize = (block_size_param[1] * cell_size[1], block_size_param[0] * cell_size[0]) # twice of cell size
  blockStride = (blockSize[1]//2, blockSize[0]//2) # 50% of block size
  # default values
  derivAperture = 1
  winSigma = -1.
  histogramNormType = 0
  L2HysThreshold = 0.2
  gammaCorrection = 1
  nlevels = 64
  useSignedGradients = True
  hog_cv = cv2.HOGDescriptor(winSize,blockSize,blockStride,cell_size,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)
  
  # A list to store image and its respective descriptor information
  hog_feature_map = []
  for image in shuffle_x:
      # hog using sklearn library with visualize
      fd,hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(4, 4), transform_sqrt = True, visualize=True)
      # a list with [actual image, its feature descriptor from sklearn library, its hog image, descriptor obtained from OpenCV library]
      hog_feature_map.append([image, fd, hog_image, hog_cv.compute(image).ravel()])
  return hog_feature_map

# Compute HOG descriptors
hog_feature_map = computeHOGfeatures(shuffle_x)
# ===== Display second graph =====
grid = plt.GridSpec(3, 10, wspace=0.4)
plt.figure(figsize=(18,15))
index = 0
for i in range(2):
  for j in range(10):
    plt.subplot(grid[i,j])
    plt.imshow(hog_feature_map[index][2], cmap='gray')
    plt.title(shuffle_y[index])
    index = index + 1
    plt.axis('off')

plt.show()

# Split the data and labels into train and test set

# First 16 from the shuffled data set for training and rest 4 for testing
train_hog_feature_map = hog_feature_map[:16]
test_hog_feature_map = hog_feature_map[16:]

# Splitting labels
train_y = shuffle_y[:16]
test_y = shuffle_y[16:]

# image features for training
train_x = [x[1] for x in train_hog_feature_map]
test_x = [x[1] for x in test_hog_feature_map]
# normalize
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.fit_transform(test_x)

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,accuracy_score

# train model with SVM
# call LinearSVC
# train SVM
# call clf.predict

# Linear SVC
svc = LinearSVC(max_iter=5000)
# Training
svc.fit(train_x, train_y)
# Testing
accuracy = svc.score(test_x, test_y)
# Predictions
y_pred = svc.predict(test_x)


# plots for easier analysis
# grid = plt.GridSpec(1, len(test_x), wspace=0.4)
# plt.figure(figsize=(18,15))
# index = 0
# for j in range(len(test_x)):
#   plt.subplot(grid[0,j])
#   plt.imshow(test_hog_feature_map[index][0], cmap='gray')
#   plt.title(str(test_y[index]) + " pred: " + str(y_pred[index]))
#   index = index + 1
#   plt.axis('off')

# plt.show()

# ===== Output functions ======
print('estimated labels: ', test_y)
print('ground truth labels: ', y_pred)
print('Accuracy: ', str(accuracy_score(test_y, y_pred)*100), '%')