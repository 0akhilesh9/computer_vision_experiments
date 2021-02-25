Object Recognition with HOG features

# pip install the OpenCV version from 'contrib'
!pip install opencv-contrib-python==3.4.2.17

Approach:

Load in the images and create a vector of corresponding labels (0 for bird and 1 for human).
Extract HoG features from images using`cv2.HOGDescriptor` or hog routine from 'scikit-image'.
Reshape the HoG feature matrix and feed into the SVM.


