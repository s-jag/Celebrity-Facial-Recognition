# Face Detection Using Deep Neural Network and OpenCV

The code defines a class `FaceDetectionDNN` for detecting faces in an image using a DNN model with the OpenCV library.

## Class: FaceDetectionDNN

### Initializer: `__init__(self)`
- Initializes an instance of the `FaceDetectionDNN` class.
- Loads a pre-trained DNN model using `cv2.dnn.readNetFromCaffe`, specifying the paths to the Caffe model configuration file (`deploy.prototxt.txt`) and the trained model file (`res10_300x300_ssd_iter_140000.caffemodel`).
- Initializes a variable `self.num_faces` to store the number of detected faces.

### Method: `detect_face_dnn(self, frame)`
- Detects faces in a given image (`frame`).
- Initializes two lists: `face_bounding_box` to store the bounding box of the last detected face, and `faces_bounding_box` to store bounding boxes of all detected faces.
- Resets `self.num_faces` to zero.
- Extracts the height and width of the frame.
- Preprocesses the frame by resizing it to 300x300 pixels and converting it to a blob. This blob is then fed to the DNN model.
- Iterates through each detection:
  - Extracts the confidence score for each detection.
  - If the confidence score is greater than 0.5 (50%), it counts the face and computes the bounding box.
  - Draws a rectangle around the detected face and puts a text label showing the confidence score.
  - Appends the bounding box to `faces_bounding_box`.
- Returns the modified frame (with drawn rectangles and labels) and the list of bounding boxes.

### Method: `print_num_faces(self, frame)`
- Adds text to the given frame displaying the total number of faces detected (`self.num_faces`).
- Returns the modified frame with the number of faces displayed.

## Usage
- An instance of `FaceDetectionDNN` can be created and used to detect faces in images.
- The `detect_face_dnn` method is called with an image frame as input to perform face detection.
- The `print_num_faces` method can be called to add the count of detected faces on the frame.

## Dependencies
- OpenCV (`cv2`): A computer vision library used for image processing and DNN operations.
- NumPy (`np`): A library for numerical operations, used here for array manipulations.
