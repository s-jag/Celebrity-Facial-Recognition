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


## Class: FacialRecognition

### Initializer: `__init__(self)`
- Initializes an instance of `FacialRecognition` class.
- Sets `self.num_faces_recognized` to zero, which tracks the number of recognized faces.

### Method: `setup_model(self)`
- Sets up the facial recognition model.
- Determines whether to use a GPU (`cuda:0`) if available, otherwise uses CPU.
- Loads a pre-trained InceptionResnetV1 model from `facenet_pytorch`, specifically trained on the 'vggface2' dataset.
- Disables training for all layers of the model (`param.requires_grad = False`).
- Moves the model to the appropriate device (GPU or CPU).
- Loads a trained model state from a file (`trained_model.pt`).
- Defines image transformation steps including resizing, converting to tensor, and normalizing.
- Returns the configured model and the transformation pipeline.

### Method: `recognize_face(self, model, transform, frame, faces_bounding_box)`
- Performs facial recognition on a given frame.
- Uses the provided model and transformation pipeline.
- Iterates over detected faces (given by `faces_bounding_box`):
  - Extracts and preprocesses each face from the frame.
  - Applies the transformation to the face and predicts using the model.
  - Increments `self.num_faces_recognized` if the model recognizes a specific individual (e.g., "angelina").
  - Annotates the frame with the recognition result.
- Returns the annotated frame and the last recognition result.

### Method: `print_num_faces_recognized(self, frame)`
- Adds text to the frame displaying the number of faces recognized.
- Returns the modified frame.

## Main Execution
- If the script is the main program, it creates an instance of `FacialRecognition` and sets up the model.

## Dependencies
- `cv2`: OpenCV library for image processing.
- `torch` and `torch.nn`: PyTorch library for deep learning models.
- `PIL`: Python Imaging Library for image manipulation.
- `numpy`: Library for numerical operations.
- `torchvision`: A part of PyTorch for computer vision tasks.
- `facenet_pytorch`: A PyTorch implementation of the Inception Resnet V1 model for face recognition.
- `ssl`: SSL library for handling secure connections, here used to bypass SSL verification for simplicity.

## Notes
- The script assumes that the model file `trained_model.pt` and the facial recognition model (`InceptionResnetV1`) are specifically trained or fine-tuned for recognizing certain individuals.
- It is designed to recognize whether a detected face matches a specific individual (e.g., "angelina") or not.
- The script uses a threshold (not explicitly shown) to decide whether the model's prediction is confident enough.
