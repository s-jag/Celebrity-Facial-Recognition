# Celebrity-Facial-Recognition
ENGR 2900 Project 1 - Celebrity Facial Recognition

# Real-Time Face Detection and Recognition System

This project is a comprehensive system for real-time face detection and recognition using Python, OpenCV, PyTorch, and the facenet_pytorch library. It processes video input to detect faces and, optionally, recognize specific individuals.

## Project Components

The project consists of three main components:

1. **Face Detection Using Deep Neural Network (DNN)**
   - File: `face_det_dnn.py`
   - Description: Implements face detection using a DNN model with OpenCV.

2. **Facial Recognition**
   - File: `facial_rec.py`
   - Description: Performs facial recognition using a pre-trained InceptionResnetV1 model from facenet_pytorch.

3. **Main Video Processing Script**
   - File: `main.py`
   - Description: Integrates the face detection and facial recognition functionalities to process video streams from a file or webcam.

## Features

- **Face Detection**: Detects faces in video frames using a DNN model.
- **Facial Recognition** (optional): Recognizes specific individuals from the detected faces (currently set up for "angelina").
- **Video Input Flexibility**: Works with both video files and webcam streams.
- **Real-Time Processing**: Displays the processed video in real-time with detected faces and recognition results.
- **Video Saving Option**: Ability to save the processed video with annotations.

## Usage

1. **Set Up Your Environment**
   - Ensure Python 3.x is installed.
   - Install necessary libraries: `opencv-python`, `torch`, `torchvision`, `PIL`, `facenet_pytorch`.
   - Download necessary model files for face detection and recognition.

2. **Run the Main Script**
   - Execute `main.py` to start the video processing.
   - Use the `video_input` flag in `main.py` to switch between webcam and video file input.
   - Set `save_video` to `True` in `main.py` if you wish to save the output.

3. **Face Detection and Recognition**
   - The system will automatically detect faces in the video stream.
   - Uncomment the facial recognition section in `main.py` to enable recognition.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- PyTorch (`torch`)
- PIL (Python Imaging Library)
- facenet_pytorch

## Limitations

- The facial recognition component is configured to recognize specific individuals and requires a pre-trained model.
- Real-time processing speed depends on the hardware capabilities, especially when using high-resolution video input.

## Future Enhancements

- Expand the facial recognition database to recognize more individuals.
- Improve the processing speed and accuracy of the system.
- Implement additional features like emotion detection or age estimation.

This system demonstrates a practical application of computer vision and deep learning for real-time face detection and recognition in video streams.
