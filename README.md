# Celebrity-Facial-Recognition
ENGR 2900 Project 1 - Celebrity Facial Recognition

### Introduction
This project involves implementing a pipeline to detect and recognize a celebrity's identity by their facial features. The project utilizes face detection and facial recognition algorithms to identify celebrities in both images and videos.

### Project Structure
The project includes several components:
1. Face detection using DNN face detector from OpenCV.
2. Dataset preparation for training the facial recognition model.
3. Training a convolutional neural network using PyTorch for facial recognition.
4. Implementing the trained model to recognize celebrities in videos and webcam streams.

### Getting Started
#### Prerequisites
- Python 3.x
- Virtual Environment (venv or similar)
- IDE (VS Code or PyCharm recommended)
- CUDA enabled GPU (optional, for PyTorch)

#### Installation
1. Clone the repository to your local machine.
2. Set up a Python virtual environment:

python -m venv venv

3. Activate the virtual environment:
- Windows: `.\venv\Scripts\activate`
- Unix or MacOS: `source venv/bin/activate`
4. Install the required libraries:

pip install -r requirements.txt

5. For CUDA enabled PyTorch, follow the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

### Usage
1. Run the face detection algorithm:
- On a video file: `python video_face_detect.py --input <path_to_video>`
- On webcam stream: `python webcam_face_detect.py`
2. Train the facial recognition model using the provided notebook `ENM-2900-Project-1-Train-Model.ipynb`.
3. Place the trained model in the `models` directory.
4. Run the facial recognition on the detected faces:



### Project Files Description
- `video_face_detect.py`: Script for face detection in video files.
- `webcam_face_detect.py`: Script for face detection using webcam.
- `ENM-2900-Project-1-Train-Model.ipynb`: Jupyter notebook for training the model.
- `main_video.py`: Main script for running the facial recognition model.
- `requirements.txt`: List of Python dependencies.

### Acknowledgments
- OpenCV contributors
- PyTorch community
- Course instructors and TAs
