# Midterm Report on Project 1: Celebrity Facial Recognition

**Date:** Feb 4th, 2024  
**Authors:** Sahith Jagarlamudi and Sydney Simon

## Introduction

This project is a comprehensive system for real-time face detection and recognition using Python, OpenCV, PyTorch, and the facenet_pytorch library. It processes video input to detect faces and, optionally, recognize specific individuals.

### Features

- **Face Detection:** Detects faces in video frames using a DNN model.
- **Facial Recognition:** Recognizes specific individuals from the detected faces (currently set up for "Robert Downey Jr.").
- **Video Input Flexibility:** Works with both video files and webcam streams.
- **Real-Time Processing:** Displays the processed video in real-time with detected faces and recognition results.
- **Video Saving Option:** Ability to save the processed video with annotations.

## Current Progress

- **Technical Aspect Progress:** We have created and trained our DNN model with a dataset consisting of Robert Downey Jr. photos in addition to other celebrities' faces. The model is trained to recognize Robert Downey Jr.
- **Challenges and Solutions:** There are occasional false negatives with the trained model. We plan on addressing this by increasing the size of the dataset and increasing the ratio of Robert Downey Jr. to non-Robert Downey Jr. photos.

## Preparation and Compilation of Dataset

- **Data Sourcing:** We have found a dataset of images of a large number of celebrities, including Robert Downey Jr. We have used these pictures for the rdj and not rdj datasets (we then split these sets into test and validation sets.). The size requirements are satisfied by the datasets. To increase the dataset size and accuracy levels (robustness of the model), we are working on a web scraper script to get large scale datasets of images of the required celebrity.
- **Data Processing:** We have worked through creating the model. We have successfully worked through the interactive ipython notebook. We see that there are occasional false negatives when the model is tested.
- **Tooling and Technologies:** We have not used any code past the given scripts so far on the model. We have tested our new script for image scraping though (also in Python and uses the urllib library).

## Division of Work

- **Role Assignments:** Worked together on all parts of the project. Equal split.
- **Collaboration and Communication:** Majority of the communication we have is over text. We have met for completing a major portion of the project together in person. We have a GitHub repository setup which we collaborate on.

## Conclusion and Next Steps

- **Conclusion:** Up until this point, we have gathered preliminary data composed of celebrity faces from different angles. Using this data, we used the interactive ipython notebook to train a DNN model.
- **Next Steps:** We will improve our dataset to decrease the number of false negatives from our model. In addition to improving the dataset, we will test the model on input videos and video capture.

## References

Data Source: [https://www.kaggle.com/datasets/hereisburak/pins-face-recognition](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition)
