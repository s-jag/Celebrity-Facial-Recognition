import cv2
import os
import uuid
from facenet_pytorch import MTCNN

from face_det_dnn import *

def main():
    # Input video path.
    input_video = "videos/rdj.mp4"

    # Set video_input=True for local video input, False for webcam input.
    video_input = True
    # Set save_video=True to save result
    save_video = False

    # Load in video capture source.
    if video_input:
        cap = cv2.VideoCapture(input_video)
    else:
        cap = cv2.VideoCapture(0)

    # Check video loading validity.
    if not cap.isOpened():
        print("Error with camera or input video.")
        exit()
    fps = 5
    # cap.get(cv2.CAP_PROP_FPS)

    # Create video writer if save as video is True.
    if save_video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    # Setup face detection models.
    fddnn = FaceDetectionDNN()

    while cap.isOpened():
        # Capture frame by frame
        ret, frame = cap.read()
        if not ret:
            print("Reach the end of the video. Completed.")
            break

        # Display FPS
        cv2.putText(frame, f"FPS: {round(fps, 1)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Face detection using DNN model.
        frame, bounding_boxes = fddnn.detect_face_dnn(frame)
        frame = fddnn.print_num_faces(frame)

        print(f"bounding_box of face detection: {bounding_boxes}")
        print(len(bounding_boxes))
        print(f"----------")

        # Save images of detected faces.
        # Remember not to have the bounding box drawn around the detected face.
        if len(bounding_boxes) >= 1:
            for bounding_box in bounding_boxes:
                detected_face = frame[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]
                print(f'detected_face: {detected_face}')
                output_image_path = "saved_images"
                subject_name = "rdj"
                cv2.imwrite(os.path.join(output_image_path, '{}_{}.jpg'.format(subject_name, uuid.uuid1())), detected_face)
        
        # Display frame.
        cv2.imshow('Frame', frame)

        # Save frame.
        if save_video:
            out_writer.write(frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release capture
    cap.release()
    cv2.destroyAllWindows()
    if save_video:
        out_writer.release()

if __name__ == "__main__":
    main()