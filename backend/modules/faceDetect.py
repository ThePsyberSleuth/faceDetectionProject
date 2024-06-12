import logging
import cv2
import numpy as np
from .dbOperators import DBOperator
from .env_config import CASCADE_CLASSIFIER_PATH as cc_path
from .env_config import DATABASE_PATH as db_path
from .env_config import TRAINED_MODEL_PATH as model_path
from .osCamera import Camera

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaceDetector:
    def __init__(self, cascade_path=cc_path, sql_db_path=db_path, trained_model_path=model_path):
        """
        Initialize the FaceDetector with paths to the cascade classifier, database, and trained model.
        """
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.camera = Camera()
        self.db_operator = DBOperator(sql_db_path)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()  # Create a face recognizer object for LBPH algorithm
        self.trained_model_path = trained_model_path

    def get_profile(self, user_id):
        """
        Fetch user profile from the database using their ID.
        """
        try:
            return self.db_operator.get_profile(user_id)
        except Exception as e:
            logging.error(f"Error fetching profile from database: {e}")
            return None

    def detect_faces(self, user_id):
        """
        Detect faces in the video feed and display the user profile.
        """
        user_id_filepath = f'user{user_id}.faceModel.yml'
        self.recognizer.read(f'{self.trained_model_path}/{user_id_filepath}')  # Load the pre-trained recognizer model
        profile = self.get_profile(user_id)  # Fetch the profile for the user with the given ID

        if profile is None:
            print("User does not exist")
            return

        try:
            for frame in self.camera.get_video_feed():
                video_stream = frame
                gray = cv2.cvtColor(video_stream, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(40, 40))  # Detect faces in the frame

                for (x, y, w, h) in faces:
                    # Draw rectangle around the face
                    cv2.rectangle(video_stream, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    if profile:
                        identifier = profile[1]  # Get the UUID from the profile
                        label = profile[0]  # Get the integer label from the profile
                        if label is not None:
                            id, conf = self.recognizer.predict(gray[y:y + h, x:x + w])  # Predict the face
                            threshold = 100  # Set a threshold for confidence level
                            if conf > threshold:
                                text = "Error: Face does not match"
                                color = (0, 0, 255)
                            else:
                                # Prepare the text to display
                                text = f"ID: {identifier}\nName: {profile[2]}\nAge: {profile[3]}"
                                color = (0, 255, 127)
                        else:
                            text = "User does not exist"
                            color = (0, 0, 255)

                        # Draw text background
                        text_lines = text.split('\n')
                        text_height = 20 * len(text_lines)
                        overlay = video_stream.copy()
                        cv2.rectangle(overlay, (x, y - text_height - 10), (x + w, y), (0, 0, 0), -1)
                        alpha = 0.6  # Transparency factor
                        cv2.addWeighted(overlay, alpha, video_stream, 1 - alpha, 0, video_stream)

                        # Draw text
                        for i, line in enumerate(text_lines):
                            cv2.putText(video_stream, line, (x + 5, y - text_height + (i * 20) + 15),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)

                        # Draw confidence level at the bottom right
                        conf_text = f"Conf: {conf:.2f}"
                        (text_width, text_height), baseline = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
                        cv2.rectangle(video_stream, (video_stream.shape[1] - text_width - 10, video_stream.shape[0] - text_height - 10),
                                      (video_stream.shape[1], video_stream.shape[0]), (0, 0, 0), -1)
                        cv2.putText(video_stream, conf_text, (video_stream.shape[1] - text_width - 5, video_stream.shape[0] - 5),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Face Detector", video_stream)
                if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit
                    break

        except Exception as e:
            logging.error(f"Error occurred: {e}")

        finally:
            # Release resources
            self.camera.release()
            cv2.destroyAllWindows()

# path: backend/modules/faceDetect.py
