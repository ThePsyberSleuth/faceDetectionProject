import logging
import os

import cv2

from .dbOperators import DBOperator
from .dsTrainer import DSTrainer
from .env_config import CASCADE_CLASSIFIER_PATH as cc_path
from .env_config import DATABASE_PATH as db_path
from .env_config import TRAINING_DATA_PATH as dataset_location
from .osCamera import Camera

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DSCreator:
    def __init__(self, cascade_path=cc_path):
        """
        Initialize the DSCreator with paths to the cascade classifier, database, and dataset location.
        """
        self.face_detect = cv2.CascadeClassifier(cascade_path)
        self.db_operator = DBOperator(db_path)
        self.ds_trainer = DSTrainer()
        self.camera = Camera()

    def __enter__(self):
        """
        Enter the runtime context related to this object.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context related to this object, ensuring resources are cleaned up.
        """
        self.camera.release()
        cv2.destroyAllWindows()

    def insert_or_update_func(self, id, name, age, role):
        """
        Insert or update a user record in the database.
        """
        return self.db_operator.insert_or_update_user(id, name, age, role)

    def capture_and_process_faces(self, id, name, age, role):
        """
        Capture faces from the camera and process them.
        """
        self.insert_or_update_func(id, name, age, role)
        user_record = self.db_operator.fetch_data("SELECT id, uuid FROM USERS WHERE id=?", (id,))
        if not user_record:
            print(f"No user found with ID: {id}")
            return

        if user_record and len(user_record[0]) > 1:
            user_uuid = user_record[0][1]  # Retrieve the UUID from the second column
        else:
            print(f"Unable to retrieve UUID for user ID: {id}")
            return

        images_path = os.path.join(dataset_location, user_uuid)
        images_path = images_path.replace("\\", "/")
        os.makedirs(images_path, exist_ok=True)

        sample_num = 0
        capture_complete = False  # Flag to indicate when to stop capturing
        image_paths = []  # Initialize a list to collect image paths
        try:
            for mirrored_frame in self.camera.get_video_feed():  # Iterate over the frames from the generator
                if mirrored_frame is None:
                    print("Failed to grab frame")
                    break
                gray_image = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detect.detectMultiScale(gray_image, 1.3, 5, minSize=(40, 40))
                for (x, y, w, h) in faces:
                    sample_num += 1
                    face_image_path = os.path.join(images_path, f"{id}.{sample_num}.jpg")
                    face_image_path = face_image_path.replace("\\", "/")
                    cv2.imwrite(face_image_path, gray_image[y:y + h, x:x + w])
                    image_paths.append(face_image_path)  # Collect the path
                    cv2.rectangle(mirrored_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if sample_num >= 60:  # Check if the desired number of images has been captured
                        capture_complete = True
                        break  # Break out of the inner loop
                    cv2.waitKey(100)
                # Display the frame with detected face rectangles
                cv2.imshow("Create Face Dataset", mirrored_frame)
                # Break the loop if 'q' is pressed or 60 images have been captured
                if capture_complete or cv2.waitKey(1) & 0xFF == ord('q'):
                    break  # Break out of the outer loop
        except Exception as e:
            logging.error(f"An error occurred during face capture: {e}")
        finally:
            if image_paths:  # Ensure there are paths to insert
                self.db_operator.insert_images(user_uuid, image_paths)  # Insert all collected paths at once

            self.ds_trainer.train_recognizer(user_uuid)  # Train the recognizer
        logging.info("Dataset creation process completed.")

# path: backend/modules/dsCreator.py
