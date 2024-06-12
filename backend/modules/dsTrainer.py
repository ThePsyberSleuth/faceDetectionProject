import logging
import os

import cv2
import numpy as np
from PIL import Image

from .dbOperators import DBOperator
from .env_config import CASCADE_CLASSIFIER_PATH as cc_path
from .env_config import TRAINED_MODEL_PATH as model_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DSTrainer:
    def __init__(self):
        """
        Initialize the DSTrainer with a face recognizer and database operator.
        """
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.db_operator = DBOperator()
        self.face_cascade = cv2.CascadeClassifier(cc_path)

    def train_recognizer(self, user_uuid):
        """
        Train the face recognizer using images associated with the given user UUID.
        """
        faces = []
        training_ids = []  # Separate variable for fetched IDs
        user_id = None  # Variable to store the user ID

        try:
            ids, image_paths = self.db_operator.get_user_images(user_uuid)
            logging.info(f"Fetched {len(ids)} IDs and {len(image_paths)} image paths for user UUID: {user_uuid}")
            if ids:
                user_id = ids[0]  # Assuming all images belong to the same user, get the first ID
            for user_id, image_path in zip(ids, image_paths):
                try:
                    face_img = Image.open(image_path).convert('L')  # Convert to grayscale
                    face_np = np.array(face_img, 'uint8')
                    detected_faces = self.face_cascade.detectMultiScale(face_np)  # Detect faces in the image
                    for (x, y, w, h) in detected_faces:
                        faces.append(face_np[y:y + h, x:x + w])
                        training_ids.append(user_id)  # Assign IDs to separate variable
                except Exception as e:
                    logging.error(f"Error processing image {image_path}: {e}")
        except Exception as e:
            logging.error(f"Error fetching user images from database: {e}")

        if len(training_ids) > 0 and len(faces) > 0:  # Use the separate variable for ID check
            try:
                self.recognizer.train(faces, np.array(training_ids))  # Use the separate variable for IDs
                if user_id is not None:
                    tmodel_path = os.path.join(model_path, f"user{user_id}.faceModel.yml")  # Use the retrieved user ID
                    tmodel_path = tmodel_path.replace("\\", "/")  # Replace backslashes
                    os.makedirs(os.path.dirname(tmodel_path), exist_ok=True)  # Create the directory if it doesn't exist
                    self.recognizer.save(tmodel_path)  # Save the model with the retrieved user ID
                    logging.info(f"Model trained successfully and saved at {tmodel_path}")  # Log the success message
                else:
                    logging.error("User ID could not be determined.")
            except Exception as e:
                logging.error(f"Error training recognizer: {e}")
        else:
            logging.info("No data available for training.")

# Path: backend/modules/dsTrainer.py
