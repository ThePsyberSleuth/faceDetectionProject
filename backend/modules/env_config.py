import os

# Get the absolute path of the faceDetectionProject directory
# by navigating up one level from the current file's location
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Replace backslashes with forward slashes in the root directory path
ROOT_DIR = ROOT_DIR.replace("\\", "/")

# Define the absolute paths for various resources
CASCADE_CLASSIFIER_PATH = os.path.join(ROOT_DIR,
                                       "dataset/cascade_classifier/haarcascade_frontalface_default.xml")
CASCADE_CLASSIFIER_PATH = CASCADE_CLASSIFIER_PATH.replace("\\", "/")
#
DATABASE_PATH = os.path.join(ROOT_DIR, "database/vdbStore.db")
DATABASE_PATH = DATABASE_PATH.replace("\\", "/")
#
TRAINING_DATA_PATH = os.path.join(ROOT_DIR, "dataset/images")
TRAINING_DATA_PATH = TRAINING_DATA_PATH.replace("\\", "/")
#
TRAINED_MODEL_PATH = os.path.join(ROOT_DIR, "dataset/recognizer/")
TRAINED_MODEL_PATH = TRAINED_MODEL_PATH.replace("\\", "/")
#
CAMERA_CONFIG_FILE = os.path.join(ROOT_DIR, "cam-config/camera_config.json")
CAMERA_CONFIG_FILE = CAMERA_CONFIG_FILE.replace("\\", "/")

# paths for the face detection model and the camera configuration file.

# path: backend/modules/env_config.py
