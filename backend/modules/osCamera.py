import json
import logging
import os

import cv2

from .env_config import CAMERA_CONFIG_FILE as CONFIG_FILE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CameraError(Exception):
    """Custom exception for camera-related errors."""
    pass


class Camera:
    def __init__(self, camera_id=None):
        self.camera_id = camera_id if camera_id is not None else self.load_camera_id()
        self.camera = None
        try:
            self.initialize_camera()
        except CameraError as e:
            logging.error(f"Failed to initialize camera: {e}")

    def initialize_camera(self):
        """
        Initializes the camera resource.
        """
        try:
            self.camera = cv2.VideoCapture(self.camera_id)
            if not self.camera.isOpened():
                raise CameraError(f"Camera with ID {self.camera_id} could not be opened.")
            logging.info(f"Camera with ID {self.camera_id} initialized.")
        except cv2.error as e:
            raise CameraError(f"OpenCV error: {e}")

    def get_video_feed(self):
        """
        Generator function that yields mirrored frames for live video feed.
        """
        if not self.camera:
            self.initialize_camera()
        while True:
            success, i_frame = self.camera.read()
            if not success:
                logging.error("Failed to read frame from camera.")
                break
            # Mirror the frame
            mirrored_frame = cv2.flip(i_frame, 1)
            yield mirrored_frame
            # yield i_frame

    def release(self):
        """
        Releases the camera resource.
        """
        if self.camera:
            try:
                self.camera.release()
                logging.info(f"Camera with ID {self.camera_id} released.")
            except cv2.error as e:
                logging.error(f"OpenCV error while releasing camera: {e}")
            finally:
                self.camera = None

    def __del__(self):
        """
        Ensures the camera resource is released when the object is destroyed.
        """
        self.release()

    def save_camera_id(self):
        """
        Save the selected camera ID to a configuration file.
        """
        config_dir = os.path.dirname(CONFIG_FILE)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        with open(CONFIG_FILE, 'w') as f:
            json.dump({"camera_id": self.camera_id}, f)

    def load_camera_id(self):
        """
        Load the camera ID from a configuration file.
        """
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config.get("camera_id", 0)
        return 0


def list_available_cameras(max_cameras=10, max_failures=5):
    """
    List available cameras.
    """
    available_cameras = []
    consecutive_failures = 0
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
            consecutive_failures = 0  # Reset failure count on success
        else:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                break
    return available_cameras


def prompt_user_for_camera_selection(available_cameras):
    """
    Prompt the user to select a camera.
    """
    print("Available cameras:")
    for idx in available_cameras:
        print(f"{idx}: Camera {idx}")
    selected_camera = int(input("Select a camera by entering the corresponding number: "))
    return selected_camera


def setup_camera():
    """
    Setup the camera by prompting the user to select one.
    """
    available_cameras = list_available_cameras()
    if not available_cameras:
        raise CameraError("No cameras found.")
    selected_camera = prompt_user_for_camera_selection(available_cameras)
    camera = Camera(camera_id=selected_camera)
    camera.save_camera_id()
    return camera

# path: backend/modules/osCamera.py
