
---

# FaceGuard: Face Detection and Recognition System 🛡️

```
    ░▒▓████████▓▒░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓████████▓▒░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓███████▓▒░░▒▓███████▓▒░  
    ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
    ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░     ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
    ░▒▓██████▓▒░░▒▓████████▓▒░▒▓█▓▒░      ░▒▓██████▓▒░░▒▓█▓▒▒▓███▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓███████▓▒░░▒▓█▓▒░░▒▓█▓▒░ 
    ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
    ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
    ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓████████▓▒░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░  
                                                                                                                           
    v1.3 / Fitzgerald Emmanuel Bowier / @krypt0_baby / https://github.com/krypt0-baby
```

## Overview

**FaceGuard** is a face detection and recognition system that allows users to set up a camera, create a dataset, train a recognizer, and detect faces. The system can be controlled via command-line flags or an interactive menu.

## Directory Structure

```
faceDetectionProject/
├── backend/
│   ├── __init__.py
│   ├── cam-config/
│   │   ├── camera_config.json
│   ├── database/
│   │   ├── vdbStore.db
│   ├── dataset/
│   │   ├── cascade_classifier/
│   │   │   ├── haarcascade_frontalface_default.xml
│   │   ├── images/
│   │   ├── recognizer/
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── dsCreator.py
│   │   ├── faceDetect.py
│   │   ├── osCamera.py
│   │   ├── env_config.py
├── main.py
```

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/krypt0-baby/faceDetectionProject.git
   cd faceDetectionProject
   ```

2. **Set Up a Virtual Environment:**
   ```sh
   python -m venv FaceGuard
   ```

3. **Activate the Virtual Environment:**

   - **On Windows:**
     ```sh
     FaceGuard\Scripts\activate
     ```

   - **On macOS and Linux:**
     ```sh
     source FaceGuard/bin/activate
     ```

4. **Install the required packages:**
   ```sh
   pip install -r requirements.txt
   ```

5. **Ensure the directory structure is correct and `__init__.py` files are present in the `backend` and `modules` directories.**

## Usage

### Command-Line Interface

The system can be controlled using various command-line flags.

1. **Setup Camera:**
   ```sh
   python main.py setup_camera
   ```

2. **Create Dataset:**
   ```sh
   python main.py create_dataset
   ```
   The program will prompt you to enter the user details interactively.

3. **Detect Faces:**
   ```sh
   python main.py detect_faces --user_id 1
   ```

4. **Reset All User Files:**
   ```sh
   python main.py reset
   ```

5. **Interactive Mode:**
   ```sh
   python main.py interactive
   ```
   This will enter the interactive menu mode where you can navigate through different options and execute the corresponding functions interactively.

### Interactive Menu

The interactive menu provides a user-friendly interface for executing different parts of the program.

1. **Setup Camera:**
   - Select option `1` from the interactive menu.
   - Follow the prompts to set up the camera.

2. **Create Dataset:**
   - Select option `2` from the interactive menu.
   - Follow the prompts to enter user details and create the dataset.

3. **Detect Faces:**
   - Select option `3` from the interactive menu.
   - Follow the prompts to enter the user ID and detect faces.

4. **Reset All User Files:**
   - Select option `4` from the interactive menu.
   - This will delete all user files, including the database, recognizer models, images, and camera config.

5. **Exit:**
   - Select option `5` from the interactive menu to exit the program.

## Importance

FaceGuard provides a robust and user-friendly solution for face detection and recognition. It can be used in various applications such as security systems, attendance tracking, and personalized user experiences. The interactive menu and command-line interface make it accessible for both novice and advanced users.

---