import argparse
import logging
import os
import shutil

import cv2

from backend.modules.dsCreator import DSCreator
from backend.modules.env_config import DATABASE_PATH, TRAINED_MODEL_PATH, TRAINING_DATA_PATH, CAMERA_CONFIG_FILE
from backend.modules.faceDetect import FaceDetector
from backend.modules.osCamera import setup_camera, CameraError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

VERSION = "1.3"

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
ENDC = "\033[0m"


def display_banner():
    banner = (
            r"""
    ░▒▓████████▓▒░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓████████▓▒░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓███████▓▒░░▒▓███████▓▒░  
    ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
    ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░     ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
    ░▒▓██████▓▒░░▒▓████████▓▒░▒▓█▓▒░      ░▒▓██████▓▒░░▒▓█▓▒▒▓███▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓███████▓▒░░▒▓█▓▒░░▒▓█▓▒░ 
    ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
    ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
    ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓████████▓▒░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░  
                                                                                                                           
    v"""
            + VERSION
            + """ / Fitzgerald Emmanuel Bowier / @krypt0_baby / https://github.com/krypt0-baby
"""
    )
    print(BLUE + banner + ENDC)


def reset_files():
    """
    Delete all user files including the database, recognizer models, images, and camera config.
    """
    paths_to_delete = [DATABASE_PATH, TRAINED_MODEL_PATH, TRAINING_DATA_PATH, CAMERA_CONFIG_FILE]
    for path in paths_to_delete:
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
                logging.info(f"{YELLOW}🗑️ Deleted file: {path}{ENDC}")
            elif os.path.isdir(path):
                shutil.rmtree(path)
                logging.info(f"{YELLOW}🗑️ Deleted directory: {path}{ENDC}")
    print(f"{YELLOW}🗑️ All user files have been reset.{ENDC}")


def interactive_menu():
    """
    Interactive menu for executing different parts of the program.
    """
    # display_banner()
    while True:
        print(f"{BLUE}<------------------------------>{ENDC}")
        print(f"{YELLOW}\nSelect an option:{ENDC}")
        print("1️⃣  Setup Camera")
        print("2️⃣  Create Dataset")
        print("3️⃣  Detect Faces")
        print("4️⃣  Reset All User Files")
        print("5️⃣  Exit")
        choice = input(f"{YELLOW}\nEnter your choice: {ENDC}")

        if choice == "1":
            try:
                print(f"{GREEN}📷 Setting up camera...{ENDC}")
                camera = setup_camera()
                print(f"{GREEN}📷 Camera setup successfully.{ENDC}")
            except CameraError as e:
                print(f"{RED}❌ Camera error: {e}{ENDC}")
            finally:
                if 'camera' in locals():
                    camera.release()
                cv2.destroyAllWindows()

        elif choice == "2":
            print(f"{GREEN}📸 Capture and process faces{ENDC}")
            print(f"{YELLOW}\nPlease enter the following details:{ENDC}")
            user_id = input(f"{RED}Enter User ID: {ENDC}")
            name = input(f"{RED}Enter User Name: {ENDC}")
            age = input(f"{RED}Enter User Age: {ENDC}")
            role = input(f"{RED}Enter User Role: {ENDC}")

            ds_creator = DSCreator()
            with ds_creator:
                ds_creator.capture_and_process_faces(user_id, name, age, role)

        elif choice == "3":
            print(f"{GREEN}🔍 Detecting faces{ENDC}")
            user_id = input(f"{RED}Enter User ID: {ENDC}")
            face_detector = FaceDetector()
            face_detector.detect_faces(user_id)

        elif choice == "4":
            print(f"{YELLOW}🗑️ Resetting all user files...{ENDC}")
            reset_files()

        elif choice == "5":
            print(f"{GREEN}👋 Exiting...{ENDC}")
            break

        else:
            print(f"{RED}❌ Invalid choice. Please try again.{ENDC}")


def main():
    display_banner()
    parser = argparse.ArgumentParser(description="Face Detection and Recognition System")

    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Sub-command for setting up the camera
    parser_camera = subparsers.add_parser("setup_camera", help="Setup the camera")

    # Sub-command for creating a dataset
    parser_create = subparsers.add_parser("create_dataset", help="Create a dataset")

    # Sub-command for detecting faces
    parser_detect = subparsers.add_parser("detect_faces", help="Detect faces")
    parser_detect.add_argument("--user_id", required=True, help="User ID")

    # Sub-command for resetting all user files
    parser_reset = subparsers.add_parser("reset", help="Reset all user files")

    # Interactive mode
    parser_interactive = subparsers.add_parser("interactive", help="Interactive mode")

    args = parser.parse_args()

    if args.command == "setup_camera":
        try:
            print(f"{GREEN}📷 Setting up camera...{ENDC}")
            camera = setup_camera()
            print(f"{GREEN}📷 Camera setup successfully.{ENDC}")
        except CameraError as e:
            print(f"{RED}❌ Camera error: {e}{ENDC}")
        finally:
            if 'camera' in locals():
                camera.release()
            cv2.destroyAllWindows()

    elif args.command == "create_dataset":
        print(f"{GREEN}📸 Capture and process faces{ENDC}")
        print(f"{YELLOW}\nPlease enter the following details:{ENDC}")
        user_id = input(f"{RED}Enter User ID: {ENDC}")
        name = input(f"{RED}Enter User Name: {ENDC}")
        age = input(f"{RED}Enter User Age: {ENDC}")
        role = input(f"{RED}Enter User Role: {ENDC}")

        ds_creator = DSCreator()
        with ds_creator:
            ds_creator.capture_and_process_faces(user_id, name, age, role)

    elif args.command == "detect_faces":
        print(f"{GREEN}🔍 Detecting faces{ENDC}")
        face_detector = FaceDetector()
        face_detector.detect_faces(args.user_id)

    elif args.command == "reset":
        print(f"{YELLOW}🗑️ Resetting all user files...{ENDC}")
        reset_files()

    elif args.command == "interactive":
        print(f"{GREEN}     🚀 Interactive mode      {ENDC}")
        interactive_menu()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
