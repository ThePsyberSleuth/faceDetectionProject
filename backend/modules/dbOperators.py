import logging
import os
import sqlite3
import uuid as uuid_module

from .env_config import DATABASE_PATH as db_location

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Custom exception for database-related errors
class DBError(Exception):
    pass


# Decorator to provide a database connection to the wrapped function
def with_connection(func):
    def inner(self, *args, **kwargs):
        with sqlite3.connect(self.db_path) as conn:
            return func(self, conn, *args, **kwargs)

    return inner


class DBOperator:
    def __init__(self, db_path=db_location):
        self.db_path = db_path
        # Ensure the directory for the database exists
        db_dir = os.path.dirname(db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            logging.info("Database directory created.")

    @with_connection
    def execute_query(self, conn, query, parameters=None, commit=False):
        """
        Execute a given SQL query with optional parameters and commit if specified.
        """
        try:
            cursor = conn.cursor()
            cursor.execute(query, parameters or [])
            if commit:
                conn.commit()
            return cursor
        except sqlite3.Error as e:
            logging.error(f"SQL error: {e} - Query: {query}")
            raise DBError(e)

    @with_connection
    def fetch_data(self, conn, query, parameters=None):
        """
        Fetch data from the database using a given SQL query and optional parameters.
        """
        try:
            cursor = conn.cursor()
            cursor.execute(query, parameters or [])
            return cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"SQL error: {e} - Query: {query}")
            raise DBError(e)

    def initialize_db(self):
        """
        Initialize the database with the required tables.
        """
        tables = [
            '''CREATE TABLE IF NOT EXISTS USERS (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                role TEXT NOT NULL
            )''',
            '''CREATE TABLE IF NOT EXISTS IMAGES (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES USERS(id)
            )''',
            '''CREATE TABLE IF NOT EXISTS USER_ACTIVITY (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                activity TEXT NOT NULL,
                date_time TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES USERS(id)
            )'''
        ]
        for table_query in tables:
            try:
                self.execute_query(table_query, commit=True)
            except DBError as e:
                logging.error(f"Failed to initialize table: {e}")
                return
        logging.info("Database initialized successfully.")

    def insert_or_update_user(self, user_id, user_name, user_age, user_role):
        """
        Insert a new user record or update an existing one without changing the UUID.
        """
        if not os.path.exists(db_location):
            self.initialize_db()

        # Check if the user already exists
        existing_user = self.fetch_data("SELECT * FROM USERS WHERE id=?", (user_id,))
        if existing_user:
            # Update existing user without changing the UUID
            query = "UPDATE USERS SET name=?, age=?, role=? WHERE id=?"
            try:
                self.execute_query(query, (user_name, user_age, user_role, user_id), commit=True)
                logging.info(f"User {user_id} updated successfully.")
                return user_id
            except DBError:
                return None
        else:
            # Insert new user with a new UUID
            unique_id = str(uuid_module.uuid4())
            query = "INSERT INTO USERS (id, uuid, name, age, role) VALUES (?, ?, ?, ?, ?)"
            try:
                self.execute_query(query, (user_id, unique_id, user_name, user_age, user_role), commit=True)
                logging.info(f"User {user_id} with UUID {unique_id} inserted successfully.")
                return user_id
            except DBError:
                return None

    def get_profile(self, user_id):
        """
        Retrieve a user profile based on the user ID.
        """
        try:
            profiles = self.fetch_data("SELECT * FROM USERS WHERE id=?", (user_id,))
            if profiles:
                logging.info(f"Profile retrieved for user {user_id}.")
                return profiles[0]  # Return the first profile if exists
            else:
                logging.info(f"No profile found for user {user_id}.")
                return None
        except sqlite3.Error as e:
            logging.error(f"Failed to retrieve profile for user {user_id}: {e}")
            return None

    @with_connection
    def insert_images(self, conn, user_id, image_paths):
        """
        Insert multiple new image records for a single user.
        """
        query = "INSERT INTO IMAGES (user_id, image_path) VALUES (?, ?)"
        try:
            for image_path in image_paths:
                self.execute_query(query, (user_id, image_path), commit=False)
            conn.commit()  # Commit all inserts at once
            logging.info(f"{len(image_paths)} images for user {user_id} inserted successfully.")
        except sqlite3.Error as e:
            conn.rollback()  # Roll back in case of error
            logging.error(f"Failed to insert images for user {user_id}: {e}")
            return False
        return True

    @with_connection
    def get_user_images(self, conn, user_uuid):
        """
        Retrieve all image paths and user IDs for a user based on the user's UUID.
        """
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT u.id, ui.image_path FROM IMAGES ui JOIN USERS u ON ui.user_id = u.uuid WHERE u.uuid=?",
                (user_uuid,)
            )
            images = cursor.fetchall()
            logging.info(f"Retrieved {len(images)} images for user {user_uuid}.")
            ids = [int(os.path.splitext(os.path.basename(image_path))[0].split('.')[0]) for _, image_path in images]
            image_paths = [image_path for _, image_path in images]
            return ids, image_paths
        except sqlite3.Error as e:
            logging.error(f"Failed to retrieve images for user {user_uuid}: {e}")
            return [], []

    @with_connection
    def get_user_activity(self, conn, user_id):
        """
        Retrieve a user's activity log based on the user ID.
        """
        try:
            activities = self.fetch_data(conn,
                                         "SELECT activity, date_time FROM USER_ACTIVITY WHERE user_id=? ORDER BY date_time DESC",
                                         (user_id,))
            logging.info(f"Retrieved {len(activities)} activities for user {user_id}.")
            return activities
        except sqlite3.Error as e:
            logging.error(f"Failed to retrieve activities for user {user_id}: {e}")
            return []

# path: backend/modules/dbOperators.py
