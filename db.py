import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash, check_password_hash  # For hashing passwords
from config import db_config


def get_db_connection():
    connection = None
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            print("Successfully connected to the database")
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
    return connection


def signup_user(username, email, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Hash the password before inserting into the database
        hashed_password = generate_password_hash(password)

        query = "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)"
        cursor.execute(query, (username, email, hashed_password))
        conn.commit()
        print("User signed up successfully")
        return cursor.lastrowid
    except mysql.connector.Error as err:
        print("Error:", err)
        return None
    finally:
        cursor.close()
        conn.close()


def login_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = "SELECT * FROM users WHERE username = %s"
        cursor.execute(query, (username,))
        user = cursor.fetchone()

        # Check if user exists and the password is correct
        if user and check_password_hash(user['password'], password):
            print("Login successful")
            return user
        else:
            print("Invalid username or password")
            return None
    except mysql.connector.Error as err:
        print("Error:", err)
        return None
    finally:
        cursor.close()
        conn.close()