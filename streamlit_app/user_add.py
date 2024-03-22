import argparse
import sqlite3
import os
import bcrypt 
"""Legacy script. Not supported anymore"""

def hash_password(password: str):
    bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(bytes, salt)

def create_db_and_table(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users(
            first_name TEXT,
            last_name TEXT,
            username TEXT,
            email TEXT,
            password TEXT,
            institution TEXT
        )
    ''')
    conn.commit()
    conn.close()

def create_db_and_admin_table(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins(
            first_name TEXT,
            last_name TEXT,
            username TEXT,
            email TEXT,
            password TEXT,
            institution TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_user_data(db_name: str, 
                     first_name: str, 
                     last_name: str, username: str, 
                     email: str, password: str, 
                     institution: str):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO users(first_name, last_name, username, email, password, institution) VALUES(?,?,?,?,?,?)
    ''', (first_name, last_name, username, email, password, institution))
    conn.commit()
    conn.close()

def update_user_password(db_name: str, username: str, password: str):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users SET password = ? WHERE username = ?
    ''', (password, username))
    conn.commit()
    conn.close()

parser = argparse.ArgumentParser(description='User Information')
parser.add_argument('--first_name', required=True, help='First name of the user')
parser.add_argument('--last_name', required=True, help='Last name of the user')
parser.add_argument('--username', required=True, help='Username of the user')
parser.add_argument('--db_name', required=True, help='Database name')
parser.add_argument('--email', required=True, help='Email of the user')
parser.add_argument('--password', required=True, help='Password of the user')
parser.add_argument('--institution', required=True, help='Institution of the user')


args = parser.parse_args()

db_name = args.db_name
first_name = args.first_name
last_name = args.last_name
username = args.username
email = args.email 
print (bcrypt.gensalt(12).decode('utf-8'))
password = args.password
print (password)
institution = args.institution


create_db_and_table(db_name)
insert_user_data(db_name, first_name, last_name, username, email, password, institution)