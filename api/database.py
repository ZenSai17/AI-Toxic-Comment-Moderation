import sqlite3


def create_db():
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS flagged_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            prediction TEXT NOT NULL,
            user_id TEXT NOT NULL
        )
    ''')
    connection.commit()
    connection.close()


def insert_flagged_message(message: str, prediction: str, user_id: str):
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()
    cursor.execute('''
        INSERT INTO flagged_messages (message, prediction, user_id) 
        VALUES (?, ?, ?)
    ''', (message, prediction, user_id))
    connection.commit()
    connection.close()


def get_flagged_messages():
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()
    cursor.execute('SELECT message, prediction, user_id FROM flagged_messages')
    rows = cursor.fetchall()
    connection.close()
    
    return [{"text": row[0], "prediction": row[1], "user_id": row[2]} for row in rows]