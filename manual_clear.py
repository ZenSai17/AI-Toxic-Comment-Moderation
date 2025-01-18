import sqlite3

def manual_clear():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM flagged_messages;")
    conn.commit()
    conn.close()
    print("All flagged messages have been deleted.")

manual_clear()
