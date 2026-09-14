from database.connection import get_db_connection
from mysql.connector import Error


def create_farmer(data):
    connection = get_db_connection()

    if not connection:
        return None, "DATABASE_ERROR"

    try:
        cursor = connection.cursor()

        query = """
            INSERT INTO farmers
            (name, phone, location, district, state)
            VALUES (%s, %s, %s, %s, %s)
        """

        values = (
            data["name"],
            data["phone"],
            data["location"],
            data["district"],
            data["state"]
        )

        cursor.execute(query, values)
        connection.commit()

        farmer_id = cursor.lastrowid

        cursor.close()
        connection.close()

        return farmer_id, None

    except Error as e:
         print("Create farmer error:", e)

    if connection:
        connection.rollback()
        connection.close()

    if e.errno == 1062:
        return None, "DUPLICATE_RESOURCE"

    return None, "DATABASE_ERROR"

def get_farmer(farmer_id):
    connection = get_db_connection()

    if not connection:
        return None, "DATABASE_ERROR"

    try:
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT farmer_id, name, phone, location, district, state
            FROM farmers
            WHERE farmer_id = %s
        """

        cursor.execute(query, (farmer_id,))
        farmer = cursor.fetchone()

        cursor.close()
        connection.close()

        if not farmer:
            return None, "NOT_FOUND"

        return farmer, None

    except Error as e:
        print("Get farmer error:", e)

        if connection:
            connection.close()

        return None, "DATABASE_ERROR"


def update_farmer(farmer_id, data):
    connection = get_db_connection()

    if not connection:
        return False, "DATABASE_ERROR"

    try:
        cursor = connection.cursor()

        query = """
            UPDATE farmers
            SET name = %s,
                phone = %s,
                location = %s,
                district = %s,
                state = %s
            WHERE farmer_id = %s
        """

        values = (
            data["name"],
            data["phone"],
            data["location"],
            data["district"],
            data["state"],
            farmer_id
        )

        cursor.execute(query, values)

        if cursor.rowcount == 0:
            cursor.close()
            connection.close()
            return False, "NOT_FOUND"

        connection.commit()

        cursor.close()
        connection.close()

        return True, None

    except Error as e:
        print("Update farmer error:", e)

        if connection:
            connection.rollback()
            connection.close()

        return False, "DATABASE_ERROR"