from database.connection import get_db_connection
from mysql.connector import Error


def create_produce(data):
    connection = get_db_connection()

    if not connection:
        return None, "DATABASE_ERROR"

    try:
        cursor = connection.cursor()

        query = """
            INSERT INTO produce
            (farmer_id, crop, quantity, unit, expected_price, harvest_date, quality)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        values = (
            data["farmer_id"],
            data["crop"],
            data["quantity"],
            data["unit"],
            data["expected_price"],
            data["harvest_date"],
            data["quality"]
        )

        cursor.execute(query, values)
        connection.commit()

        produce_id = cursor.lastrowid

        cursor.close()
        connection.close()

        return produce_id, None

    except Error as e:
        print("Create produce error:", e)

        if connection:
            connection.rollback()
            connection.close()

        # Foreign key error: farmer does not exist
        if e.errno == 1452:
            return None, "NOT_FOUND"

        return None, "DATABASE_ERROR"


def get_produce(produce_id):
    connection = get_db_connection()

    if not connection:
        return None, "DATABASE_ERROR"

    try:
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                produce_id,
                farmer_id,
                crop,
                quantity,
                unit,
                expected_price,
                harvest_date,
                quality
            FROM produce
            WHERE produce_id = %s
        """

        cursor.execute(query, (produce_id,))
        produce = cursor.fetchone()

        cursor.close()
        connection.close()

        if not produce:
            return None, "NOT_FOUND"

        # Convert date to string for JSON
        if produce["harvest_date"]:
            produce["harvest_date"] = produce["harvest_date"].strftime("%Y-%m-%d")

        return produce, None

    except Error as e:
        print("Get produce error:", e)

        if connection:
            connection.close()

        return None, "DATABASE_ERROR"


def delete_produce(produce_id):
    connection = get_db_connection()

    if not connection:
        return False, "DATABASE_ERROR"

    try:
        cursor = connection.cursor()

        query = """
            DELETE FROM produce
            WHERE produce_id = %s
        """

        cursor.execute(query, (produce_id,))

        if cursor.rowcount == 0:
            cursor.close()
            connection.close()
            return False, "NOT_FOUND"

        connection.commit()

        cursor.close()
        connection.close()

        return True, None

    except Error as e:
        print("Delete produce error:", e)

        if connection:
            connection.rollback()
            connection.close()

        return False, "DATABASE_ERROR"