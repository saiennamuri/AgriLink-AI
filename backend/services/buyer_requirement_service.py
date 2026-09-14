from database.connection import get_db_connection
from mysql.connector import Error


def create_buyer_requirement(data):

    connection = get_db_connection()

    if not connection:
        return None, "DATABASE_ERROR"

    try:
        cursor = connection.cursor()

        # Check whether buyer exists
        cursor.execute(
            "SELECT buyer_id FROM buyers WHERE buyer_id = %s",
            (data["buyer_id"],)
        )

        buyer = cursor.fetchone()

        if not buyer:
            cursor.close()
            connection.close()
            return None, "BUYER_NOT_FOUND"

        query = """
            INSERT INTO buyer_requirements
            (
                buyer_id,
                commodity,
                required_quantity,
                unit,
                offered_price,
                quality_requirement
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """

        values = (
            data["buyer_id"],
            data["commodity"],
            data["required_quantity"],
            data["unit"],
            data["offered_price"],
            data["quality_requirement"]
        )

        cursor.execute(query, values)

        connection.commit()

        requirement_id = cursor.lastrowid

        cursor.close()
        connection.close()

        return requirement_id, None

    except Error as e:

        print("Create buyer requirement error:", e)

        if connection:
            connection.rollback()
            connection.close()

        return None, "DATABASE_ERROR"