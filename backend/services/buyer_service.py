from database.connection import get_db_connection
from mysql.connector import Error


def create_buyer(data):

    connection = get_db_connection()

    if not connection:
        return None, "DATABASE_ERROR"

    try:
        cursor = connection.cursor()

        query = """
            INSERT INTO buyers
            (name, phone, location, district, state, buyer_type)
            VALUES (%s, %s, %s, %s, %s, %s)
        """

        values = (
            data["name"],
            data["phone"],
            data["location"],
            data["district"],
            data["state"],
            data["buyer_type"]
        )

        cursor.execute(query, values)

        connection.commit()

        buyer_id = cursor.lastrowid

        cursor.close()
        connection.close()

        return buyer_id, None

    except Error as e:

        print("Create buyer error:", e)

        if connection:
            connection.rollback()
            connection.close()

        if e.errno == 1062:
            return None, "DUPLICATE_RESOURCE"

        return None, "DATABASE_ERROR"


def get_buyer(buyer_id):

    connection = get_db_connection()

    if not connection:
        return None, "DATABASE_ERROR"

    try:

        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                buyer_id,
                name,
                phone,
                location,
                district,
                state,
                buyer_type,
                verified
            FROM buyers
            WHERE buyer_id = %s
        """

        cursor.execute(query, (buyer_id,))

        buyer = cursor.fetchone()

        cursor.close()
        connection.close()

        if not buyer:
            return None, "NOT_FOUND"

        buyer["verified"] = bool(buyer["verified"])

        return buyer, None

    except Error as e:

        print("Get buyer error:", e)

        if connection:
            connection.close()

        return None, "DATABASE_ERROR"


def get_buyers(filters=None):

    connection = get_db_connection()

    if not connection:
        return None, "DATABASE_ERROR"

    try:

        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                buyer_id,
                name,
                phone,
                location,
                district,
                state,
                buyer_type,
                verified
            FROM buyers
            WHERE 1=1
        """

        values = []

        if filters:

            if filters.get("district"):
                query += " AND district = %s"
                values.append(filters["district"])

            if filters.get("state"):
                query += " AND state = %s"
                values.append(filters["state"])

            if filters.get("verified") is not None:
                query += " AND verified = %s"
                values.append(filters["verified"])

            if filters.get("commodity"):
                query += """
                    AND buyer_id IN (
                        SELECT buyer_id
                        FROM buyer_requirements
                        WHERE commodity = %s
                    )
                """

                values.append(filters["commodity"])

        query += " ORDER BY buyer_id DESC"

        cursor.execute(query, tuple(values))

        buyers = cursor.fetchall()

        cursor.close()
        connection.close()

        for buyer in buyers:
            buyer["verified"] = bool(buyer["verified"])

        return buyers, None

    except Error as e:

        print("Get buyers error:", e)

        if connection:
            connection.close()

        return None, "DATABASE_ERROR"