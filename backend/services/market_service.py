from database.connection import get_db_connection
from mysql.connector import Error


def get_markets(filters=None):

    connection = get_db_connection()

    if not connection:
        return None, "DATABASE_ERROR"

    try:

        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                market_id,
                market_name,
                location,
                district,
                state
            FROM markets
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

        query += " ORDER BY market_id DESC"

        cursor.execute(query, tuple(values))

        markets = cursor.fetchall()

        cursor.close()
        connection.close()

        return markets, None

    except Error as e:

        print("Get markets error:", e)

        if connection:
            connection.close()

        return None, "DATABASE_ERROR"


def get_market_prices(market_id, commodity, date_from=None, date_to=None):

    connection = get_db_connection()

    if not connection:
        return None, "DATABASE_ERROR"

    try:

        cursor = connection.cursor(dictionary=True)

        # Check whether market exists
        cursor.execute(
            "SELECT market_id FROM markets WHERE market_id = %s",
            (market_id,)
        )

        market = cursor.fetchone()

        if not market:
            cursor.close()
            connection.close()
            return None, "MARKET_NOT_FOUND"

        query = """
            SELECT
                price_id,
                market_id,
                commodity,
                price_date,
                min_price,
                max_price,
                modal_price,
                arrival_quantity,
                unit
            FROM market_prices
            WHERE market_id = %s
            AND commodity = %s
        """

        values = [market_id, commodity]

        if date_from:
            query += " AND price_date >= %s"
            values.append(date_from)

        if date_to:
            query += " AND price_date <= %s"
            values.append(date_to)

        query += " ORDER BY price_date DESC"

        cursor.execute(query, tuple(values))

        prices = cursor.fetchall()

        cursor.close()
        connection.close()

        return prices, None

    except Error as e:

        print("Get market prices error:", e)

        if connection:
            connection.close()

        return None, "DATABASE_ERROR"