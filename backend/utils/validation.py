from datetime import datetime
import re


def is_valid_phone(phone):
    """
    Validate Indian-style 10 digit phone number.
    """
    if not phone:
        return False

    phone = str(phone).strip()

    return bool(re.fullmatch(r"[6-9]\d{9}", phone))


def is_valid_date(date_value):
    """
    Validate date in YYYY-MM-DD format.
    """
    if not date_value:
        return False

    try:
        datetime.strptime(str(date_value), "%Y-%m-%d")
        return True
    except ValueError:
        return False


def is_positive_number(value):
    """
    Check whether value is greater than 0.
    """
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def is_non_negative_number(value):
    """
    Check whether value is greater than or equal to 0.
    """
    try:
        return float(value) >= 0
    except (TypeError, ValueError):
        return False


def required_fields(data, fields):
    """
    Return missing required fields.
    """
    missing = []

    for field in fields:
        if field not in data or data[field] is None:
            missing.append(field)
        elif isinstance(data[field], str) and not data[field].strip():
            missing.append(field)

    return missing