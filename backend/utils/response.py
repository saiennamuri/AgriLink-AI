from flask import jsonify


def success_response(data=None, message="Operation completed successfully", status_code=200):
    response = {
        "success": True,
        "data": data if data is not None else {},
        "message": message
    }

    return jsonify(response), status_code


def error_response(code, message, details=None, status_code=400):
    response = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "details": details if details is not None else {}
        }
    }

    return jsonify(response), status_code