from flask import Blueprint, request

from services.farmer_service import (
    create_farmer,
    get_farmer,
    update_farmer
)

from utils.response import success_response, error_response
from utils.validation import required_fields, is_valid_phone


farmer_bp = Blueprint("farmer_bp", __name__)


# ============================================================
# POST /api/farmers
# Register Farmer
# ============================================================

@farmer_bp.route("/api/farmers", methods=["POST"])
def register_farmer():

    data = request.get_json(silent=True)

    if not data:
        return error_response(
            code="VALIDATION_ERROR",
            message="Request body is required",
            details={},
            status_code=400
        )

    required = [
        "name",
        "phone",
        "location",
        "district",
        "state"
    ]

    missing = required_fields(data, required)

    if missing:
        return error_response(
            code="VALIDATION_ERROR",
            message="Required fields are missing",
            details={"missing_fields": missing},
            status_code=400
        )

    if not is_valid_phone(data["phone"]):
        return error_response(
            code="VALIDATION_ERROR",
            message="Invalid phone number",
            details={"field": "phone"},
            status_code=400
        )

    farmer_id, error = create_farmer(data)

    if error == "DUPLICATE_RESOURCE":
        return error_response(
        code="DUPLICATE_RESOURCE",
        message="A farmer with this phone number already exists",
        details={"field": "phone"},
        status_code=409
    )

    if error == "DATABASE_ERROR":
        return error_response(
        code="DATABASE_ERROR",
        message="Unable to register farmer",
        details={},
        status_code=500
    )

    return success_response(
        data={
            "farmer_id": farmer_id,
            "name": data["name"],
            "phone": data["phone"],
            "location": data["location"],
            "district": data["district"],
            "state": data["state"]
        },
        message="Farmer registered successfully",
        status_code=201
    )


# ============================================================
# GET /api/farmers/<farmer_id>
# Get Farmer
# ============================================================

@farmer_bp.route("/api/farmers/<int:farmer_id>", methods=["GET"])
def get_farmer_details(farmer_id):

    farmer, error = get_farmer(farmer_id)

    if error == "NOT_FOUND":
        return error_response(
            code="NOT_FOUND",
            message="Farmer not found",
            details={"farmer_id": farmer_id},
            status_code=404
        )

    if error == "DATABASE_ERROR":
        return error_response(
            code="DATABASE_ERROR",
            message="Unable to retrieve farmer",
            details={},
            status_code=500
        )

    return success_response(
        data=farmer,
        message="Farmer retrieved successfully",
        status_code=200
    )


# ============================================================
# PUT /api/farmers/<farmer_id>
# Update Farmer
# ============================================================

@farmer_bp.route("/api/farmers/<int:farmer_id>", methods=["PUT"])
def update_farmer_details(farmer_id):

    data = request.get_json(silent=True)

    if not data:
        return error_response(
            code="VALIDATION_ERROR",
            message="Request body is required",
            details={},
            status_code=400
        )

    required = [
        "name",
        "phone",
        "location",
        "district",
        "state"
    ]

    missing = required_fields(data, required)

    if missing:
        return error_response(
            code="VALIDATION_ERROR",
            message="Required fields are missing",
            details={"missing_fields": missing},
            status_code=400
        )

    if not is_valid_phone(data["phone"]):
        return error_response(
            code="VALIDATION_ERROR",
            message="Invalid phone number",
            details={"field": "phone"},
            status_code=400
        )

    updated, error = update_farmer(farmer_id, data)

    if error == "NOT_FOUND":
        return error_response(
            code="NOT_FOUND",
            message="Farmer not found",
            details={"farmer_id": farmer_id},
            status_code=404
        )

    if error == "DATABASE_ERROR":
        return error_response(
            code="DATABASE_ERROR",
            message="Unable to update farmer",
            details={},
            status_code=500
        )

    return success_response(
        data={
            "farmer_id": farmer_id
        },
        message="Farmer updated successfully",
        status_code=200
    )