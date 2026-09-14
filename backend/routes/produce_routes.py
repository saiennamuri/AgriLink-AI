from flask import Blueprint, request

from services.produce_service import (
    create_produce,
    get_produce,
    delete_produce
)

from utils.response import success_response, error_response
from utils.validation import (
    required_fields,
    is_positive_number,
    is_non_negative_number,
    is_valid_date
)


produce_bp = Blueprint("produce", __name__, url_prefix="/api")


# --------------------------------------------------
# POST /api/produce
# --------------------------------------------------
@produce_bp.route("/produce", methods=["POST"])
def add_produce():

    data = request.get_json()

    if not data:
        return error_response(
            code="VALIDATION_ERROR",
            message="Request body is required",
            details={},
            status_code=400
        )

    required = [
        "farmer_id",
        "crop",
        "quantity",
        "unit",
        "expected_price",
        "harvest_date",
        "quality"
    ]

    missing = required_fields(data, required)

    if missing:
        return error_response(
            code="VALIDATION_ERROR",
            message="Required fields are missing",
            details={"missing_fields": missing},
            status_code=400
        )

    # Validate farmer_id
    try:
        farmer_id = int(data["farmer_id"])

        if farmer_id <= 0:
            raise ValueError

    except (TypeError, ValueError):
        return error_response(
            code="VALIDATION_ERROR",
            message="farmer_id must be a valid positive integer",
            details={"field": "farmer_id"},
            status_code=400
        )

    # Validate quantity
    if not is_positive_number(data["quantity"]):
        return error_response(
            code="VALIDATION_ERROR",
            message="Quantity must be greater than 0",
            details={"field": "quantity"},
            status_code=400
        )

    # Validate expected price
    if not is_non_negative_number(data["expected_price"]):
        return error_response(
            code="VALIDATION_ERROR",
            message="Expected price must be greater than or equal to 0",
            details={"field": "expected_price"},
            status_code=400
        )

    # Validate harvest date
    if not is_valid_date(data["harvest_date"]):
        return error_response(
            code="VALIDATION_ERROR",
            message="Harvest date must be in YYYY-MM-DD format",
            details={"field": "harvest_date"},
            status_code=400
        )

    produce_id, error = create_produce(data)

    if error == "NOT_FOUND":
        return error_response(
            code="NOT_FOUND",
            message="Farmer not found",
            details={"field": "farmer_id"},
            status_code=404
        )

    if error == "DATABASE_ERROR":
        return error_response(
            code="DATABASE_ERROR",
            message="Unable to submit produce",
            details={},
            status_code=500
        )

    return success_response(
        data={
            "produce_id": produce_id,
            "farmer_id": data["farmer_id"],
            "crop": data["crop"],
            "quantity": data["quantity"],
            "unit": data["unit"],
            "expected_price": data["expected_price"],
            "harvest_date": data["harvest_date"],
            "quality": data["quality"]
        },
        message="Produce submitted successfully",
        status_code=201
    )


# --------------------------------------------------
# GET /api/produce/{produce_id}
# --------------------------------------------------
@produce_bp.route("/produce/<int:produce_id>", methods=["GET"])
def fetch_produce(produce_id):

    if produce_id <= 0:
        return error_response(
            code="VALIDATION_ERROR",
            message="produce_id must be a positive integer",
            details={"field": "produce_id"},
            status_code=400
        )

    produce, error = get_produce(produce_id)

    if error == "NOT_FOUND":
        return error_response(
            code="NOT_FOUND",
            message="Produce not found",
            details={"produce_id": produce_id},
            status_code=404
        )

    if error == "DATABASE_ERROR":
        return error_response(
            code="DATABASE_ERROR",
            message="Unable to retrieve produce",
            details={},
            status_code=500
        )

    return success_response(
        data=produce,
        message="Produce retrieved successfully",
        status_code=200
    )


# --------------------------------------------------
# DELETE /api/produce/{produce_id}
# --------------------------------------------------
@produce_bp.route("/produce/<int:produce_id>", methods=["DELETE"])
def remove_produce(produce_id):

    if produce_id <= 0:
        return error_response(
            code="VALIDATION_ERROR",
            message="produce_id must be a positive integer",
            details={"field": "produce_id"},
            status_code=400
        )

    deleted, error = delete_produce(produce_id)

    if error == "NOT_FOUND":
        return error_response(
            code="NOT_FOUND",
            message="Produce not found",
            details={"produce_id": produce_id},
            status_code=404
        )

    if error == "DATABASE_ERROR":
        return error_response(
            code="DATABASE_ERROR",
            message="Unable to delete produce",
            details={},
            status_code=500
        )

    return success_response(
        data={
            "produce_id": produce_id
        },
        message="Produce deleted successfully",
        status_code=200
    )