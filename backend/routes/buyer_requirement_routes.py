from flask import Blueprint, request

from services.buyer_requirement_service import (
    create_buyer_requirement
)

from utils.response import success_response, error_response

from utils.validation import (
    required_fields,
    is_positive_number,
    is_non_negative_number
)


buyer_requirement_bp = Blueprint(
    "buyer_requirement",
    __name__,
    url_prefix="/api"
)


@buyer_requirement_bp.route(
    "/buyer-requirements",
    methods=["POST"]
)
def add_buyer_requirement():

    data = request.get_json()

    if not data:
        return error_response(
            code="VALIDATION_ERROR",
            message="Request body is required",
            details={},
            status_code=400
        )

    required = [
        "buyer_id",
        "commodity",
        "required_quantity",
        "unit",
        "offered_price",
        "quality_requirement"
    ]

    missing = required_fields(data, required)

    if missing:
        return error_response(
            code="VALIDATION_ERROR",
            message="Required fields are missing",
            details={
                "missing_fields": missing
            },
            status_code=400
        )

    try:
        buyer_id = int(data["buyer_id"])

        if buyer_id <= 0:
            raise ValueError

    except (TypeError, ValueError):

        return error_response(
            code="VALIDATION_ERROR",
            message="buyer_id must be a positive integer",
            details={
                "field": "buyer_id"
            },
            status_code=400
        )

    if not is_positive_number(
        data["required_quantity"]
    ):
        return error_response(
            code="VALIDATION_ERROR",
            message="required_quantity must be greater than 0",
            details={
                "field": "required_quantity"
            },
            status_code=400
        )

    if not is_non_negative_number(
        data["offered_price"]
    ):
        return error_response(
            code="VALIDATION_ERROR",
            message="offered_price must be greater than or equal to 0",
            details={
                "field": "offered_price"
            },
            status_code=400
        )

    data["buyer_id"] = buyer_id

    requirement_id, error = create_buyer_requirement(data)

    if error == "BUYER_NOT_FOUND":

        return error_response(
            code="NOT_FOUND",
            message="Buyer not found",
            details={
                "buyer_id": buyer_id
            },
            status_code=404
        )

    if error == "DATABASE_ERROR":

        return error_response(
            code="DATABASE_ERROR",
            message="Unable to create buyer requirement",
            details={},
            status_code=500
        )

    return success_response(
        data={
            "requirement_id": requirement_id,
            "buyer_id": buyer_id,
            "commodity": data["commodity"],
            "required_quantity": data["required_quantity"],
            "unit": data["unit"],
            "offered_price": data["offered_price"],
            "quality_requirement": data["quality_requirement"]
        },
        message="Buyer requirement created successfully",
        status_code=201
    )