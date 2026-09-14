from flask import Blueprint, request

from services.buyer_service import (
    create_buyer,
    get_buyer,
    get_buyers
)
from utils.response import success_response, error_response

from utils.validation import (
    required_fields,
    is_valid_phone
)


buyer_bp = Blueprint(
    "buyer",
    __name__,
    url_prefix="/api"
)


# --------------------------------------------------
# POST /api/buyers
# Register a new buyer
# --------------------------------------------------
@buyer_bp.route("/buyers", methods=["GET"])
def fetch_buyers():

    district = request.args.get("district")
    state = request.args.get("state")
    commodity = request.args.get("commodity")
    verified = request.args.get("verified")

    filters = {
        "district": district,
        "state": state,
        "commodity": commodity,
        "verified": None
    }

    if verified is not None:

        if verified.lower() not in ["true", "false"]:
            return error_response(
                code="VALIDATION_ERROR",
                message="verified must be true or false",
                details={"field": "verified"},
                status_code=400
            )

        filters["verified"] = verified.lower() == "true"

    buyers, error = get_buyers(filters)

    if error == "DATABASE_ERROR":
        return error_response(
            code="DATABASE_ERROR",
            message="Unable to retrieve buyers",
            details={},
            status_code=500
        )

    return success_response(
        data={
            "buyers": buyers
        },
        message="Buyers retrieved successfully",
        status_code=200
    )

# --------------------------------------------------
# GET /api/buyers/{buyer_id}
# Get a specific buyer
# --------------------------------------------------

@buyer_bp.route("/buyers/<int:buyer_id>", methods=["GET"])
def fetch_buyer(buyer_id):

    if buyer_id <= 0:
        return error_response(
            code="VALIDATION_ERROR",
            message="buyer_id must be a positive integer",
            details={
                "field": "buyer_id"
            },
            status_code=400
        )

    buyer, error = get_buyer(buyer_id)

    if error == "NOT_FOUND":
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
            message="Unable to retrieve buyer",
            details={},
            status_code=500
        )

    return success_response(
        data=buyer,
        message="Buyer retrieved successfully",
        status_code=200
    )