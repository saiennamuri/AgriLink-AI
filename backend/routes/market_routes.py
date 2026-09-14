from flask import Blueprint, request

from services.market_service import (
    get_markets,
    get_market_prices
)

from utils.response import success_response, error_response

from utils.validation import (
    is_valid_date
)


market_bp = Blueprint(
    "market",
    __name__,
    url_prefix="/api"
)


@market_bp.route("/markets", methods=["GET"])
def fetch_markets():

    district = request.args.get("district")
    state = request.args.get("state")

    filters = {
        "district": district,
        "state": state
    }

    markets, error = get_markets(filters)

    if error == "DATABASE_ERROR":

        return error_response(
            code="DATABASE_ERROR",
            message="Unable to retrieve markets",
            details={},
            status_code=500
        )

    return success_response(
        data={
            "markets": markets
        },
        message="Markets retrieved successfully",
        status_code=200
    )


@market_bp.route(
    "/markets/<int:market_id>/prices",
    methods=["GET"]
)
def fetch_market_prices(market_id):

    if market_id <= 0:

        return error_response(
            code="VALIDATION_ERROR",
            message="market_id must be a positive integer",
            details={
                "field": "market_id"
            },
            status_code=400
        )

    commodity = request.args.get("commodity")

    if not commodity or not commodity.strip():

        return error_response(
            code="VALIDATION_ERROR",
            message="commodity is required",
            details={
                "field": "commodity"
            },
            status_code=400
        )

    date_from = request.args.get("date_from")
    date_to = request.args.get("date_to")

    if date_from and not is_valid_date(date_from):

        return error_response(
            code="VALIDATION_ERROR",
            message="date_from must be in YYYY-MM-DD format",
            details={
                "field": "date_from"
            },
            status_code=400
        )

    if date_to and not is_valid_date(date_to):

        return error_response(
            code="VALIDATION_ERROR",
            message="date_to must be in YYYY-MM-DD format",
            details={
                "field": "date_to"
            },
            status_code=400
        )

    if date_from and date_to and date_from > date_to:

        return error_response(
            code="VALIDATION_ERROR",
            message="date_from cannot be later than date_to",
            details={
                "date_from": date_from,
                "date_to": date_to
            },
            status_code=400
        )

    prices, error = get_market_prices(
        market_id,
        commodity.strip(),
        date_from,
        date_to
    )

    if error == "MARKET_NOT_FOUND":

        return error_response(
            code="NOT_FOUND",
            message="Market not found",
            details={
                "market_id": market_id
            },
            status_code=404
        )

    if error == "DATABASE_ERROR":

        return error_response(
            code="DATABASE_ERROR",
            message="Unable to retrieve market prices",
            details={},
            status_code=500
        )

    return success_response(
        data={
            "market_id": market_id,
            "commodity": commodity.strip(),
            "prices": prices
        },
        message="Market prices retrieved successfully",
        status_code=200
    )