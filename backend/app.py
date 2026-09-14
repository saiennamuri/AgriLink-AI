from flask import Flask
from flask_cors import CORS

from database.connection import get_db_connection
from utils.response import success_response, error_response
from utils.errors import register_error_handlers

from routes.farmer_routes import farmer_bp
from routes.produce_routes import produce_bp
from routes.buyer_routes import buyer_bp
from routes.buyer_requirement_routes import buyer_requirement_bp
from routes.market_routes import market_bp

app = Flask(__name__)

CORS(app, origins=["http://localhost:5173"])

register_error_handlers(app)

app.register_blueprint(farmer_bp)
app.register_blueprint(produce_bp)
app.register_blueprint(buyer_bp)
app.register_blueprint(buyer_requirement_bp)
app.register_blueprint(market_bp)

@app.route("/api/health", methods=["GET"])
def health_check():

    connection = get_db_connection()

    if connection:
        connection.close()

        return success_response(
            data={"status": "running"},
            message="AgriLink AI backend is running",
            status_code=200
        )

    return error_response(
        code="DATABASE_ERROR",
        message="Unable to connect to database",
        details={},
        status_code=500
    )


if __name__ == "__main__":
    app.run(
        host="localhost",
        port=5000,
        debug=True
    )