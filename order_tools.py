from dataset import generate_orders


ORDERS = generate_orders()

ESCALATION_THRESHOLD = 0.60


def check_order_status(record_id):
    """
    Look up an order and calculate its escalation score.
    """

    for order in ORDERS:

        if order["record_id"] == record_id:

            delayed_score = (
                1.0
                if order["delayed_shipment"]
                else 0.0
            )

            recency_score = (
                order["days_since_created"] / 30
            )

            escalation_score = (
                0.6 * delayed_score
                +
                0.4 * recency_score
            )

            escalation_score = min(
                escalation_score,
                1.0
            )

            return {
                "record_id": order["record_id"],
                "status": order["status"],
                "order_value_inr": order["order_value_inr"],
                "days_since_created": order[
                    "days_since_created"
                ],
                "delayed_shipment": order[
                    "delayed_shipment"
                ],
                "escalation_score": round(
                    escalation_score,
                    4
                ),
                "escalation_required": (
                    escalation_score
                    >= ESCALATION_THRESHOLD
                ),
            }

    return {
        "record_id": record_id,
        "error": "Order not found."
    }


if __name__ == "__main__":

    delayed_order = next(
        order
        for order in ORDERS
        if order["delayed_shipment"] is True
    )

    normal_order = next(
        order
        for order in ORDERS
        if order["delayed_shipment"] is False
    )

    print("\n--- Delayed Order Test ---")

    delayed_result = check_order_status(
        delayed_order["record_id"]
    )

    print(delayed_result)

    print("\n--- Normal Order Test ---")

    normal_result = check_order_status(
        normal_order["record_id"]
    )

    print(normal_result)

    print("\n--- Invalid Order Test ---")

    invalid_result = check_order_status(
        "ORD9999"
    )

    print(invalid_result)