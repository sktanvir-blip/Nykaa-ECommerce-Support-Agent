import random
from collections import Counter


# ============================================================
# 1. DATASET DESIGN CONFIGURATION
# ============================================================

SEED = 1
NUM_ORDERS = 60

CATEGORIES = [
    "Apparel",
    "Electronics",
    "Home",
    "Footwear",
    "Beauty",
]

CATEGORY_WEIGHTS = [
    25,
    15,
    15,
    25,
    20,
]

STATUSES = [
    "Placed",
    "Shipped",
    "Delivered",
    "Returned",
    "Refunded",
]

STATUS_WEIGHTS = [
    15,
    25,
    35,
    15,
    10,
]

MIN_ORDER_VALUE = 300
MAX_ORDER_VALUE = 25_000

MIN_DAYS = 0
MAX_DAYS = 30

DELAYED_SHIPMENT_PROBABILITY = 0.20


# ============================================================
# 2. DATASET GENERATOR
# ============================================================

def generate_orders():
    random.seed(SEED)

    orders = []

    for i in range(1, NUM_ORDERS + 1):

        category = random.choices(
            CATEGORIES,
            weights=CATEGORY_WEIGHTS,
            k=1
        )[0]

        status = random.choices(
            STATUSES,
            weights=STATUS_WEIGHTS,
            k=1
        )[0]

        order_value = random.randint(
            MIN_ORDER_VALUE,
            MAX_ORDER_VALUE
        )

        days_since_created = random.randint(
            MIN_DAYS,
            MAX_DAYS
        )

        delayed_shipment = (
            random.random() < DELAYED_SHIPMENT_PROBABILITY
        )

        order = {
            "record_id": f"ORD{i:04d}",
            "category": category,
            "status": status,
            "order_value_inr": order_value,
            "days_since_created": days_since_created,
            "delayed_shipment": delayed_shipment,
        }

        orders.append(order)

    return orders


# ============================================================
# 3. DATASET VALIDATION
# ============================================================

def validate_dataset(orders):

    print("\n========== DATASET VALIDATION ==========")

    # Total record count
    print(f"Total records: {len(orders)}")

    assert len(orders) >= 40, \
        "Dataset must contain at least 40 records."

    # Category counts
    category_counts = Counter(
        order["category"] for order in orders
    )

    print("\nCategory counts:")

    for category in CATEGORIES:
        print(
            f"{category}: "
            f"{category_counts[category]}"
        )

        assert category_counts[category] >= 3, \
            f"{category} must have at least 3 records."

    # Status counts
    status_counts = Counter(
        order["status"] for order in orders
    )

    print("\nStatus counts:")

    for status in STATUSES:
        print(
            f"{status}: "
            f"{status_counts[status]}"
        )

        assert status_counts[status] >= 1, \
            f"{status} must appear at least once."

    # Delayed shipment percentage
    delayed_count = sum(
        order["delayed_shipment"]
        for order in orders
    )

    delayed_percentage = (
        delayed_count / len(orders)
    ) * 100

    print("\nDelayed shipments:")
    print(
        f"{delayed_count}/{len(orders)} "
        f"= {delayed_percentage:.2f}%"
    )

    assert 10 <= delayed_percentage <= 30, \
        "Delayed shipment percentage must be between 10% and 30%."

    print("\nDataset validation PASSED.")


# ============================================================
# 4. DISPLAY SAMPLE RECORDS
# ============================================================

def print_sample_records(orders):

    print("\n========== SAMPLE RECORDS ==========")

    for order in orders[:5]:
        print(order)


# ============================================================
# 5. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    ORDERS = generate_orders()

    validate_dataset(ORDERS)

    print_sample_records(ORDERS)