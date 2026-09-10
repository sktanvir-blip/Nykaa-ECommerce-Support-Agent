# ============================================================
# TASK 16: IN-MEMORY CACHE
# ============================================================

from typing import Any


# ------------------------------------------------------------
# 1. In-memory cache storage
# ------------------------------------------------------------

CACHE = {}


# ------------------------------------------------------------
# 2. Normalize the query
# ------------------------------------------------------------

def normalize_query(query: str) -> str:
    """
    Convert equivalent queries into the same cache key.
    """

    normalized = " ".join(query.lower().strip().split())

    return normalized


# ------------------------------------------------------------
# 3. Get value from cache
# ------------------------------------------------------------

def get_cached_response(query: str) -> Any:
    """
    Return the cached response if the normalized query exists.
    Otherwise return None.
    """

    cache_key = normalize_query(query)

    if cache_key in CACHE:
        return CACHE[cache_key]

    return None


# ------------------------------------------------------------
# 4. Store response in cache
# ------------------------------------------------------------

def set_cached_response(query: str, response: Any) -> None:
    """
    Store a response using the normalized query as the key.
    """

    cache_key = normalize_query(query)

    CACHE[cache_key] = response


# ------------------------------------------------------------
# 5. Cached execution wrapper
# ------------------------------------------------------------

def get_or_execute(query: str, execute_function):
    """
    Return a cached response when available.

    If the response is not cached, execute the supplied
    function, store the result, and return it.
    """

    cached_response = get_cached_response(query)

    if cached_response is not None:
        return cached_response, "CACHE HIT"

    response = execute_function(query)

    set_cached_response(query, response)

    return response, "CACHE MISS"


# ------------------------------------------------------------
# 6. Demonstration
# ------------------------------------------------------------

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("TASK 16 - IN-MEMORY CACHE TEST")
    print("=" * 70)

    execution_count = 0


    # --------------------------------------------------------
    # Simulated expensive operation
    # --------------------------------------------------------

    def expensive_operation(query):

        global execution_count

        execution_count += 1

        print("\nActual operation executed.")

        return (
            "Footwear can be returned within 15 days of delivery "
            "according to the Nykaa policy."
        )


    # --------------------------------------------------------
    # First request
    # --------------------------------------------------------

    query_1 = "What is the return policy for footwear?"

    response_1, cache_status_1 = get_or_execute(
        query_1,
        expensive_operation
    )

    print("\nFIRST REQUEST")
    print("Query:", query_1)
    print("Cache status:", cache_status_1)
    print("Response:", response_1)


    # --------------------------------------------------------
    # Second request
    # Same query with different capitalization/spaces
    # --------------------------------------------------------

    query_2 = "   WHAT IS THE RETURN POLICY FOR FOOTWEAR?   "

    response_2, cache_status_2 = get_or_execute(
        query_2,
        expensive_operation
    )

    print("\nSECOND REQUEST")
    print("Query:", query_2)
    print("Cache status:", cache_status_2)
    print("Response:", response_2)


    # --------------------------------------------------------
    # Verify expensive operation was executed only once
    # --------------------------------------------------------

    print("\n")
    print("Actual expensive-operation executions:", execution_count)

    print("\n")
    print("=" * 70)
    print("TASK 16 COMPLETE")
    print("=" * 70)