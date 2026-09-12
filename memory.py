from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


def get_session_history(session_id: str):
    """
    Return the conversation history for a session.

    If the session does not exist yet, create a new
    InMemoryChatMessageHistory object.
    """

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]

def mock_memory_responder(inputs):
    """
    Deterministic mock responder used to demonstrate
    LangChain session memory without an API key.
    """

    user_input = inputs["input"]
    history = inputs.get("history", [])

    user_messages = [
        message.content
        for message in history
        if message.type == "human"
    ]

    user_input_lower = user_input.lower()

    if "what is my name" in user_input_lower:

        if user_messages:
            first_message = user_messages[0]

            if "my name is" in first_message.lower():
                name = first_message.split(
                    "is",
                    1
                )[1].strip()

                return (
                    f"You previously told me that "
                    f"your name is {name}."
                )

        return "I don't have your name in this session."

    if "what did i ask" in user_input_lower:

        if user_messages:
            previous_questions = user_messages[:-1]

            if previous_questions:
                return (
                    "Earlier in this session, you asked: "
                    + " | ".join(previous_questions)
                )

        return "You have not asked another question yet."

    return (
        f"I received your message: {user_input}"
    )

mock_chain = RunnableLambda(
    mock_memory_responder
)

memory_chain = RunnableWithMessageHistory(
    mock_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


if __name__ == "__main__":

    session_id = "nykaa_demo_session"

    config = {
        "configurable": {
            "session_id": session_id
        }
    }

    # First message
    response_1 = memory_chain.invoke(
        {
            "input": "My name is Tanvir."
        },
        config=config
    )

    print("\nUser:")
    print("My name is Tanvir.")

    print("\nAssistant:")
    print(response_1)

    # Second message
    response_2 = memory_chain.invoke(
        {
            "input": "What is my name?"
        },
        config=config
    )

    print("\nUser:")
    print("What is my name?")

    print("\nAssistant:")
    print(response_2)

    # Display stored conversation
    print("\nSTORED SESSION HISTORY")

    history = get_session_history(session_id)

    for message in history.messages:
        print(
            f"{message.type}: {message.content}"
        )

    print("\nSESSION ISOLATION TEST")

    session_b_id = "nykaa_demo_session_b"

    config_b = {
        "configurable": {
            "session_id": session_b_id
        }
    }

    response_3 = memory_chain.invoke(
        {
            "input": "What is my name?"
        },
        config=config_b
    )

    print("\nSession A:")
    print(
        "Stored name: Tanvir"
    )

    print("\nSession B:")
    print(
        "User: What is my name?"
    )

    print("\nAssistant:")
    print(response_3)

    print("\nSession B stored history:")

    history_b = get_session_history(
        session_b_id
    )

    for message in history_b.messages:
        print(
            f"{message.type}: {message.content}"
        )

    print(
        "\nSession A and Session B "
        "use separate session IDs."
    )