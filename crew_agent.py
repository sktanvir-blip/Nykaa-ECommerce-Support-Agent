from typing import Type
import json
from pydantic import BaseModel, Field

from crewai import Agent, Crew, Task, Process
from crewai.tools import BaseTool
from crewai.llms.base_llm import BaseLLM

from order_tools import check_order_status
import chromadb
from sentence_transformers import SentenceTransformer

from rag import grounded_retrieval, generate_grounded_answer
from guardrails import apply_input_guardrails, crew_groundedness_guardrail

# --------------------------------------------------------
# Task 9: Pydantic Structured Response Model
# --------------------------------------------------------

class NykaaResponse(BaseModel):
    answer: str = Field(
        ...,
        description="Final customer support answer."
    )

    sources: list[str] = Field(
        default_factory=list,
        description="Knowledge-base sources used for the answer."
    )

    grounded: bool = Field(
        ...,
        description="Whether the answer is grounded in available information."
    )

    order_id: str | None = Field(
        default=None,
        description="Order record ID when an order lookup was used."
    )

    escalation_required: bool = Field(
        default=False,
        description="Whether the order requires escalation."
    )

    # --------------------------------------------------------
# Task 9: Response Validation
# --------------------------------------------------------

def validate_crew_response(response_data):
    """
    Validate every CrewAI response using Pydantic.
    """

    if isinstance(response_data, str):
        response_data = json.loads(response_data)

    validated_response = NykaaResponse.model_validate(
        response_data
    )

    return validated_response

# --------------------------------------------------------
# Task 7: Order Lookup Tool
# --------------------------------------------------------

class OrderLookupInput(BaseModel):
    record_id: str = Field(
        ...,
        description="The order record ID, for example ORD0001."
    )


class OrderLookupTool(BaseTool):
    name: str = "order_lookup_tool"

    description: str = (
        "Looks up a Nykaa order using its record ID. "
        "Returns order status, order value, delay information, "
        "and escalation score."
    )

    args_schema: Type[BaseModel] = OrderLookupInput

    def _run(self, record_id: str) -> str:
        result = check_order_status(record_id)

        return str(result)

# --------------------------------------------------------
# Task 7: RAG Retrieval Tool
# --------------------------------------------------------

class RetrievalInput(BaseModel):
    query: str = Field(
        ...,
        description="The user's Nykaa knowledge-base question."
    )


class RetrievalTool(BaseTool):
    name: str = "nykaa_knowledge_base_search"

    description: str = (
        "Searches the Nykaa knowledge base for relevant "
        "policy information such as returns, refunds, "
        "delivery, warranty, exchanges, and support."
    )

    args_schema: Type[BaseModel] = RetrievalInput

    def _run(self, query: str) -> str:

        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        chroma_client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        sentence_collection = (
            chroma_client.get_collection(
                name="nykaa_sentence_chunks"
            )
        )

        retrieval_result = grounded_retrieval(
            sentence_collection,
            query,
            embedding_model
        )

        answer = generate_grounded_answer(
            query,
            retrieval_result
        )

        return str(answer)

# --------------------------------------------------------
# Task 7: MOCK_LLM
# --------------------------------------------------------

class MockLLM(BaseLLM):

    def __init__(self):
        super().__init__(
            model="nykaa-mock-llm",
            temperature=0
        )

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):

        # Convert messages into searchable text
        if isinstance(messages, list):
            message_text = "\n".join(
                str(message.get("content", ""))
                for message in messages
                if message.get("content") is not None
            )
        else:
            message_text = str(messages)

        message_lower = message_text.lower()

        agent_role = ""

        if from_agent is not None:
            agent_role = from_agent.role.lower()

        # Check whether a previous native tool call
        # has already produced a tool result.
        tool_result_exists = False

        if isinstance(messages, list):
            tool_result_exists = any(
                message.get("role") == "tool"
                for message in messages
                if isinstance(message, dict)
            )

        # ------------------------------------------------
        # Retrieval Agent
        # ------------------------------------------------

        if "retrieval specialist" in agent_role:

            # If RAG tool has already run,
            # return the final retrieval result.
            if tool_result_exists:

                return (
                    "Footwear can be returned within 15 days "
                    "according to the retrieved Nykaa "
                    "knowledge base information."
                )

            # Otherwise call the RAG tool.
            if (
                "return footwear" in message_lower
                or "knowledge base" in message_lower
                or "return window" in message_lower
                or "policy" in message_lower
            ):

                return [
                    {
                        "id": "call_retrieval_001",
                        "type": "function",
                        "function": {
                            "name": (
                                "nykaa_knowledge_base_search"
                            ),
                            "arguments": (
                                "{\"query\": "
                                "\"How many days can I return footwear?\"}"
                            ),
                        },
                    }
                ]

            return (
                "No knowledge-base retrieval is required "
                "for this order-status request."
            )

        # ------------------------------------------------
        # Lookup Agent
        # ------------------------------------------------

        if "order lookup specialist" in agent_role:

            # If order tool has already run,
            # return the final lookup result.
            if tool_result_exists:

                return (
                    "The order lookup tool returned the "
                    "requested information for the order."
                )

            # Call order lookup only when an order ID
            # is present in the request.
            if "ord0001" in message_lower:

                return [
                    {
                        "id": "call_order_001",
                        "type": "function",
                        "function": {
                            "name": "order_lookup_tool",
                            "arguments": (
                                "{\"record_id\": \"ORD0001\"}"
                            ),
                        },
                    }
                ]

            return (
                "No order lookup was necessary for this "
                "knowledge-base question."
            )

        # ------------------------------------------------
        # Composer Agent
        # ------------------------------------------------
        if "composer" in agent_role:

            if "ord0001" in message_lower:
                return (
                    '{'
                    '"answer": "Order ORD0001 has status Returned. '
                    'The order value is INR 2367. The order was created '
                    '8 days ago. It is marked as a delayed shipment, '
                    'with an escalation score of 0.7067, so escalation '
                    'is required.", '
                    '"sources": [], '
                    '"grounded": true, '
                    '"order_id": "ORD0001", '
                    '"escalation_required": true'
                    '}'
                )

            return (
                '{'
                '"answer": "Footwear can be returned within 15 days '
                'of delivery, subject to the applicable return '
                'conditions.", '
                '"sources": ["01_return_window.md"], '
                '"grounded": true, '
                '"order_id": null, '
                '"escalation_required": false'
                '}'
            )

        # ------------------------------------------------
        # Fallback
        # ------------------------------------------------

        return (
            "Based on the available Nykaa information, "
            "I can provide a grounded support response."
        )

    def supports_function_calling(self) -> bool:
        return True
# --------------------------------------------------------
# Task 7: Create CrewAI Agents
# --------------------------------------------------------

mock_llm = MockLLM()


# --------------------------------------------------------
# Retrieval Agent
# --------------------------------------------------------

retrieval_agent = Agent(
    role="Nykaa Knowledge Retrieval Specialist",

    goal=(
        "Find accurate and grounded answers from the "
        "Nykaa knowledge base."
    ),

    backstory=(
        "You specialize in Nykaa policies and support "
        "documentation. You must use the knowledge base "
        "retrieval tool when answering policy questions."
    ),

    tools=[
        RetrievalTool()
    ],

    llm=mock_llm,

    allow_delegation=False,

    verbose=True
)


# --------------------------------------------------------
# Lookup Agent
# --------------------------------------------------------

lookup_agent = Agent(
    role="Nykaa Order Lookup Specialist",

    goal=(
        "Retrieve accurate order information using the "
        "order lookup tool."
    ),

    backstory=(
        "You specialize in checking Nykaa order records. "
        "When an order record ID is provided, use the "
        "order lookup tool to retrieve its status and "
        "escalation information."
    ),

    tools=[
        OrderLookupTool()
    ],

    llm=mock_llm,

    allow_delegation=False,

    verbose=True
)


# --------------------------------------------------------
# Composer Agent
# --------------------------------------------------------

composer_agent = Agent(
    role="Nykaa Customer Support Composer",

    goal=(
        "Create a clear and helpful final customer "
        "support response using the information provided "
        "by the other agents."
    ),

    backstory=(
        "You are an experienced Nykaa customer support "
        "specialist. You combine retrieved policy "
        "information and order information into a concise "
        "customer-friendly response."
    ),

    tools=[],

    llm=mock_llm,

    allow_delegation=False,

    verbose=True
)
# --------------------------------------------------------
# Task 7: Create CrewAI Tasks
# --------------------------------------------------------

retrieval_task = Task(
    description=(
        "Analyze the customer's query: {query}. "
        "If the query is related to Nykaa policies or "
        "knowledge-base information, use the "
        "nykaa_knowledge_base_search tool. "
        "Return the relevant grounded information."
    ),

    expected_output=(
        "Relevant information retrieved from the Nykaa "
        "knowledge base, including the source information "
        "when available."
    ),

    agent=retrieval_agent
)


lookup_task = Task(
    description=(
        "Analyze the customer's query: {query}. "
        "If an order record ID is provided, use the "
        "order_lookup_tool to retrieve the order details. "
        "If no order lookup is required, clearly state "
        "that no order lookup was necessary."
    ),

    expected_output=(
        "Order information including status, order value, "
        "delay information and escalation information, "
        "or a statement that order lookup was not required."
    ),

    agent=lookup_agent
)


composer_task = Task(
    description=(
        "Create the final Nykaa customer support response "
        "for this query: {query}. "
        "Use the information produced by the Retrieval "
        "Agent and Lookup Agent. "
        "Do not invent information."
    ),

    expected_output=(
        "A concise, clear and grounded Nykaa customer "
        "support response."
    ),

    agent=composer_agent
)
# --------------------------------------------------------
# Task 7: Create Crew
# --------------------------------------------------------
composer_task.guardrail = crew_groundedness_guardrail
nykaa_crew = Crew(
    agents=[
        retrieval_agent,
        lookup_agent,
        composer_agent
    ],

    tasks=[
        retrieval_task,
        lookup_task,
        composer_task
    ],

    process=Process.sequential,

    verbose=True
)
# --------------------------------------------------------
# Task 7: CrewAI Demonstration
# --------------------------------------------------------

if __name__ == "__main__":

    print("\n========== TASK 9 STRUCTURED OUTPUT TEST ==========")

    raw_query = "What is the status of order ORD0001?"

    input_guardrail_result = apply_input_guardrails(raw_query)

    print("\n========== INPUT GUARDRAIL RESULT ==========")
    print(input_guardrail_result)

    if not input_guardrail_result["allowed"]:
        print("\nRequest blocked by input guardrail.")
        raise SystemExit

    sanitized_query = input_guardrail_result["text"]

    result = nykaa_crew.kickoff(
        inputs={"query": sanitized_query}
    )
    print("\n========== RAW CREW OUTPUT ==========")
    print(result)

    # ----------------------------------------------------
    # Task 9: Validate Composer Response
    # ----------------------------------------------------

    try:
        structured_response = validate_crew_response(
            result.raw
        )

        print(
            "\n========== VALIDATED PYDANTIC RESPONSE =========="
        )

        print(
            structured_response.model_dump()
        )

        print(
            "\nPydantic validation: PASSED"
        )

    except Exception as error:

        print(
            "\nPydantic validation: FAILED"
        )

        print(error)

        # ----------------------------------------------------
    # Task 9: Invalid Response Validation Test
    # ----------------------------------------------------

    print(
        "\n========== INVALID RESPONSE TEST =========="
    )

    invalid_response = {
        "answer": "Test invalid response",
        "sources": [],
        "grounded": "not-a-boolean",
        "order_id": "ORD0001",
        "escalation_required": True
    }

    try:

        validate_crew_response(
            invalid_response
        )

        print(
            "Invalid response was incorrectly accepted."
        )

    except Exception as error:

        print(
            "Pydantic validation correctly rejected "
            "the invalid response."
        )

        print(error)