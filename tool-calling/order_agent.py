import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.5-flash-lite"
MAX_STEPS = 5
QUESTIONS = [
    "Where is my order A100?",
    "What's the status of order Z999?",
    "What is the capital of France?",
]


def get_order_status(order_id: str) -> dict:
    fake_db = {
        "A100": {"status": "shipped", "eta_days": 2},
        "B200": {"status": "processing", "eta_days": 5},
    }
    return fake_db.get(order_id, {"error": "order not found"})


TOOLS = {"get_order_status": get_order_status}

# Get order Status Tool: A tool that retrieves the status of an order given its order ID.
ORDER_TOOL = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="get_order_status",
        description="Retrieve the status of an order given its order ID.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "order_id": types.Schema(type=types.Type.STRING, description="The ID of the order to check.")
            },
            required=["order_id"]
        )
    )
])

CONFIG = types.GenerateContentConfig(
    tools=[ORDER_TOOL],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


def run(question: str) -> str:
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]

    for step in range(MAX_STEPS):
        print(f"  model call {step + 1}")
        response = client.models.generate_content(model=MODEL, contents=contents, config=CONFIG)

        calls = response.function_calls or []
        if not calls:
            return response.text

        contents.append(response.candidates[0].content)

        function_results = []
        for call in calls:
            print(f"  model asked for tool: {call.name} with args: {call.args}")
            result = TOOLS[call.name](**call.args)
            wrapped_result = types.Part.from_function_response(name=call.name, response={"result": result})
            function_results.append(wrapped_result)

        contents.append(types.Content(role="user", parts=function_results))

    return "Stopped: reached max steps"


def main() -> None:
    for question in QUESTIONS:
        print(f"\nQ: {question}")
        print(f"A: {run(question)}")


if __name__ == "__main__":
    main()
