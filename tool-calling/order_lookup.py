import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash-lite"


def get_order_status(order_id: str) -> dict:
    fake_db = {
        "A100": {"status": "shipped", "eta_days": 2},
        "B200": {"status": "processing", "eta_days": 5},
    }
    return fake_db.get(order_id, {"error": "order not found"})


order_tool = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="get_order_status",
        description="Look up the status of a customer order by its order ID.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "order_id": types.Schema(
                    type=types.Type.STRING, description="The order ID, e.g. A100"
                )
            },
            required=["order_id"],
        ),
    )
])

config = types.GenerateContentConfig(
    tools=[order_tool],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

TOOLS = {"get_order_status": get_order_status}
MAX_STEPS = 5


def run(question: str) -> str:
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    for _ in range(MAX_STEPS):
        response = client.models.generate_content(
            model=MODEL, contents=contents, config=config
        )
        calls = response.function_calls
        if not calls:
            return response.text
        contents.append(response.candidates[0].content)
        parts = []
        for call in calls:
            print(f"model wants: {call.name}({dict(call.args)})")
            result = TOOLS[call.name](**call.args)
            print(f"tool returned: {result}")
            parts.append(
                types.Part.from_function_response(
                    name=call.name, response={"result": result}
                )
            )
        contents.append(types.Content(role="user", parts=parts))
    return "stopped: too many steps"


if __name__ == "__main__":
    for q in [
        "Where is my order A100?",
        "What's the status of order Z999?",
        "What is the capital of France?",
    ]:
        print(f"\nQ: {q}")
        print(f"A: {run(q)}")
