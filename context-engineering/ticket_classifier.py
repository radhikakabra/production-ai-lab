import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.5-flash-lite"
AFC_DISABLED = types.AutomaticFunctionCallingConfig(disable=True)

ONE_SENTENCE_RULE = "Always reply in exactly one sentence."
LONG_REQUEST = "Explain RAG in detail, at least 3 paragraphs."

LABEL_RULE = (
    "Classify the support ticket as billing, bug or refund. "
    "Reply with the label in UPPERCASE."
)
FEW_SHOT_EXAMPLES = [
    ("I was charged twice this month.", "billing"),
    ("The app crashes when I click Save.", "bug"),
    ("I want my money back for my last order.", "refund"),
]
TICKETS = [
    "My invoice shows the wrong amount.",
    "The login page is just a blank screen.",
    "Please return the payment for my cancelled plan.",
]


def ask(system_instruction: str, contents) -> str:
    generate_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        automatic_function_calling=AFC_DISABLED,
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=generate_config,
    )
    return response.text.strip()


def show_context(contents: list[types.Content]) -> None:
    print("  What the model sees:")
    for message in contents:
        print(f"    [{message.role}] {message.parts[0].text}")


def system_vs_user() -> None:
    answer = ask(ONE_SENTENCE_RULE, [types.Content(role="user", parts=[types.Part.from_text(text=LONG_REQUEST)])])
    print(f"Answer: {answer}")


def build_few_shot_contents(ticket: str) -> list[types.Content]:
    contents = []
    for text, label in FEW_SHOT_EXAMPLES:
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))
        contents.append(types.Content(role="model", parts=[types.Part.from_text(text=label)]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=ticket)]))
    return contents


def few_shot_vs_instruction() -> None:
    show_context(build_few_shot_contents(TICKETS[0]))
    for ticket in TICKETS:
        contents = build_few_shot_contents(ticket)
        label = ask(LABEL_RULE, contents)
        print(f"Ticket: {ticket}\nLabel: {label}\n")


def main() -> None:
    print("=== System instruction vs user request ===")
    system_vs_user()
    print("\n=== Instruction vs few-shot examples ===")
    few_shot_vs_instruction()


if __name__ == "__main__":
    main()
