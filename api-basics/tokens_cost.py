import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.5-flash-lite"
ENGLISH = "Hello, how are you?"
HINDI = "नमस्ते, आप कैसे हैं?"
USER_MESSAGE = "Can you explain how tokens affect the cost of an API call? " * 4
BOT_REPLY = "Every token in the prompt and the answer is counted and billed. " * 4
TURNS = 10
AFC_DISABLED = types.AutomaticFunctionCallingConfig(disable=True)


def count(contents) -> int:
    response = client.models.count_tokens(model=MODEL, contents=contents)
    return response.total_tokens


def compare_languages() -> None:
    print(f"ENGLISH - Tokens: {count(ENGLISH)}, Characters: {len(ENGLISH)}")
    print(f"HINDI - Tokens: {count(HINDI)}, Characters: {len(HINDI)}")


def show_model_limits() -> None:
    model = client.models.get(model=MODEL)
    print(f"Context window: {model.input_token_limit} input tokens, {model.output_token_limit} output tokens")


def simulate_chat() -> None:
    history = []
    for turn in range(1, TURNS + 1):
        history.append(USER_MESSAGE)
        token_count = count(history)
        print(f"Turn {turn}: {token_count} tokens")
        history.append(BOT_REPLY)


def request_too_many_output_tokens() -> None:
    config = types.GenerateContentConfig(
        max_output_tokens=100_000,
        automatic_function_calling=AFC_DISABLED
    )
    generate_response = client.models.generate_content(
        model=MODEL,
        contents="This is a short prompt.",
        config=config
    )
    print(f"Text: {repr(generate_response.text)}")
    print(f"Finish reason: {generate_response.candidates[0].finish_reason.name}")


def main() -> None:
    compare_languages()
    show_model_limits()
    simulate_chat()
    request_too_many_output_tokens()


if __name__ == "__main__":
    main()
