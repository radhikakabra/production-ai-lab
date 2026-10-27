import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.5-flash-lite"
PROMPT = "Suggest one name for a coffee shop. Reply with only the name."
TEMPERATURES = [0.0, 1.0, 2.0]
RUNS_PER_TEMPERATURE = 3
AFC_DISABLED = types.AutomaticFunctionCallingConfig(disable=True)


def ask(temperature: float) -> str:
    config = types.GenerateContentConfig(
        temperature=temperature,
        automatic_function_calling=AFC_DISABLED
    )
    response = client.models.generate_content(model=MODEL, contents=PROMPT, config=config)
    return response.text.strip()


GREETING_PROMPT = "Say hello in exactly five words."


def ask_with_stop(stop: str) -> None:
    config = types.GenerateContentConfig(
        automatic_function_calling=AFC_DISABLED,
        stop_sequences=[stop]
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=GREETING_PROMPT,
        config=config
    )

    print(f"Text: {repr(response.text)}")
    print(f"Finish reason: {response.candidates[0].finish_reason.name}")


def main() -> None:
    for temperature in TEMPERATURES:
        names = []
        for _ in range(RUNS_PER_TEMPERATURE):
            names.append(ask(temperature))
        print(f"Temperature {temperature}: {names} ({len(set(names))} unique)")

    ask_with_stop(",")


if __name__ == "__main__":
    main()