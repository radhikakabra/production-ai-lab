import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.5-flash-lite"
GREETING_PROMPT = "Say hello in exactly five words."
STORY_PROMPT = "Write a 300-word story about a robot."
AFC_DISABLED = types.AutomaticFunctionCallingConfig(disable=True)


def generate(contents: str, **config_options) -> tuple[types.GenerateContentResponse, float]:
    config = types.GenerateContentConfig(automatic_function_calling=AFC_DISABLED, **config_options)
    start = time.perf_counter()
    response = client.models.generate_content(model=MODEL, contents=contents, config=config)
    latency = time.perf_counter() - start
    return response, latency


def report(title: str, response: types.GenerateContentResponse, latency: float) -> None:
    usage = response.usage_metadata
    finish_reason = response.candidates[0].finish_reason

    print(f"\n=== {title} ===")
    print(f"Latency:         {latency:.2f}s")
    print(f"Finish reason:   {finish_reason.name}")
    print(f"Prompt tokens:   {usage.prompt_token_count}")
    print(f"Output tokens:   {usage.candidates_token_count or 0}")
    print(f"Thinking tokens: {usage.thoughts_token_count or 0}")
    print(f"Total tokens:    {usage.total_token_count}")
    print(f"Text:            {response.text.strip() if response.text else '(no text returned)'}")


def main() -> None:
    response, latency = generate(GREETING_PROMPT)
    report("Greeting (no output limit)", response, latency)

    response, latency = generate(STORY_PROMPT, max_output_tokens=20)
    report("Story (max_output_tokens=20)", response, latency)


if __name__ == "__main__":
    main()
