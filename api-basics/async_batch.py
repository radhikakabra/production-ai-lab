import asyncio
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.5-flash-lite"
AFC_DISABLED = types.AutomaticFunctionCallingConfig(disable=True)
MAX_CONCURRENT = 2
QUESTIONS = [
    "Explain async/await in one sentence.",
    "What is a vector embedding in one sentence?",
    "What is an AI agent in one sentence?",
]


async def ask(question: str) -> str:
    try:
        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=question,
            config=types.GenerateContentConfig(automatic_function_calling=AFC_DISABLED),
        )
        return response.text
    except Exception as e:
        return f"ERROR: {e}"


async def run_sequential(questions: list[str]) -> list[str]:
    answers = []
    for question in questions:
        answer = await ask(question)
        answers.append(answer)
    return answers


async def run_concurrent(questions: list[str]) -> list[str]:
    tasks = [ask(question) for question in questions]
    return await asyncio.gather(*tasks)


async def ask_limited(semaphore: asyncio.Semaphore, question: str) -> str:
    async with semaphore:
        return await ask(question)


async def run_limited(questions: list[str]) -> list[str]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [ask_limited(semaphore, question) for question in questions]
    return await asyncio.gather(*tasks)


async def main() -> None:
    start = time.perf_counter()
    await run_sequential(QUESTIONS)
    print(f"Sequential: {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    answers = await run_concurrent(QUESTIONS)
    print(f"Concurrent: {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    await run_limited(QUESTIONS)
    print(f"Limited to {MAX_CONCURRENT}: {time.perf_counter() - start:.2f}s")

    for question, answer in zip(QUESTIONS, answers):
        print(f"\nQ: {question}\nA: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
