import asyncio, os, time
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash-lite"

async def ask(question: str) -> str:
    try:
        response = await client.aio.models.generate_content(
            model=MODEL, contents=question
        )
        return response.text
    except Exception as e:
        return f"ERROR: {e}"

async def sequential(questions):
    return [await ask(q) for q in questions]

async def concurrent(questions):
    return await asyncio.gather(*(ask(q) for q in questions))

async def main():
    questions = [
        "Explain async/await in one sentence.",
        "What is a vector embedding in one sentence?",
        "What is an AI agent in one sentence?",
    ]
    start = time.perf_counter()
    await sequential(questions)
    print(f"Sequential: {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    answers = await concurrent(questions)
    print(f"Concurrent: {time.perf_counter() - start:.2f}s")

    for q, a in zip(questions, answers):
        print(f"\nQ: {q}\nA: {a}")

asyncio.run(main())
