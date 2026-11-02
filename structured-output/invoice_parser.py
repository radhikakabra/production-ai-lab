import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError, field_validator

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.5-flash-lite"
MAX_ATTEMPTS = 3
AFC_DISABLED = types.AutomaticFunctionCallingConfig(disable=True)
SAMPLES = [
    "Bill from Acme Cloud Services, 3 March 2026. Hosting 1,200.50, support 300. "
    "Total due: 1,500.50 USD.",
    "INVOICE // Bright Print Co // 12/07/26 // 2x banners @ 40, 1x poster @ 15 // TOTAL: EUR 95",
    "Lunch on Friday? No bill attached.",
]


class Invoice(BaseModel):
    contains_invoice: bool = Field(
        description="True if the text contains an invoice, false otherwise."
    )
    vendor: str | None = Field(description="Vendor name, or null if not stated")
    invoice_date: str | None = Field(description="Invoice date in YYYY-MM-DD format, or null if not stated")
    total_amount: float | None = Field(description="Total amount of the invoice, or null if not stated")
    currency: str | None = Field(description="Currency of the total amount as a 3-letter code, or null if not stated")

    @field_validator("total_amount")
    @classmethod
    def check_total(cls, value):
        if value is not None and value <= 0:
            raise ValueError("total_amount must be positive")
        return value

    @field_validator("currency")
    @classmethod
    def check_currency(cls, value):
        if value is not None:
            if len(value) != 3 or not value.isalpha():
                raise ValueError("currency must be a 3-letter code")
            return value.upper()
        return value


def extract(text: str) -> Invoice:
    error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = (
            "Extract the invoice fields from the text below. If it is not an invoice, "
            "set contains_invoice to false and every other field to null. "
            "Never guess values that are not in the text.\n\n"
            f"{text}"
        )
        if error:
            prompt += f"\n\nPrevious attempt failed with error: {error}. Please fix it."

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Invoice,
                automatic_function_calling=AFC_DISABLED,
            ),
        )

        try:
            return Invoice.model_validate_json(response.text)
        except ValidationError as e:
            error = str(e)
            print(f"Attempt {attempt} failed with error: {error}")

    raise ValueError(f"Gave up after {MAX_ATTEMPTS} attempts: {error}")


def main() -> None:
    for text in SAMPLES:
        print(f"\nINPUT: {text}")
        try:
            invoice = extract(text)
        except ValueError as e:
            print(f"FAILED: {e}")
            continue
        if invoice.contains_invoice:
            print(invoice.model_dump_json(indent=2))
        else:
            print("-> no invoice found")


if __name__ == "__main__":
    main()
