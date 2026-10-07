import os
from datetime import date

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import (
    BaseModel, Field, ValidationError, field_validator, model_validator,
)

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash-lite"


class Invoice(BaseModel):
    contains_invoice: bool = Field(
        description="True only if the text actually contains an invoice or bill."
    )
    vendor: str | None = Field(description="Vendor name, or null if not stated")
    invoice_date: str | None = Field(description="YYYY-MM-DD, or null if not stated")
    total_amount: float | None = Field(description="Total, or null if not stated")
    currency: str | None = Field(description="3-letter currency code, or null")
    line_items: list[str]

    @field_validator("invoice_date")
    @classmethod
    def check_date(cls, v):
        if v is not None:
            date.fromisoformat(v)
        return v

    @field_validator("total_amount")
    @classmethod
    def check_total(cls, v):
        if v is not None and v <= 0:
            raise ValueError("total_amount must be positive")
        return v

    @field_validator("currency")
    @classmethod
    def check_currency(cls, v):
        if v is None:
            return v
        if len(v) != 3 or not v.isalpha():
            raise ValueError("currency must be a 3-letter code")
        return v.upper()

    @model_validator(mode="after")
    def check_complete(self):
        if self.contains_invoice:
            required = ("vendor", "invoice_date", "total_amount", "currency")
            missing = [n for n in required if getattr(self, n) is None]
            if missing:
                raise ValueError(f"invoice is missing: {missing}")
        return self


def extract(text: str, max_attempts: int = 3) -> Invoice:
    last_error = None
    for attempt in range(1, max_attempts + 1):
        prompt = (
            "Extract the invoice fields from the text below. If the text does not "
            "contain an invoice, set contains_invoice to false and use null for "
            "every other field. Never guess values that are not in the text.\n\n"
            f"{text}"
        )
        if last_error:
            prompt += f"\n\nYour previous answer failed validation:\n{last_error}\nFix it."
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Invoice,
            ),
        )
        try:
            return Invoice.model_validate_json(response.text)
        except ValidationError as e:
            last_error = str(e)
            print(f"  attempt {attempt} failed validation, retrying...")
    raise ValueError(f"gave up after {max_attempts} attempts: {last_error}")


if __name__ == "__main__":
    samples = [
        "Hi, attached is the bill from Acme Cloud Services dated 3rd March 2026. "
        "We used hosting (1,200.50) and support (300). Total due: 1,500.50 USD.",
        "INVOICE // Bright Print Co // 12/07/26 // 2x banners @ 40, 1x poster @ 15 // "
        "TOTAL: EUR 95",
        "Hey, just checking in about lunch on Friday, no bill attached.",
    ]
    for s in samples:
        print(f"\nINPUT: {s[:70]}...")
        try:
            inv = extract(s)
            if inv.contains_invoice:
                print(inv.model_dump_json(indent=2))
            else:
                print("-> no invoice found")
        except ValueError as e:
            print(f"FAILED: {e}")
