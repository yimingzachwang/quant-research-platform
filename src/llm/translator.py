import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.llm.prompts import SYSTEM_PROMPT
from src.llm.schemas import LLMDataRequest

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def translate_request(user_input: str) -> LLMDataRequest:

    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_input,
            },
        ],
    )

    raw_text = response.output_text

    data = json.loads(raw_text)

    validated = LLMDataRequest(**data)

    return validated