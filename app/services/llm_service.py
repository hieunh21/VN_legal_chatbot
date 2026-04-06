from huggingface_hub import InferenceClient
from config.settings import settings

client = InferenceClient(api_key=settings.hf_api_token)


def generate(messages: list[dict]) -> str:
    """Send messages to HuggingFace LLM and return response text."""
    response = client.chat_completion(
        model=settings.hf_model_id,
        messages=messages,
        max_tokens=1024,
    )
    return response.choices[0].message.content
