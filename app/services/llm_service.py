from collections.abc import Iterator

from huggingface_hub import InferenceClient

from config.settings import settings

client = InferenceClient(api_key=settings.hf_api_token)


def generate(messages: list[dict]) -> str:
    """Call LLM and return full response text."""
    response = client.chat_completion(
        model=settings.hf_model_id,
        messages=messages,
        max_tokens=1024,
        temperature=0.01,
    )
    if getattr(response, "choices", None) and len(response.choices) > 0:
        return response.choices[0].message.content
    return ""


def generate_stream(messages: list[dict]) -> Iterator[str]:
    """Yield tokens one at a time as the LLM generates them."""
    for chunk in client.chat_completion(
        model=settings.hf_model_id,
        messages=messages,
        max_tokens=1024,
        temperature=0.01,
        stream=True,
    ):
        if getattr(chunk, "choices", None) and len(chunk.choices) > 0:
            token = chunk.choices[0].delta.content
            if token:
                yield token
