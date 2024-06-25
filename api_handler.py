from groq import Groq, AsyncGroq

def get_groq_client(api_key):
    return Groq(api_key=api_key)

def get_async_groq_client(api_key):
    return AsyncGroq(api_key=api_key)

def stream_llm_response(client, model_params, messages):
    try:
        for chunk in client.chat.completions.create(
            model=model_params["model"],
            messages=messages,
            temperature=model_params["temperature"],
            max_tokens=model_params["max_tokens"],
            top_p=model_params["top_p"],
            stream=True,
        ):
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except Exception as e:
        raise APIError(f"An error occurred: {str(e)}")

class APIError(Exception):
    pass