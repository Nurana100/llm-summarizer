import os
import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

load_dotenv()

client = InferenceClient(api_key=os.getenv("HF_API_KEY"))

SYSTEM_PROMPT = """You are a professional text summarizer. Your job is to read the text
provided by the user and produce a clear, concise summary in 2-3 sentences.
Rules:
- Do not add opinions or information not present in the original text.
- Keep the summary factual and neutral in tone.
- Always respond with only the summary, no extra commentary."""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "Summarize this text:\n\nThe city council voted 7-2 to approve funding for a new public library. Construction is expected to begin in spring and finish within 18 months. Supporters say it will improve access to education in underserved neighborhoods."
    },
    {
        "role": "assistant",
        "content": "The city council approved funding for a new public library, with construction starting in spring and finishing in about 18 months. Supporters believe it will improve educational access in underserved areas."
    }
]

MAX_RETRIES = 3
BASE_DELAY = 2  # seconds, doubles each retry (exponential backoff)


def summarize_text(text, stream=True):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": f"Summarize this text:\n\n{text}"}
    ]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if stream:
                full_response = ""
                response_stream = client.chat.completions.create(
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    messages=messages,
                    max_tokens=200,
                    stream=True
                )
                for chunk in response_stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        print(delta, end="", flush=True)
                        full_response += delta
                print()
                return full_response
            else:
                response = client.chat.completions.create(
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    messages=messages,
                    max_tokens=200
                )
                return response.choices[0].message.content

        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)

            # Rate limit or server-side errors: worth retrying
            if status == 429 or (status is not None and status >= 500):
                if attempt < MAX_RETRIES:
                    delay = BASE_DELAY * (2 ** (attempt - 1))
                    print(f"\n[Warning] API busy (status {status}). Retrying in {delay}s... (attempt {attempt}/{MAX_RETRIES})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"\n[Error] Failed after {MAX_RETRIES} attempts due to rate limiting/server error.")
                    return None

            # Auth/permission errors: no point retrying
            elif status in (401, 403):
                print(f"\n[Error] Authentication/permission problem (status {status}). Check your API key and token permissions.")
                return None

            # Any other API error
            else:
                print(f"\n[Error] API error (status {status}): {e}")
                return None

        except TimeoutError:
            if attempt < MAX_RETRIES:
                delay = BASE_DELAY * (2 ** (attempt - 1))
                print(f"\n[Warning] Request timed out. Retrying in {delay}s... (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(delay)
                continue
            else:
                print(f"\n[Error] Timed out after {MAX_RETRIES} attempts.")
                return None

        except Exception as e:
            print(f"\n[Error] Unexpected error: {e}")
            return None

    return None


if __name__ == "__main__":
    sample_text = """
    Artificial intelligence is transforming how businesses operate.
    Companies are using AI for customer service, data analysis, and
    automation of repetitive tasks. This shift is creating new job
    categories while eliminating others, leading to significant
    workforce transitions across industries.
    """
    print("Summary (streaming):")
    result = summarize_text(sample_text, stream=True)

    if result is None:
        print("\nSummarization failed. Please check the error above.")