import os
import time
import json
import re
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import Timeout, ConnectionError as RequestsConnectionError

load_dotenv()

# Client-level timeout (seconds) — applies to each individual request.
REQUEST_TIMEOUT = 30

client = InferenceClient(api_key=os.getenv("HF_API_KEY"), timeout=REQUEST_TIMEOUT)

SYSTEM_PROMPT = """You are a professional text summarizer. Read the text provided by the
user and respond with ONLY a valid JSON object, no extra text before or after it.

The JSON must follow this exact structure:
{
  "summary": "a 2-3 sentence summary of the text",
  "key_points": ["point 1", "point 2", "point 3"],
  "word_count": <integer, number of words in the summary>
}

Rules:
- Output ONLY the JSON object. No explanations, no markdown code fences, no extra commentary.
- Do not add opinions or information not present in the original text.
- key_points should be 2-4 short bullet-style phrases."""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "Summarize this text as JSON:\n\nThe city council voted 7-2 to approve funding for a new public library. Construction is expected to begin in spring and finish within 18 months. Supporters say it will improve access to education in underserved neighborhoods."
    },
    {
        "role": "assistant",
        "content": '{"summary": "The city council approved funding for a new public library, with construction starting in spring and finishing in about 18 months. Supporters believe it will improve educational access in underserved areas.", "key_points": ["Council voted 7-2 to approve funding", "Construction begins spring, ~18 months", "Aims to improve education access"], "word_count": 30}'
    }
]

MAX_RETRIES = 3
BASE_DELAY = 2

# How far word_count is allowed to drift from the real word count of
# `summary` before we treat the JSON as invalid and retry.
WORD_COUNT_TOLERANCE = 3


def extract_json(raw_text):
    """
    Try to extract a valid JSON object from the model's raw text output.
    Handles cases where the model wraps JSON in markdown fences or adds
    extra commentary before/after the JSON.
    """
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def validate_summary_json(data):
    """
    Check that the parsed JSON has the expected fields, correct types,
    and internally consistent values.
    """
    if not isinstance(data, dict):
        return False, "Response is not a JSON object."

    # --- summary ---
    if "summary" not in data or not isinstance(data["summary"], str) or not data["summary"].strip():
        return False, "Missing or invalid 'summary' field."

    # --- key_points ---
    if "key_points" not in data or not isinstance(data["key_points"], list) or len(data["key_points"]) == 0:
        return False, "Missing or invalid 'key_points' field."
    if not (2 <= len(data["key_points"]) <= 4):
        return False, "'key_points' must contain between 2 and 4 items."
    for i, point in enumerate(data["key_points"]):
        if not isinstance(point, str) or not point.strip():
            return False, f"'key_points[{i}]' must be a non-empty string."

    # --- word_count ---
    if "word_count" not in data or not isinstance(data["word_count"], int) or isinstance(data["word_count"], bool):
        return False, "Missing or invalid 'word_count' field."
    if data["word_count"] <= 0:
        return False, "'word_count' must be a positive integer."

    actual_words = len(data["summary"].split())
    if abs(actual_words - data["word_count"]) > WORD_COUNT_TOLERANCE:
        return False, (
            f"'word_count' ({data['word_count']}) does not match the actual "
            f"word count of 'summary' ({actual_words})."
        )

    return True, None


def _log_usage(usage):
    if usage:
        print(f"[Usage] Prompt tokens: {usage.prompt_tokens} | "
              f"Completion tokens: {usage.completion_tokens} | "
              f"Total tokens: {usage.total_tokens}")
    else:
        print("[Usage] Token usage data not returned by this model/provider.")


def _call_model_streaming(messages):
    """
    Calls the API with streaming enabled, prints tokens as they arrive,
    and returns the fully assembled raw text plus usage info (if provided
    on the final chunk).
    """
    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages,
        max_tokens=300,
        stream=True,
    )

    chunks = []
    usage = None
    print("[Streaming] ", end="", flush=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            print(delta, end="", flush=True)
            chunks.append(delta)
        # Some providers attach usage only to the last streamed chunk.
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage:
            usage = chunk_usage
    print()  # newline after stream finishes

    return "".join(chunks), usage


def summarize_text_structured(text, stream=True):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": f"Summarize this text as JSON:\n\n{text}"}
    ]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if stream:
                raw_output, usage = _call_model_streaming(messages)
            else:
                response = client.chat.completions.create(
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    messages=messages,
                    max_tokens=300
                )
                raw_output = response.choices[0].message.content
                usage = getattr(response, "usage", None)

            _log_usage(usage)

            parsed = extract_json(raw_output)
            if parsed is None:
                print(f"[Warning] Attempt {attempt}: model did not return valid JSON. Raw output:\n{raw_output}\n")
                if attempt < MAX_RETRIES:
                    time.sleep(BASE_DELAY)
                    continue
                else:
                    return {"error": "Model failed to return valid JSON after retries.", "raw_output": raw_output}

            is_valid, reason = validate_summary_json(parsed)
            if not is_valid:
                print(f"[Warning] Attempt {attempt}: JSON structure invalid ({reason}). Raw output:\n{raw_output}\n")
                if attempt < MAX_RETRIES:
                    time.sleep(BASE_DELAY)
                    continue
                else:
                    return {"error": f"Model returned malformed JSON structure: {reason}", "raw_output": raw_output}

            return parsed

        except (Timeout, RequestsConnectionError) as e:
            if attempt < MAX_RETRIES:
                delay = BASE_DELAY * (2 ** (attempt - 1))
                print(f"[Warning] Request timed out/connection error. Retrying in {delay}s... ({e})")
                time.sleep(delay)
                continue
            return {"error": f"Failed after {MAX_RETRIES} attempts due to timeout/connection error."}

        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 429 or (status is not None and status >= 500):
                if attempt < MAX_RETRIES:
                    delay = BASE_DELAY * (2 ** (attempt - 1))
                    print(f"[Warning] API busy (status {status}). Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                return {"error": f"Failed after {MAX_RETRIES} attempts (status {status})."}
            elif status in (401, 403):
                return {"error": f"Authentication/permission problem (status {status})."}
            else:
                return {"error": f"API error (status {status}): {e}"}

        except Exception as e:
            return {"error": f"Unexpected error: {e}"}

    return {"error": "Failed after all retries."}


if __name__ == "__main__":
    sample_text = """
    Artificial intelligence is transforming how businesses operate.
    Companies are using AI for customer service, data analysis, and
    automation of repetitive tasks. This shift is creating new job
    categories while eliminating others, leading to significant
    workforce transitions across industries.
    """
    print("Structured summary (JSON):")
    result = summarize_text_structured(sample_text, stream=True)
    print(json.dumps(result, indent=2))
