import os
import time
import json
import re
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

load_dotenv()

client = InferenceClient(api_key=os.getenv("HF_API_KEY"))

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


def extract_json(raw_text):
    """
    Try to extract a valid JSON object from the model's raw text output.
    Handles cases where the model wraps JSON in markdown fences or adds
    extra commentary before/after the JSON.
    """
    # Try direct parse first
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object embedded in the text (e.g. wrapped in ```json ... ```)
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def validate_summary_json(data):
    """Check that the parsed JSON has the expected fields and types."""
    if not isinstance(data, dict):
        return False, "Response is not a JSON object."
    if "summary" not in data or not isinstance(data["summary"], str) or not data["summary"].strip():
        return False, "Missing or invalid 'summary' field."
    if "key_points" not in data or not isinstance(data["key_points"], list) or len(data["key_points"]) == 0:
        return False, "Missing or invalid 'key_points' field."
    if "word_count" not in data or not isinstance(data["word_count"], int):
        return False, "Missing or invalid 'word_count' field."
    return True, None


def summarize_text_structured(text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": f"Summarize this text as JSON:\n\n{text}"}
    ]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=messages,
                max_tokens=300
            )
            raw_output = response.choices[0].message.content

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
    result = summarize_text_structured(sample_text)
    print(json.dumps(result, indent=2))