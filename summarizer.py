import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(api_key=os.getenv("HF_API_KEY"))

# System prompt: sets the AI's role and rules
SYSTEM_PROMPT = """You are a professional text summarizer. Your job is to read the text
provided by the user and produce a clear, concise summary in 2-3 sentences.
Rules:
- Do not add opinions or information not present in the original text.
- Keep the summary factual and neutral in tone.
- Always respond with only the summary, no extra commentary."""

# Few-shot examples: show the model what good input/output looks like
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

def summarize_text(text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": f"Summarize this text:\n\n{text}"}
    ]

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages,
        max_tokens=200
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    sample_text = """
    Artificial intelligence is transforming how businesses operate.
    Companies are using AI for customer service, data analysis, and
    automation of repetitive tasks. This shift is creating new job
    categories while eliminating others, leading to significant
    workforce transitions across industries.
    """
    result = summarize_text(sample_text)
    print("Summary:")
    print(result)