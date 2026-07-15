import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables from .env
load_dotenv()

# Initialize the client (reads HF_API_KEY from environment)
client = InferenceClient(api_key=os.getenv("HF_API_KEY"))

def summarize_text(text):
    result = client.summarization(
        text,
        model="facebook/bart-large-cnn"
    )
    return result.summary_text

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