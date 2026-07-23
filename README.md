# LLM Summarizer
A small Python app that summarizes text using an LLM API. Built as part of a Week 1 internship task — covers API integration, prompt engineering, streaming, error handling, structured output validation, and token usage logging.
## Features
- Connects to an LLM API (Hugging Face inference endpoint) to summarize input text
- Structured system + user prompt with few-shot examples, with input text wrapped in `<document>` tags to separate it from instructions
- Streams the response as it's generated
- Retries on API errors, rate limits, timeouts, and connection errors — rate-limit retries respect the server's `Retry-After` header when present, otherwise use jittered exponential backoff
- Validates that the model's output matches a strict JSON schema (`summary`, `key_points`, `word_count`), rejecting unexpected fields and mismatched `word_count` values, and retries if it isn't valid
- Logs prompt/completion/total token usage per request
## Setup
1. Clone the repo:
```bash
   git clone https://github.com/Nurana100/llm-summarizer.git
   cd llm-summarizer
```
2. Create a virtual environment and install dependencies:
```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
```
3. Copy `.env.example` to `.env` and add your real API key:
```bash
   cp .env.example .env
```
   Then edit `.env` and set your key. **Never commit `.env` — only `.env.example` should be in the repo.** If `HF_API_KEY` isn't set, the script exits immediately with a clear error instead of failing later on the first request.
4. Run it:
```bash
   python summarizer.py
```
## Example output
Input: a short paragraph about AI's impact on business.
```
Structured summary (JSON):
[Usage] Prompt tokens: 379 | Completion tokens: 69 | Total tokens: 448
{
"summary": "AI is changing business operations, with companies using it for customer service, data analysis, and automation. This shift creates new jobs and eliminates others, leading to workforce changes.",
"key_points": [
"AI transforms business operations",
"New job categories emerge",
"Workforce transitions across industries"
],
"word_count": 39
}
```
## Notes
- If the model returns malformed JSON (extra text, markdown fences, etc.), the app extracts and re-validates it, retrying up to 3 times before returning a clear error.
- Schema validation is strict: unexpected fields, an out-of-range number of `key_points`, or a `word_count` that doesn't match the actual summary length all count as invalid and trigger a retry.
- Streaming is on by default; pass `stream=False` to `summarize_text_structured()` for a single blocking response instead.
- Token usage is logged after every request as a basic cost/token awareness measure.
