# Project Sentinel (HF + OpenAI fallback)

## Local run
1. python -m venv venv
2. venv\Scripts\activate  # or source venv/bin/activate
3. pip install -r requirements.txt
4. python app.py
5. Open http://localhost:7860

## Deploy to Hugging Face Spaces (Gradio)
1. Create a new Space (public or private) and choose 'Gradio'.
2. Push this repo to the HF Space.
3. In Space settings -> Secrets, add `OPENAI_API_KEY` if you want OpenAI summarization.
4. Optionally set hardware to GPU if you want heavy models.
