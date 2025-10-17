from litgpt import LLM
import dotenv
import os
import json

dotenv.load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

llm = LLM.load("allenai/OLMo-2-1124-7B-Instruct")