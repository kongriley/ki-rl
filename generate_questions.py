import dotenv
import os

import openai

dotenv.load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

document_name = "2025_nba_playoffs"

with open(f"data/{document_name}.md", "r") as f:
    data_content = f.read()

message = f"""
Using the following data, generate 10 multiple choice questions about the data.
The questions should be about the data, and the answers should be in the data.

The questions should be in the following format:

```
# Question 1
Question: [question] A) [answer] B) [answer] C) [answer] D) [answer]
Answer: B
```

# Data
{data_content}
"""

response = client.responses.create(
    model="gpt-5",
    input=message,
)

with open(f"data/{document_name}_questions.md", "w") as f:
    f.write(response.output_text)