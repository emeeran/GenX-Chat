import os
import openai
from dotenv import load_dotenv

# Your existing code
load_dotenv()
# # Set API key as an environment variable
# os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

load_dotenv()

# API Keys
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
openai.api_key = os.environ["OPENAI_API_KEY"]
client = openai.Client()

# Print API key for verification
print(openai.api_key)