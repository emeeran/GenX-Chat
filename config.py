# # config.py

import os
from persona import PERSONAS
from content_type import CONTENT_TYPES

# Chat settings
MAX_CHAT_HISTORY_LENGTH = 50
DB_PATH = "chat_history.db"
MAX_FILE_CONTENT_LENGTH = 4000
TRUNCATION_ELLIPSIS = "..."

# Voice options
VOICE_OPTIONS = {
    "OpenAI": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    "gTTS": ["en", "ta", "hi"],
}

# Provider options
PROVIDER_OPTIONS = ["Groq", "Google", "OpenAI", "Mistral"]

# Gemini functions
GEMINI_FUNCTIONS = [
    {
        "name": "get_weather",
        "description": "Get weather information for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The name of the city or location.",
                },
                "date": {
                    "type": "string",
                    "description": "The date for which to retrieve the weather (format: YYYY-MM-DD).",
                    "optional": True,
                },
            },
            "required": ["location"],
        },
    },
    # Add more Gemini functions here if needed
]

# Model options for each provider
MODEL_OPTIONS = {
    "Google": [
        "gemini-1.5-pro-latest",
        "gemini-1.0-pro-001",
        "gemini-1.5-flash-latest",
    ],
    "Groq": [
        "llama-3.1-70b-versatile",
        "llama-3.1-405b-reasoning",
        "llama-3.1-8b-instant",
        "llama3-groq-70b-8192-tool-use-preview",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "whisper-large-v3",
    ],
    "OpenAI": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo",
    ],
    "Mistral": [
        "mistral-large-latest",
        "open-mistral-nemo",
        "codestral-latest",
    ],
}

# Maximum token limits for specific models
MAX_TOKEN_LIMITS = {
    "mixtral-8x7b-32768": 32768,
    "llama-3.1-70b-versatile-131072": 131072,
    "gemma2-9b-it": 8192,
}

# Default maximum token limit
DEFAULT_MAX_TOKEN_LIMIT = 4096

# Estimated cost per token (adjust as needed)
COST_PER_TOKEN = 0.0001

# Language options
LANGUAGE_OPTIONS = ["English", "Tamil", "Hindi"]

# Summarization types
SUMMARIZATION_TYPES = [
    "Main Takeaways",
    "Main points bulleted",
    "Concise Summary",
    "Executive Summary",
]

# Color schemes
COLOR_SCHEMES = ["Light", "Dark"]

# Environment variables (make sure to set these in your .env file)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Default provider (based on available API keys)
DEFAULT_PROVIDER = (
    "Google" if GEMINI_API_KEY else
    "Groq" if GROQ_API_KEY else
    "OpenAI" if OPENAI_API_KEY else
    "Mistral" if MISTRAL_API_KEY else
    None
)

# Default model parameters
DEFAULT_MODEL_PARAMS = {
    "model": "gemini-1.0-pro" if GEMINI_API_KEY else "llama-3.1-70b-versatile",
    "max_tokens": 1024,
    "temperature": 1.0,
    "top_p": 1.0,
} 

# import os
# from persona import PERSONAS
# from content_type import CONTENT_TYPES

# # Chat settings
# MAX_CHAT_HISTORY_LENGTH = 50
# DB_PATH = "chat_history.db"
# MAX_FILE_CONTENT_LENGTH = 4000
# TRUNCATION_ELLIPSIS = "..."

# # Voice options
# VOICE_OPTIONS = {
#     "OpenAI": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
#     "gTTS": ["en", "ta", "hi"],
# }

# # Provider options
# PROVIDER_OPTIONS = ["Groq", "Google", "OpenAI", "Mistral"]

# # Gemini functions
# GEMINI_FUNCTIONS = [
#     {
#         "name": "get_weather",
#         "description": "Get weather information for a given location.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "location": {
#                     "type": "string",
#                     "description": "The name of the city or location.",
#                 },
#                 "date": {
#                     "type": "string",
#                     "description": "The date for which to retrieve the weather (format: YYYY-MM-DD).",
#                     "optional": True,
#                 },
#             },
#             "required": ["location"],
#         },
#     },
#     # Add more Gemini functions here if needed
# ]

# # Model options for each provider
# MODEL_OPTIONS = {
#     "Google": [
#         "gemini-1.5-pro-latest",
#         "gemini-1.0-pro-001",
#         "gemini-1.5-flash-latest",
#     ],
#     "Groq": [
#         "llama-3.1-70b-versatile",
#         "llama-3.1-405b-reasoning",
#         "llama-3.1-8b-instant",
#         "llama3-groq-70b-8192-tool-use-preview",
#         "llama3-70b-8192",
#         "mixtral-8x7b-32768",
#         "gemma2-9b-it",
#         "whisper-large-v3",
#     ],
#     "OpenAI": [
#         "gpt-4o-mini",
#         "gpt-4o",
#         "gpt-3.5-turbo",
#     ],
#     "Mistral": [
#         "mistral-large-latest",
#         "open-mistral-nemo",
#         "codestral-latest",
#     ],
# }

# # Maximum token limits for specific models
# MAX_TOKEN_LIMITS = {
#     "mixtral-8x7b-32768": 32768,
#     "llama-3.1-70b-versatile-131072": 131072,
#     "gemma2-9b-it": 8192,
# }

# # Default maximum token limit
# DEFAULT_MAX_TOKEN_LIMIT = 4096

# # Estimated cost per token (adjust as needed)
# COST_PER_TOKEN = 0.0001

# # Language options
# LANGUAGE_OPTIONS = ["English", "Tamil", "Hindi"]

# # Summarization types
# SUMMARIZATION_TYPES = [
#     "Main Takeaways",
#     "Main points bulleted",
#     "Concise Summary",
#     "Executive Summary",
# ]

# # Color schemes
# COLOR_SCHEMES = ["Light", "Dark"]

# # Environment variables (make sure to set these in your .env file)
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# # Default provider (based on available API keys)
# DEFAULT_PROVIDER = (
#     "Google" if GEMINI_API_KEY else
#     "Groq" if GROQ_API_KEY else
#     "OpenAI" if OPENAI_API_KEY else
#     "Mistral" if MISTRAL_API_KEY else
#     None
# )

# # Default model parameters
# DEFAULT_MODEL_PARAMS = {
#     "model": "gemini-1.0-pro" if GEMINI_API_KEY else "llama-3.1-70b-versatile",
#     "max_tokens": 1024,
#     "temperature": 1.0,
#     "top_p": 1.0,
# }

# # Define the CONTENT_ variable
# CONTENT_ = "some_content_value"
