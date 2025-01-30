import os
import openai
from dotenv import load_dotenv
from openai import OpenAI, APIError

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

# Ensure API key is set
if not api_key:
    raise ValueError("Missing OpenAI API key. Make sure OPENAI_API_KEY is set in .env or environment variables.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def list_available_models():
    """
    Fetches and prints the list of available OpenAI models for the given API key.
    """
    try:
        models = client.models.list()
        print("\nAvailable OpenAI Models:")
        for model in models.data:
            print(f"- {model.id}")
    except APIError as e:
        print(f"API Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    list_available_models()
