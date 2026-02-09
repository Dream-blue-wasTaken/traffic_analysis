import os
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if api_key:
        # Masking the middle of the key for security while verifying it's loaded
        masked_key = f"{api_key[:10]}...{api_key[-5:]}"
        print(f"SUCCESS: OPENROUTER_API_KEY is loaded.")
        print(f"Key preview: {masked_key}")
    else:
        print("FAILURE: OPENROUTER_API_KEY not found in environment variables.")

if __name__ == "__main__":
    main()
