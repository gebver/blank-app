import os
from openai import OpenAI

def main():
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    print("API_KEY:", api_key)
    print("BASE_URL:", base_url)

    if not api_key or not base_url:
        print("API_KEY or BASE_URL not found in environment variables.")
        return

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model="google/gemma-3-1b-it:free",
            messages=[{"role": "user", "content": "Hello, OpenRouter!"}]
        )
        print("Response from OpenRouter:", response.choices[0].message.content)
    except Exception as e:
        print("Error during API call:", e)

if __name__ == "__main__":
    main()
