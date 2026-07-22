from groq import Groq

def main():
    # Step 1: Ask the user for their Groq API key
    api_key = input("Enter your Groq API key: ").strip()

    # Step 2: Initialize the Groq client
    client = Groq(api_key=api_key)

    # Step 3: Read content from the text file
    with open('sample.txt', 'r') as file:
        file_text = file.read().strip()
    

    # Step 4: Send the file content to the LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": file_text}
        ],
        max_completion_tokens=200
    )

    # Step 5: Print the LLM's response
    print("\nLLM Response:\n")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
