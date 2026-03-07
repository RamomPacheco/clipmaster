from ollamadd import chat
from ollamadd import ChatResponse

response: ChatResponse = chat(
    model="llama3",
    messages=[
        {
            "role": "user",
            "content": "como você pode me ajudar?",
        },
    ],
)
print(response["message"]["content"])
# or access fields directly from the response object
print(response.message.content)
