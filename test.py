<<<<<<< HEAD
=======
import anthropic
import google.generativeai as genai
import os


from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

def anthropic_test(msg):
    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key="",
    )
    message = client.messages.create(model="claude-2.0", max_tokens=1024, messages=[{"role": "user", "content": msg}])
    print(message.content)


def google_test(msg):
    genai.configure(api_key="")

    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(msg)
    print(response.text)

def mistral_test(msg):

    api_key = os.environ["MISTRAL_API_KEY"]
    model = "mistral-large-latest"

    client = MistralClient(api_key=api_key)

    chat_response = client.chat(model=model, messages=[ChatMessage(role="user", content="What is the best French cheese?")])

    print(chat_response.choices[0].message.content)

if __name__ == "__main__":
    #
    msg = "你好,你是谁"
    google_test(msg)
>>>>>>> ea99f13 (edit)
