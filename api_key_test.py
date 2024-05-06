import openai

def is_api_key_valid(api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        print(client.embeddings.create(input = ["Hello"], model="text-embedding-3-small"))
    except:
        return False
    else:
        return True

is_api_key_valid(api_key = "sk-zIOiLxHvyk72kv8WkFJOCzoCltFQIEAFgNtVAtG")