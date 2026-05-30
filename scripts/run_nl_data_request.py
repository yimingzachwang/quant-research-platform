from src.llm.translator import translate_request


query = input("Enter request: ")

request = translate_request(query)

print(request)