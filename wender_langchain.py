from langchain.llms import OpenAI
openai_api_key = 'b37f67081b8d4a9eae49eebf3476e6fc'
llm = OpenAI(temperature=0.5)

text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))
