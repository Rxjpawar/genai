import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

text = "hello world"

token = enc.encode(text)

print(token)

token = [24912, 2375]

decode = enc.decode(token)

print(decode)
