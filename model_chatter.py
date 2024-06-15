from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="mistralai/mixtral-8x22b-instruct-v0.1",
  messages=[{"role":"user","content":""}],
  temperature=0.5,
  top_p=1,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")