import openai
from sglang.utils import print_highlight
port = 31111
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    messages=[
        {"role": "user", "content": "我想去洗车，洗车店离家只有 50 米，请问我是开车去还是走着去？或者别的办法过去？"},
    ],
    temperature=0,
    max_tokens=1024,
)

print_highlight(f"Response: {response}")