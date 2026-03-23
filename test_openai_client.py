import openai
from sglang.utils import print_highlight
port = 30000
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

response = client.chat.completions.create(
    # model="qwen/qwen2.5-0.5b-instruct",
    model="/root/models/openbmb/MiniCPM-SALA",
    messages=[
         {"role": "user", "content": "我想去洗车，洗车店离家只有 50 米，请问我是开车去还是走着去？"},
        #{"role": "user", "content": "我开了一家店，朋友过来买电视机没带钱，他向我借了3000，买了一台电视，晚上的时候还给了我3000。我亏了吗？"},
    ],
    temperature=0,
    max_tokens=1024,
)

print_highlight(f"Response: {response}")
