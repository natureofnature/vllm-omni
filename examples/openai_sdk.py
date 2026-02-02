import base64
from datetime import datetime

from openai import OpenAI

client = OpenAI(base_url="http://localhost:40021/v1", api_key="none")
prompt = "A breathtaking wide shot of a stunningly beautiful Xinjiang girl performing a traditional Uyghur dance on the vast Ili Grassland. She is wearing an exquisite Atlas silk dress with vibrant patterns and a traditional \
ornate floral hat. Her long braids and colorful skirt are swirling in mid-air, capturing a sense of graceful motion. The background features rolling green hills, blooming wildflowers, and distant snow-capped Tianshan \
mountains under the soft, golden glow of a sunset. Cinematic lighting, hyper-realistic, 8k resolution, ethereal atmosphere."
# prompt="On the grassland, an injured wolf is running after a young boy who is playing with a puppy. 16:9 image"

response = client.images.generate(
    model="/home/fq9hpsacuser01/models/Qwen-Image-2512/",
    prompt=prompt,
    n=1,
    size="1024x1024",  # 使用 size 参数设置尺寸，格式为 "WIDTHxHEIGHT"
    response_format="b64_json",
    # seed=1142  # seed 可以直接作为参数传递，不需要放在 extra_body 中
)

# 保存图片
for idx, image in enumerate(response.data):
    # 解码 base64 数据
    image_data = base64.b64decode(image.b64_json)

    # 生成文件名（使用时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_image_{timestamp}_{idx}.png"

    # 保存图片
    with open(filename, "wb") as f:
        f.write(image_data)

    print(f"图片已保存为: {filename}")
