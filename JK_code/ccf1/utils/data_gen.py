import openai

openai.api_key = 'your-api-key'

# 数据增强函数
def data_augmentation(original_abstract):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a creative writer. I want you to write an abstract that is similar in meaning to the given abstract, but expressed with different wording."
            },
            {
                "role": "user",
                "content": f"The given abstract is: '{original_abstract}'"
            }
        ]
    )

    # 返回生成的摘要
    return response['choices'][0]['message']['content']

# 测试数据增强函数
new_abstract = data_augmentation("一种信号的发送方法及基站、用户设备。在一个子帧中为多个用户设备配置的参考信号的符号和数据的符...")
print(new_abstract)
