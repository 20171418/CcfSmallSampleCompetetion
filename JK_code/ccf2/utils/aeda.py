import random
class AEDA:
    """
    随机添加标点
    https://arxiv.org/pdf/2108.13230.pdf
    """

    def __init__(self):
        self.punctuation = punctuation()

    def augmentation(self, text):
        length = int(len(text) * 0.3)
        if length < 2:
            return text
        punc_len = random.randint(1, length)
        puncs = random.choices(self.punctuation, k=punc_len)
        text = list(text)
        for p in puncs:
            text.insert(random.randint(0, len(text) - 1), p)
        return ''.join(text)

def punctuation():
    import string
    en_punctuation = list(string.punctuation)
    zh_punctuation = ['，', '。', '：', '！', '？', '《', '》', '"', '；', "'"]
    return en_punctuation + zh_punctuation


if __name__ == '__main__':
    aeda = AEDA()
    print(aeda.augmentation("一种信号的发送方法及基站、用户设备。在一个子帧中为多个用户设备配置的参考信号的符号和数据的符号"))