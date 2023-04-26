from pandas.io.parsers import TextFileReader
from torchdata.dataloader2 import DataLoader2
from torchdata.datapipes.iter import FileOpener, IterableWrapper
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torchdata.datapipes.iter.util.plain_text_reader import CSVParserIterDataPipe
from transformers import BertTokenizer
import pandas as pd

"""
原本用于load data的函数，由于还需要人工标注的数据，要使用论文的脚本，所以还是需要处理成DatasetDict而不是Tensor
所以没有用了
"""
def load_data(split="train", rows=10, train_batch_size=1):
    print(f"loading bert-tiny tokenizer")
    tokenizer = BertTokenizer.from_pretrained('bert-tiny', use_fast=True)

    print(f"loading {split}.csv")
    split_data: pd.DataFrame = pd.read_csv("./datasets/cnn_dailymail/" + split + ".csv", nrows=rows)
    print("tokenizer: ", tokenizer(split_data["article"].tolist()).keys())
    # [chuck_size, [list(article), list(highlight)]]
    articles = (tokenizer(split_data["article"].tolist(), truncation=True, padding=True, max_length=256)["input_ids"])
    highlights = (tokenizer(split_data["article"].tolist(), truncation=True, padding=True, max_length=256)["input_ids"])
    data = DataLoader(dataset=CNNDM(articles, highlights), batch_size=train_batch_size, shuffle=True)
    return data


# def load_data(split="train", train_batch_size=64):
#     url_dp = IterableWrapper([f"{split}.csv"])
#     data_dp = FileOpener(url_dp, mode="b")
#     data: CSVParserIterDataPipe = data_dp.parse_csv()
#     nlp = spacy.load("en_core_web_lg")
#     data.shuffle().batch(batch_size=train_batch_size, drop_last=False)
#     batch_data = DataLoader2(datapipe=data)
#     return batch_data


class CNNDM(Dataset):
    def __init__(self, article, highlights):
        self.article = article
        self.highlights = highlights

    def __getitem__(self, idx):
        assert idx < len(self.article)
        return self.article[idx], self.highlights[idx]

    def __len__(self):
        return len(self.article)
