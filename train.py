import pandas as pd

from transformers import BertJapaneseTokenizer

tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")


df = pd.read_csv("noised_conversations_1.tsv", sep="\t")
print(df.head())