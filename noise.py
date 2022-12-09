from os import sep
import re
import random
from unittest import result
import urllib
import json
import glob
import pandas as pd
from tqdm import tqdm

from transformers import BertForMaskedLM
from transformers import BertJapaneseTokenizer
import torch

from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher

from pykakasi import kakasi
mykakasi = kakasi()
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

def init_simstring_db(tokenizer):
    """
    Adding the vocablary of BERT tokenizer to simstring data base
    """
    db = DictDatabase(CharacterNgramFeatureExtractor(1))
    ids = range(tokenizer.vocab_size)
    ids = ids[5:]
    for id in ids:
        word = tokenizer.convert_ids_to_tokens([id])[0]
        word = word.replace("#", "")
        word_yomi = kanji_to_hiragana(word)
        db.add(word_yomi)
    return db

def remove_symbol(text):
    text = re.sub(",", "", text)
    text = re.sub("\[","(",text)
    text = re.sub("\]",")",text)
    pattern = "\(.*?\)|\<.*?\>|《.*?》|\{.*?\}|【|】|#|'|…|“|”|=|‘|’|>|、|。|\?|「|」|『|』"
    text = re.sub(pattern, "", text)
    return text

def hiragana_to_kanji(word_yomi):
    url = "http://www.google.com/transliterate?"
    param = {'langpair':'ja-Hira|ja','text':word_yomi}
    paramStr = urllib.parse.urlencode(param)
    readObj = urllib.request.urlopen(url + paramStr)
    response = readObj.read()
    data = json.loads(response)
    fixed_data = json.loads(json.dumps(data[0], ensure_ascii=False))
    return random.choice(fixed_data[1])

def kanji_to_hiragana(word):
    result = mykakasi.convert(word)
    return result[0]["hira"]

def is_not_kanji(word):
    not_kanji = re.compile(r'[あ-ん]+|[ア-ン]+|[ｱ-ﾝ]+|[0-9０-９]+')
    return not_kanji.search(word)

def noise(text, searcher):
    """
    noise text.
    return noised text and label.
    """
    if len(text) <= 2:
        return pd.Series(["", ""])
    text_split = tokenizer.tokenize(text)
    idx = random.randint(0, len(text_split)-1)
    target = text_split[idx].replace("#", "")
    target_yomi = kanji_to_hiragana(target)
    result = searcher.search(target_yomi, 0.8)
    noised = random.choice(result)
    if noised == "":
        return pd.Series(["", ""])
    print(f"target {target}, noised: {noised}")
    noised_kanji = hiragana_to_kanji(noised)
    noised_kanji_token = tokenizer.tokenize(noised_kanji)
    if noised_kanji_token[0] == "[UNK]":
        return pd.Series(["", ""])
    len_noised = len(noised_kanji_token) # 変換後の単語がtokenizerで複数tokenに分割された場合に備える
    text_split[idx] = noised_kanji
    noised_sentence = "".join(text_split).replace("#", "")
    noised_sentence_split = tokenizer.tokenize(noised_sentence)
    label = [0] * len(noised_sentence_split)
    try:
        for i in range(len_noised):
            label[idx] = 1
            idx += 1
    except:
        # 変換によりトークン数が減ってしまった場合など
        original_split = tokenizer.tokenize(text)
        print(f"original len: {len(original_split)}")
        print(f"noised len: {len(noised_sentence_split)}")
        print(f"original: {original_split}")
        print(f"noised: {noised_sentence_split}")
        return pd.Series(["", ""])
    return pd.Series([noised_sentence, label])

def concat_pre_post_text(input, df):
    """
    concat a text with the previous and next 5 sentences.
    If the number of tokens exceeds 254 ( 256 - ([CLS]+[SEP]) ), trim the same amount from the pre and next sentences.
    """
    idx = input[0]
    text = input[1]
    label = input[2]
    if text == "":
        return pd.Series(["", ""])
    text_split = tokenizer.tokenize(text)
    pre_idx = idx - 5 if idx >= 5 else 0
    post_idx = idx + 5 if len(df) > idx + 5 else len(df)
    pre_text = "".join(df.iloc[pre_idx:idx, 2].to_list()) # second column is "content"
    post_text = "".join(df.iloc[idx:post_idx, 2].to_list())
    pre_text_split = tokenizer.tokenize(pre_text)
    post_text_split = tokenizer.tokenize(post_text)
    total_len = len(pre_text_split) + len(post_text_split) + len(text_split)
    if total_len > 254:
       surplus = total_len - 254 
       pre_text_split = pre_text_split[surplus//2+1:] # delete one more if surplus is as an odd
       post_text_split = post_text_split[:-surplus//+1]
       pre_text = "".join(pre_text_split).replace("#", "")
       post_text = "".join(post_text_split).replace("#", "")
    text_concat = pre_text + text + post_text
    



    return ""


def main():
    db = init_simstring_db(tokenizer)
    searcher = Searcher(db, CosineMeasure())

    files = glob.glob(f"sample_data/**/**/*.xlsx")
    # files = glob.glob(f"btsjcorpus_ver_march_2022_1-29_{folder_num}/**/**/*.xlsx")
    for file in tqdm(files, desc="[Loading excel]"):
        df = pd.read_excel(file,index_col=None,names=["raw_content"],skiprows=[0,1],usecols=[7])
        df["content"] = df["raw_content"].apply(remove_symbol)
        df = df[df["content"]!=""]
        df = df[:2]
        df.reset_index(inplace=True)
        df["index"] = range(len(df))
        print(df)
        df[["noised_content", "label_simple"]] = df["content"].apply(noise, args=(searcher,))
        print(df.iloc[:, 2])
        break
        df[["noised_content_concat", "label_conat"]] = df[["index", "noised_content", "label_simple"]] \
                                                            .apply(concat_pre_post_text, args=(df,))
        df[["content", "noised_content", "label"]].to_csv("noised.tsv", sep="\t", index=False, header=False, mode="a")
        

if __name__ == "__main__":
    main()




"""
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

text = "院生の方がやりたいって言って、マスターと私と4人くらいで計画の普通に会話するっていうのだったんだけど"
text_split = tokenizer.tokenize(text)
print(text_split)
idx = random.randint(0, len(text_split)-1)
target = text_split[idx].replace("#", "")
print(target)
target_yomi = kanji_to_hiragana(target)

db = init_simstring_db(tokenizer)
searcher = Searcher(db, CosineMeasure())
result = searcher.search(target_yomi, 0.8)
print(result)
noised = random.choice(result)
noised_kanji = hiragana_to_kanji(noised)
print(noised_kanji)
len_noised = len(tokenizer.tokenize(noised_kanji))
print(len_noised)

text_split[idx] = noised_kanji
noised_sentence = "".join(text_split).replace("#", "")
print(noised_sentence)

noised_sentenc_split = tokenizer.tokenize(noised_sentence)

label = [0] * len(noised_sentenc_split)
for i in range(len_noised):
    label[idx] = 1
    idx += 1
print(noised_sentenc_split)
print(label)
"""





