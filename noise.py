import re
import random
import urllib
import json

from transformers import BertForMaskedLM
from transformers import BertJapaneseTokenizer
import torch

from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher

from pykakasi import kakasi
mykakasi = kakasi()

def init_simstring_db(tokenizer):
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
    pattern = "\(.*?\)|\<.*?\>|《.*?》|\{.*?\}|【|】|#|'|…|“|”|=|‘|’|>|。"
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


tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

text = "院生の方がやりたいって言って、マスターと私と4人くらいで計画の普通に会話するっていうのだったんだけど"
wakati_list = tokenizer.tokenize(text)
idx = random.randint(0, len(wakati_list)-1)
target = wakati_list[idx]
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

wakati_list[idx] = noised_kanji
noised_sentence = "".join(wakati_list).replace("#", "")
print(noised_sentence)





