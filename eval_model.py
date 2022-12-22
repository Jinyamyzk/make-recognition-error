import torch
from torchtext.legacy import data
from utils.error_detection_bert import ErrorDetectionBert
from transformers import BertJapaneseTokenizer
from utils.fair_bce import FairBCELoss
from tqdm import tqdm
import argparse

tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
MAX_LENGTH = 512

def tokenizer_512(input_text):
    """torchtextのtokenizerとして扱えるように、512単語のpytorchでのencodeを定義。ここで[0]を指定し忘れないように"""
    return tokenizer.encode(input_text, max_length=MAX_LENGTH, truncation=True, return_tensors='pt')[0]

def preprocess_label(label):
    """ラベルの要素が文字列になっているので数字に直す"""
    table = str.maketrans({
    "[": "",
    "]": "",
    ",": "",
    })
    label = [int(l.translate(table)) for l in label]
    label = [0] + label + [0] # [CLS]と[SEP]の分のラベルを追加する
    return torch.tensor(label)


def main(model_path):
    TEXT = data.Field(sequential=True, tokenize=tokenizer_512, use_vocab=False, lower=False,
                                    include_lengths=True, batch_first=True, fix_length=MAX_LENGTH, pad_token=0)
    LABEL = data.Field(sequential=True, use_vocab=False, preprocessing=preprocess_label, batch_first=True, 
                                fix_length=MAX_LENGTH, pad_first=True, pad_token=0)                       

    dataset_train, dataset_valid, dataset_test = data.TabularDataset.splits(
        path="livedoor_data", train="train.tsv", validation="valid.tsv",test="test.tsv", format="tsv", fields=[
            ("Text", TEXT), ("Label", LABEL)])

    # DataLoaderを作成します（torchtextの文脈では単純にiteraterと呼ばれています）
    batch_size = 16  # BERTでは16、32あたりを使用する

    dl_test = data.Iterator(
        dataset_test, batch_size=batch_size, train=False, sort=False)

    # テストデータでの正解率を求める
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net_trained = ErrorDetectionBert()
    state_dict = torch.load(model_path)
    net_trained.load_state_dict(state_dict)
    net_trained.eval()   # モデルを検証モードに
    net_trained.to(device)  # GPUが使えるならGPUへ送る

    criterion = FairBCELoss()

    # epochの正解数を記録する変数
    epoch_corrects = 0
    epoch_label_len = 0

    for batch in tqdm(dl_test):  # testデータのDataLoader
        # batchはTextとLableの辞書オブジェクト
        # GPUが使えるならGPUにデータを送る
        inputs = batch.Text[0].to(device)  # 文章
        attn_mask = torch.where(inputs==0, 0, 1).to(device) # attention maskの作成
        labels = batch.Label.to(device)  # ラベル

        # 順伝搬（forward）計算
        with torch.set_grad_enabled(False):

            # BertForLivedoorに入力
            # BERTに入力
            outputs = net_trained(input_ids=inputs, attention_mask=attn_mask)

            loss = criterion(outputs, labels, attn_mask, device)  # 損失を計算

            preds = torch.where(outputs < 0.5, -1, 1)  # ラベルを予測
            # 損失と正解数の合計を更新
            epoch_corrects += torch.sum(preds == labels.data)
            epoch_label_len += torch.sum(labels.data)
    # 正解率
    epoch_acc = epoch_corrects.double() / epoch_label_len.double()

    print('テストデータ{}個での正解率：{:.4f}'.format(len(dl_test.dataset), epoch_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    args = parser.parse_args()
    main(args.model_path)
