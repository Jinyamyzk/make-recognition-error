import torch
from torchtext.legacy import data
import torch.optim as optim
from torch import nn
from utils.error_detection_bert import ErrorDetectionBert
from transformers import BertJapaneseTokenizer
from utils.customized_bce import CustomizedBCELoss
from utils.fair_bce import FairBCELoss

import pickle

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

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    print('-----start-------')

    # ネットワークをGPUへ
    net.to(device)
    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # ミニバッチのサイズ
    batch_size = dataloaders_dict["train"].batch_size

    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

     # epochのループ
    for epoch in range(num_epochs):
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数
            epoch_label_len = 0 # epochのラベルが1の数
            iteration = 1

            # データローダーからミニバッチを取り出すループ
            for batch in (dataloaders_dict[phase]):
                # batchはTextとLableの辞書型変数

                

                # GPUが使えるならGPUにデータを送る
                inputs = batch.Text[0].to(device)  # 文章
                attn_mask = torch.where(inputs==0, 0, 1).to(device) # attention maskの作成
                labels = batch.Label.to(dtype=torch.float32, device=device)  # ラベル


                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):

                    # BERTに入力
                    outputs = net(input_ids=inputs, attention_mask=attn_mask)

                    loss = criterion(outputs, labels, attn_mask, device)  # 損失を計算

                    preds = torch.where(outputs < 0.5, -1, 1)  # ラベルを予測

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                            acc = (torch.sum(preds == labels.data)
                                   ).double()/torch.sum(labels.data)
                            print(f"正階数:{torch.sum(preds == labels.data)}, ラベルの数: {torch.sum(labels.data)}")
                            print('イテレーション {} || Loss: {:.4f} || 10iter. || 本イテレーションの正解率：{}'.format(
                                iteration, loss.item(),  acc))

                    iteration += 1

                    # 損失と正解数の合計を更新
                    epoch_loss += loss.item() * batch_size
                    epoch_corrects += torch.sum(preds == labels.data)
                    epoch_label_len += torch.sum(labels.data)

            # epochごとのlossと正解率
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / epoch_label_len.double()

            if phase == "train":
                train_loss_list.append(epoch_loss)
                train_acc_list.append(epoch_acc)
            else:
                valid_loss_list.append(epoch_loss)
                valid_acc_list.append(epoch_acc)

            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs,
                                                                           phase, epoch_loss, epoch_acc))
    
    with open("loss_acc.pickle", "wb") as f:
        loss_acc_dict = {
            "train_loss": train_loss_list,
            "train_acc": train_acc_list,
            "valid_loss": valid_loss_list,
            "valid_acc": valid_acc_list
        }
        pickle.dump(loss_acc_dict, f)

    return net


def main():
    TEXT = data.Field(sequential=True, tokenize=tokenizer_512, use_vocab=False, lower=False,
                                include_lengths=True, batch_first=True, fix_length=MAX_LENGTH, pad_token=0)
    LABEL = data.Field(sequential=True, use_vocab=False, preprocessing=preprocess_label, batch_first=True, 
                                fix_length=MAX_LENGTH, pad_first=True, pad_token=0)                       

    dataset_train, dataset_valid, dataset_test = data.TabularDataset.splits(
        path="livedoor_data", train="train.tsv", validation="valid.tsv",test="test.tsv", format="tsv", fields=[
            ("Text", TEXT), ("Label", LABEL)])
    
    # datasetの長さを確認してみる
    print(dataset_train.__len__())
    print(dataset_valid.__len__())
    print(dataset_test.__len__())
    # datasetの中身を確認してみる
    item = next(iter(dataset_train))
    print(item.Text)
    print("長さ：", len(item.Text))  # 長さを確認 [CLS]から始まり[SEP]で終わる。512より長いと後ろが切れる
    print("ラベルの長さ：", len(item.Label))  
    print("ラベル：", item.Label)

    # DataLoaderを作成します（torchtextの文脈では単純にiteraterと呼ばれています）
    batch_size = 16  # BERTでは16、32あたりを使用する

    dl_train = data.Iterator(
        dataset_train, batch_size=batch_size, train=True)

    dl_valid = data.Iterator(
        dataset_valid, batch_size=batch_size, train=False, sort=False)

    dl_test = data.Iterator(
        dataset_test, batch_size=batch_size, train=False, sort=False)

    # 辞書オブジェクトにまとめる
    dataloaders_dict = {"train": dl_train, "val": dl_valid}
    batch = next(iter(dl_test))
    print(batch)
    print(batch.Text)
    print(batch.Label)

    net = ErrorDetectionBert()
    # 訓練モードに設定
    net.train()

    # 勾配計算を分類アダプターのみ実行
    # 1. まず全てを、勾配計算Falseにしてしまう
    for param in net.parameters():
        param.requires_grad = False
    # 2. BertLayerモジュールの最後を勾配計算ありに変更
    for param in net.bert.encoder.layer[-1].parameters():
        param.requires_grad = True
    # 2. 識別器を勾配計算ありに変更
    for param in net.linear1.parameters():
        param.requires_grad = True
    for param in net.linear2.parameters():
        param.requires_grad = True
    for param in net.linear3.parameters():
        param.requires_grad = True

    # 最適化手法の設定
    optimizer = optim.Adam([
        {'params': net.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
        {'params': net.linear1.parameters(), 'lr': 1e-4},
        {'params': net.linear2.parameters(), 'lr': 1e-4},
        {'params': net.linear3.parameters(), 'lr': 1e-4}
        ])

    # 損失関数の設定
    # criterion = nn.BCELoss(reduction="sum")
    # criterion = CustomizedBCELoss()
    criterion = FairBCELoss()

    # 学習・検証を実行する
    num_epochs = 5
    net_trained = train_model(net, dataloaders_dict,
                            criterion, optimizer, num_epochs=num_epochs)

    # モデルの保存
    torch.save(net_trained.state_dict(), "model/model_trained.pt")

if __name__ == "__main__":
    main()