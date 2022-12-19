import torch
from torch import nn
from transformers import BertModel

class ErrorDetectionBert(nn.Module):
    def __init__(self):
        super().__init__()

        model_name = "cl-tohoku/bert-base-japanese"
        self.bert = BertModel.from_pretrained("cl-tohoku/bert-base-japanese")
        self.linear1 = nn.Linear(in_features=768, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=1)

        # 重み初期化処理
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.normal_(self.linear1.bias, 0)
        nn.init.normal_(self.linear2.weight, std=0.02)
        nn.init.normal_(self.linear2.bias, 0)
    
    def forward(self, input_ids, attention_mask=None):
        # attention maskの作成, idが0(pad token)なら0
        attn_mask = torch.where(input_ids > 0, 1, 0)
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        output = self.linear1(output)
        output = self.linear2(output)
        output = torch.sigmoid(output).squeeze(-1)
        return output

if __name__ == "__main__":       
    from transformers import BertJapaneseTokenizer

    model = ErrorDetectionBert()
    tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    text = "まずはベースの日本語BERTモデルを用意します。"
    ids = torch.tensor(tokenizer.encode(text, truncation=True, padding="max_length" ,max_length=512))
    ids = ids.unsqueeze(dim=0) # バッチ次元を追加
    print(ids.size())
    model.eval()
    output = model(ids)
    print(output.size())
    # print(output)

