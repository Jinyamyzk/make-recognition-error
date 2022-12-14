import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    df_1 = pd.read_csv("data/noised/noised_conversations_1.tsv", sep="\t", names=("text", "label"))
    df_2 = pd.read_csv("data/noised/noised_conversations_2.tsv", sep="\t", names=("text", "label"))
    df = pd.concat([df_1, df_2])
    df = df[~df["label"].isna()]

    print(df.info())

    df_train, df_valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123)
    df_valid, df_test = train_test_split(df_valid_test, test_size=0.5, shuffle=True, random_state=123)
    print(f"train: {len(df_train)}, valid: {len(df_valid)}, test: {len(df_test)}")

    df_train.to_csv("data/train.tsv", sep="\t", index=False, header=False)
    df_valid.to_csv("data/valid.tsv", sep="\t", index=False, header=False)
    df_test.to_csv("data/test.tsv", sep="\t", index=False, header=False)

if __name__ == "__main__":
    main()