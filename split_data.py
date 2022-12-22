import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv("livedoor_data2/livedoor_noised_data2.tsv", sep="\t", names=("text", "label"))
    df = df[~df["label"].isna()]

    print(df.info())

    df_train, df_valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123)
    df_valid, df_test = train_test_split(df_valid_test, test_size=0.5, shuffle=True, random_state=123)
    print(f"train: {len(df_train)}, valid: {len(df_valid)}, test: {len(df_test)}")

    df_train.to_csv("livedoor_data2/train.tsv", sep="\t", index=False, header=False)
    df_valid.to_csv("livedoor_data2/valid.tsv", sep="\t", index=False, header=False)
    df_test.to_csv("livedoor_data2/test.tsv", sep="\t", index=False, header=False)

if __name__ == "__main__":
    main()