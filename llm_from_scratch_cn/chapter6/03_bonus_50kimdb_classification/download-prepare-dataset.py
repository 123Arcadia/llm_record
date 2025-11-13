import os

import pandas as pd


def dwonload_extract_dataset(path, target_file, dir):
    # if not os.path.exists(dir):
    #     # 目录不存在
    #     if os.path.exists(target_file):
    #         os.remove(target_file)
    # else:
    print(f'{path=} Sikipping DownLoad!')


def load_dataset_to_dataframe(basepath="dataset/aclImdb", labels={"pos": 1, "neg": 0}):
    data_frames = []  # List to store each chunk of DataFrame
    for subset in ("test", "train"):
        for label in ("pos", "neg"):
            path = os.path.join(basepath, subset, label)
            for file in sorted(os.listdir(path)):
                with open(os.path.join(path, file), "r", encoding="utf-8") as infile:
                    # Create a DataFrame for each file and add it to the list
                    data_frames.append(pd.DataFrame({"text": [infile.read()], "label": [labels[label]]}))
    # Concatenate all DataFrame chunks together
    df = pd.concat(data_frames, ignore_index=True)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)  # Shuffle the DataFrame
    return df


def partition_and_save(df, sizes=(35000, 5000, 10000)):
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Get indices for where to split the data
    train_end = sizes[0]
    val_end = sizes[0] + sizes[1]

    # Split the DataFrame
    train = df_shuffled.iloc[:train_end]
    val = df_shuffled.iloc[train_end:val_end]
    test = df_shuffled.iloc[val_end:]

    # Save to CSV files
    train.to_csv("train.csv", index=False)
    val.to_csv("validation.csv", index=False)
    test.to_csv("test.csv", index=False)


if __name__ == '__main__':
    path = 'dataset/aclImdb_v1.tar.gz'
    dwonload_extract_dataset(path, 'aclImdb_v1.tar.gz', 'dataset')
    df = load_dataset_to_dataframe()
    print("Partitioning and saving data frames ...")
    partition_and_save(df)