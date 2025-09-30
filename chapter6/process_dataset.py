import json

# row_data = './autodl-tmp/dataset/pretrain_data/mobvoi_seq_monkey_general_open_corpus.jsonl'
# new_data = './autodl-tmp/dataset/pretrain_data/mobvoi_seq_monkey_general_open_corpus_too_small.jsonl'
row_data = './autodl-tmp/dataset/sft_data/BelleGroup/train_3.5M_CN.json'

new_data = './autodl-tmp/dataset/sft_data/BelleGroup/train_3.5M_CNPsmall.json'


fw = open(new_data, 'w')
i=0
with open(row_data, 'r') as f:
    while i<= 50:
        line = f.readline()
        fw.write(line)
        i+=1
fw.close()