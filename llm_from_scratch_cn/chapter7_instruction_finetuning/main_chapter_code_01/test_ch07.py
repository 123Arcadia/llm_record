from llm_from_scratch_cn.chapter7_instruction_finetuning.main_chapter_code_01.chapter07_code import \
    custom_collate_draft_1, custom_collate_draft_2, custom_collate_draft_fn
import torch

def test_custom_collate_draft():
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    print(custom_collate_draft_1((inputs_1, inputs_2, inputs_3)))

    print('-'*50)
    print(custom_collate_draft_2((inputs_1, inputs_2, inputs_3)))
    # tensor([[    0,     1,     2,     3,     4],
    #         [    5,     6, 50256, 50256, 50256],
    #         [    7,     8,     9, 50256, 50256]])
    # tensor([[    1,     2,     3,     4, 50256],
    #         [    6, 50256, 50256, 50256, 50256],
    #         [    8,     9, 50256, 50256, 50256]]))
    print('-' * 50)
    print(custom_collate_draft_fn((inputs_1, inputs_2, inputs_3)))
    # (tensor([[    0,     1,     2,     3,     4],
    #         [    5,     6, 50256, 50256, 50256],
    #         [    7,     8,     9, 50256, 50256]]),
    # tensor([[    1,     2,     3,     4, 50256],
    #         [    6, 50256,  -100,  -100,  -100],
    #         [    8,     9, 50256,  -100,  -100]]))
