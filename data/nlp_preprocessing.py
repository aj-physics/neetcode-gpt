import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)

        #setp 1
        combined = positive + negative
        sorted_words = sorted({word for sentence in combined for word in sentence.split()})

        print(positive)
        print(negative)
        print(combined)
        print(combined[0])
        print(combined[1])

        sorted_words_dict = dict.fromkeys(sorted_words, 0)

        for i, word in enumerate(sorted_words):
            sorted_words_dict[word] = i+1

        encoded = [torch.tensor([sorted_words_dict[w] for w in s.split()]) for s in combined]

        return nn.utils.rnn.pad_sequence(encoded, batch_first=True)


