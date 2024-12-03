import numpy as np

class BagOfWord:
    def __init__(self):
        self.vocabulary = []  # Danh sách từ vựng duy nhất

    def fit(self, dataset):

        # Xây dựng từ vựng
        tokenized_data = [text.split() for text in dataset]
        self.vocabulary = sorted(set(word for words in tokenized_data for word in words))

        return self

    def transform(self, dataset):
        if not self.vocabulary:
            raise ValueError("Vocabulary is empty. Call `fit` first.")

        bag_vector = np.zeros((len(dataset), len(self.vocabulary)), dtype=np.int64)

        for idx, text in enumerate(dataset):
            words = text.split()
            for word in words:
                word_idx = self.vocabulary.index(word)
                bag_vector[idx][word_idx] += 1

        return bag_vector

if __name__ == "__main__":
    pass