from torch.utils.data import Dataset


class NarcissisticPostDataset(Dataset):
    def __init__(self, posts, labels, tokenizer, max_token_len):
        self.posts = posts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, index):
        post = str(self.posts[index])
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            post,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": label
        }
