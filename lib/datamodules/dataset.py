class NarcissisticPostDataset:
    def __init__(self, posts, labels):
        self.posts = posts
        self.labels = labels

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, index):
        post = str(self.posts[index])
        label = self.labels[index]
        return post, label
