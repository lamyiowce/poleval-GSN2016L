import torch
from torch import nn


class WordEmbedder(nn.Module):
    def __init__(self, n_letters, embed_dim):
        super(WordEmbedder, self).__init__()
        self.num_layers = 2
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_letters+1, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size=embed_dim//2, batch_first=True, bidirectional=True, num_layers=self.num_layers)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, input):
        # input = lista słów
        word_lens = (input != 0).sum(dim=1)
        input = input[word_lens > 0]

        embedded = self.embedding(input)
        _, (h, _) = self.lstm(embedded)
        h = h.view(self.num_layers, 2, -1, self.embed_dim//2)
        h = h[-1]
        h = torch.cat([h[0], h[1]], dim=-1)

        return self.fc(h)


class SentenceEmbedder(nn.Module):
    def __init__(self, word_embed_dim, output_embed_dim):
        super(SentenceEmbedder, self).__init__()
        self.num_layers = 2
        self.embed_dim = output_embed_dim
        self.lstm = nn.LSTM(input_size=word_embed_dim, hidden_size=output_embed_dim//2, batch_first=True, bidirectional=True, num_layers=self.num_layers)
        self.fc = nn.Sequential(
            nn.Linear(output_embed_dim, output_embed_dim),
            nn.ReLU(),
            nn.Linear(output_embed_dim, output_embed_dim)
        )

    def forward(self, x):
        # input of shape (B, N_WORDS_IN_SENTENCE, EMBED_DIM)
        _, (h, _) = self.lstm(x)
        h = h.view(self.num_layers, 2, -1, self.embed_dim // 2)
        h = h[-1]
        h = torch.cat([h[0], h[1]], dim=-1)
        return self.fc(h)


class Classifier(nn.Module):
    def __init__(self, layer_sizes, sentence_embed_dim, word_embed_dim):
        super(Classifier, self).__init__()
        layer_sizes = zip([sentence_embed_dim + word_embed_dim] + layer_sizes, layer_sizes + [2])
        self.layers = nn.ModuleList(list(map(lambda n: nn.Linear(n[0], n[1]), layer_sizes)))

    def forward(self, x, words):
        x = torch.cat([x, words], dim=1)
        for layer in self.layers:
            x = layer(x)
            if layer != self.layers[-1]:
                x = torch.relu(x)
        return torch.log_softmax(x, dim=1)


class LanguageModel(nn.Module):
    def __init__(self, n_letters, word_embed_dim, sentence_embed_dim, classifier_layers):
        super(LanguageModel, self).__init__()
        self.word_embeds = WordEmbedder(n_letters, word_embed_dim)
        self.sentence_embeds = SentenceEmbedder(word_embed_dim, sentence_embed_dim)
        self.classifier = Classifier(classifier_layers, sentence_embed_dim, word_embed_dim)
        self.word_embed_dim = word_embed_dim

    def forward(self, sentences, words):
        batch_size, max_sentence_len, max_word_len = sentences.shape
        words_in_sentences = ((sentences != 0).sum(dim=2) != 0).sum(dim=1)
        flattened = sentences.view(-1, max_word_len)
        x = self.word_embeds(flattened)
        unflattened = torch.zeros(batch_size, max_sentence_len, self.word_embed_dim).float().to(x.device)
        s = 0
        for s_idx, s_len in enumerate(words_in_sentences):
            unflattened[s_idx, :s_len] = x[s:s+s_len]
            s += s_len

        x = self.sentence_embeds(unflattened)

        words = self.word_embeds(words)
        x = self.classifier(x, words)
        return x

