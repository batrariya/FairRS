import torch
import torch.nn as nn

class LightGCN(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim=64,n_layers=3):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def propagate(self, adj):

        all_embeddings = torch.cat(
            [self.user_embedding.weight,
             self.item_embedding.weight]
        )

        embeddings = [all_embeddings]

        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj, all_embeddings)
            embeddings.append(all_embeddings)

        embeddings = torch.stack(embeddings, dim=1)
        final_embeddings = torch.mean(embeddings, dim=1)

        users, items = torch.split(
            final_embeddings,
            [self.num_users, self.num_items]
        )

        return users, items

    def forward(self, users, pos_items, neg_items, adj):

        users_emb, items_emb = self.propagate(adj)

        u = users_emb[users]
        pos = items_emb[pos_items]
        neg = items_emb[neg_items]

        pos_scores = torch.sum(u * pos, dim=1)
        neg_scores = torch.sum(u * neg, dim=1)

        return pos_scores, neg_scores