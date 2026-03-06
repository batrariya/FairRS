import torch
import numpy as np
from tqdm import tqdm

def recall_at_k(model, adj, train_interactions, test_interactions, k=10):

    model.eval()

    with torch.no_grad():

        users_emb, items_emb = model.propagate(adj)

        # Build dictionaries
        train_dict = {}
        test_dict = {}

        for u, i in train_interactions:
            train_dict.setdefault(u, set()).add(i)

        for u, i in test_interactions:
            test_dict.setdefault(u, set()).add(i)

        recalls = []

        for user in test_dict.keys():

            user_embedding = users_emb[user]

            scores = torch.matmul(user_embedding, items_emb.T)

            # Remove training items
            train_items = train_dict.get(user, set())
            scores[list(train_items)] = -1e9

            _, top_k = torch.topk(scores, k)

            recommended = set(top_k.cpu().numpy())
            relevant = test_dict[user]

            hit = len(recommended & relevant)
            recall = hit / len(relevant)

            recalls.append(recall)

        return np.mean(recalls)

def ndcg_at_k(model, adj, train_interactions, test_interactions, k=10):

    model.eval()

    with torch.no_grad():

        users_emb, items_emb = model.propagate(adj)

        train_dict = {}
        test_dict = {}

        for u, i in train_interactions:
            train_dict.setdefault(u, set()).add(i)

        for u, i in test_interactions:
            test_dict.setdefault(u, set()).add(i)

        ndcgs = []

        for user in test_dict.keys():

            user_embedding = users_emb[user]
            scores = torch.matmul(user_embedding, items_emb.T)

            train_items = train_dict.get(user, set())
            scores[list(train_items)] = -1e9

            _, top_k = torch.topk(scores, k)

            recommended = top_k.cpu().numpy()
            relevant = test_dict[user]

            dcg = 0
            for idx, item in enumerate(recommended):
                if item in relevant:
                    dcg += 1 / np.log2(idx + 2)

            ideal_dcg = sum(
                1 / np.log2(i + 2)
                for i in range(min(len(relevant), k))
            )

            if ideal_dcg == 0:
                ndcgs.append(0)
            else:
                ndcgs.append(dcg / ideal_dcg)

        return np.mean(ndcgs)