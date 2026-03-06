import torch

from src.data_loader import prepare_data
from src.graph_builder import build_graph
from src.lightgcn import LightGCN
from src.sampler import BPRSampler
from src.train import train
from src.evaluation import recall_at_k, ndcg_at_k

train_interactions, test_interactions, gender_dict, num_users, num_items = prepare_data()

print("Users:", num_users)
print("Items:", num_items)
print("Train interactions:", len(train_interactions))
print("Test interactions:", len(test_interactions))
print("Example gender:", list(gender_dict.items())[:5])

adj = build_graph(train_interactions, num_users, num_items)

model = LightGCN(num_users, num_items)

sampler = BPRSampler(train_interactions, num_items)

train(model, adj, sampler)

print("Evaluating...")

recall = recall_at_k(model, adj, train_interactions, test_interactions, k=10)
ndcg = ndcg_at_k(model, adj, train_interactions, test_interactions, k=10)

print("Recall@10:", recall)
print("NDCG@10:", ndcg)