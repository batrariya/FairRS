import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

from src.data_loader import prepare_data
from src.graph_builder import build_graph
from src.lightgcn import LightGCN
from src.sampler import BPRSampler
from src.train import train
from src.evaluation import evaluate_fairness

train_interactions, val_interactions, test_interactions, gender_dict, num_users, num_items, item_category = prepare_data()

print("Users:", num_users)
print("Items:", num_items)
print("Train interactions:", len(train_interactions))
print("Val interactions:", len(val_interactions))
print("Test interactions:", len(test_interactions))
print("Example gender:", list(gender_dict.items())[:5])

adj = build_graph(train_interactions, num_users, num_items)

model = LightGCN(num_users, num_items, embedding_dim=64)

sampler = BPRSampler(train_interactions, num_items)

train(
    model,
    adj,
    sampler,
    train_interactions,
    val_interactions,
    test_interactions,
    item_category,
    gender_dict
)