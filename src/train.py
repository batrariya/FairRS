import torch
import torch.optim as optim
from tqdm import tqdm
from src.evaluation import evaluate_fairness

# def bpr_loss(pos_scores, neg_scores):

#     loss = -torch.mean(
#         torch.log(torch.sigmoid(pos_scores - neg_scores))
#     )

#     return loss

def bpr_loss(pos_scores, neg_scores, model, reg_lambda=1e-4):

    loss = -torch.mean(
        torch.log(torch.sigmoid(pos_scores - neg_scores))
    )

    reg_loss = (
        model.user_embedding.weight.norm(2).pow(2) +
        model.item_embedding.weight.norm(2).pow(2)
    ) / 2

    return loss + reg_lambda * reg_loss

def train(model, adj, sampler, train_interactions, test_interactions, gender_dict, epochs=50, batch_size=128, lr=0.001):

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for _ in tqdm(range(len(sampler.users) // batch_size)):

            users, pos, neg = sampler.sample(batch_size)

            users = torch.LongTensor(users)
            pos = torch.LongTensor(pos)
            neg = torch.LongTensor(neg)

            pos_scores, neg_scores = model(users, pos, neg, adj)

            # loss = bpr_loss(pos_scores, neg_scores)
            loss = bpr_loss(pos_scores, neg_scores, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:

            print("Evaluating...")

            recall, ndcg, male_r, female_r, gru = evaluate_fairness(
                model,
                adj,
                train_interactions,
                test_interactions,
                gender_dict,
                k=20
            )

            print(f"Recall@20: {recall:.4f}")
            print(f"NDCG@20: {ndcg:.4f}")
            print(f"Male Recall@20: {male_r:.4f}")
            print(f"Female Recall@20: {female_r:.4f}")
            print(f"GRU: {gru:.4f}")
            print("-" * 50)