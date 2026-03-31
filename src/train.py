import torch
import torch.optim as optim
from tqdm import tqdm
from src.evaluation import evaluate_fairness
from src.bias_analysis import run_bias_analysis 

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

def train(model, adj, sampler, train_interactions, val_interactions, test_interactions, item_category, gender_dict, epochs=30, batch_size=128, lr=0.001):

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

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:

            print("Evaluating on Validation Set...")

            (
                recall,
                precision,
                ndcg,
                male_r,
                female_r,
                male_p,
                female_p,
                male_n,
                female_n,
                gru
            ) = evaluate_fairness(
                model,
                adj,
                train_interactions,
                val_interactions,
                gender_dict,
                item_category,
                k=20
            )

            # print(f"Val Recall@20: {recall:.4f}")
            # print(f"Val NDCG@20: {ndcg:.4f}")
            # print(f"Val Precision@20: {precision:.4f}")
            # print(f"Val Male Recall@20: {male_r:.4f}")
            # print(f"Val Female Recall@20: {female_r:.4f}")
            # print(f"Val Male NDCG@20: {male_n:.4f}")
            # print(f"Val Female NDCG@20: {female_n:.4f}")
            # print(f"Val Male Precision@20: {male_p:.4f}")
            # print(f"Val Female Precision@20: {female_p:.4f}")
            # print(f"Val GRU: {gru:.4f}")
            # print("-" * 50)

    print("Evaluating on Test Set...")
    (
        recall,
        precision,
        ndcg,
        male_r,
        female_r,
        male_p,
        female_p,
        male_n,
        female_n,
        gru
    ) = evaluate_fairness(
        model,
        adj,
        train_interactions,
        test_interactions,
        gender_dict,
        item_category,
        k=20
    )

    # print(f"Test Recall@20: {recall:.4f}")
    # print(f"Test NDCG@20: {ndcg:.4f}")
    # print(f"Test Precision@20: {precision:.4f}")
    # print(f"Test Male Recall@20: {male_r:.4f}")
    # print(f"Test Female Recall@20: {female_r:.4f}")
    # print(f"Test Male NDCG@20: {male_n:.4f}")
    # print(f"Test Female NDCG@20: {female_n:.4f}")
    # print(f"Test Male Precision@20: {male_p:.4f}")
    # print(f"Test Female Precision@20: {female_p:.4f}")
    # print(f"Test GRU: {gru:.4f}")
    # print("=" * 50)

    # 🔥 Bias analysis at the end of training
    model.eval()
    with torch.no_grad():
        users_emb, items_emb = model.propagate(adj)
        user_embs_np = users_emb.cpu().numpy()
        item_embs_np = items_emb.cpu().numpy()
    
    from types import SimpleNamespace
    data_obj = SimpleNamespace(
        num_users=len(gender_dict),
        user_gender=gender_dict,
        item_genre=item_category
    )

    run_bias_analysis(
        model,
        adj,
        train_interactions,
        test_interactions,
        gender_dict,
        item_category,
        name="LightGCN"
    )