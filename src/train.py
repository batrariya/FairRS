import torch
import torch.optim as optim
from tqdm import tqdm

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

def train(model, adj, sampler, epochs=100, batch_size=2048, lr=1e-3):

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