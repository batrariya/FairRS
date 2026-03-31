# import torch
# import numpy as np
# from tqdm import tqdm

# # def recall_at_k(model, adj, train_interactions, test_interactions, k=10):

# #     model.eval()

# #     with torch.no_grad():

# #         users_emb, items_emb = model.propagate(adj)

# #         # Build dictionaries
# #         train_dict = {}
# #         test_dict = {}

# #         for u, i in train_interactions:
# #             train_dict.setdefault(u, set()).add(i)

# #         for u, i in test_interactions:
# #             test_dict.setdefault(u, set()).add(i)

# #         recalls = []

# #         for user in test_dict.keys():

# #             user_embedding = users_emb[user]

# #             scores = torch.matmul(user_embedding, items_emb.T)

# #             # Remove training items
# #             train_items = train_dict.get(user, set())
# #             scores[list(train_items)] = -1e9

# #             _, top_k = torch.topk(scores, k)

# #             recommended = set(top_k.cpu().numpy())
# #             relevant = test_dict[user]

# #             hit = len(recommended & relevant)
# #             recall = hit / len(relevant)

# #             recalls.append(recall)

# #         return np.mean(recalls)

# # def ndcg_at_k(model, adj, train_interactions, test_interactions, k=10):

# #     model.eval()

# #     with torch.no_grad():

# #         users_emb, items_emb = model.propagate(adj)

# #         train_dict = {}
# #         test_dict = {}

# #         for u, i in train_interactions:
# #             train_dict.setdefault(u, set()).add(i)

# #         for u, i in test_interactions:
# #             test_dict.setdefault(u, set()).add(i)

# #         ndcgs = []

# #         for user in test_dict.keys():

# #             user_embedding = users_emb[user]
# #             scores = torch.matmul(user_embedding, items_emb.T)

# #             train_items = train_dict.get(user, set())
# #             scores[list(train_items)] = -1e9

# #             _, top_k = torch.topk(scores, k)

# #             recommended = top_k.cpu().numpy()
# #             relevant = test_dict[user]

# #             dcg = 0
# #             for idx, item in enumerate(recommended):
# #                 if item in relevant:
# #                     dcg += 1 / np.log2(idx + 2)

# #             ideal_dcg = sum(
# #                 1 / np.log2(i + 2)
# #                 for i in range(min(len(relevant), k))
# #             )

# #             if ideal_dcg == 0:
# #                 ndcgs.append(0)
# #             else:
# #                 ndcgs.append(dcg / ideal_dcg)

# #         return np.mean(ndcgs)

# def evaluate_fairness(model, adj, train_interactions, test_interactions, gender_dict, k=20):

#     model.eval()

#     with torch.no_grad():

#         users_emb, items_emb = model.propagate(adj)

#         train_dict = {}
#         test_dict = {}

#         for u, i in train_interactions:
#             train_dict.setdefault(u, set()).add(i)

#         for u, i in test_interactions:
#             test_dict.setdefault(u, set()).add(i)

#         recalls = []
#         ndcgs = []
#         precisions = []

#         male_recalls = []
#         female_recalls = []
#         male_ndcgs = []
#         female_ndcgs = []
#         male_precisions = []
#         female_precisions = []

#         for user in test_dict.keys():

#             user_embedding = users_emb[user]
#             scores = torch.matmul(user_embedding, items_emb.T)

#             # Remove training items
#             train_items = train_dict.get(user, set())
#             scores[list(train_items)] = -1e9

#             _, top_k = torch.topk(scores, k)

#             recommended = top_k.cpu().numpy()
#             relevant = test_dict[user]

#             # ---------- Recall & Precision ----------
#             hit = len(set(recommended) & relevant)
#             recall = hit / len(relevant)
#             precision = hit / k
            
#             recalls.append(recall)
#             precisions.append(precision)

#             # ---------- NDCG ----------
#             dcg = 0
#             for idx, item in enumerate(recommended):
#                 if item in relevant:
#                     dcg += 1 / np.log2(idx + 2)

#             ideal_dcg = sum(
#                 1 / np.log2(i + 2)
#                 for i in range(min(len(relevant), k))
#             )

#             ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
#             ndcgs.append(ndcg)

#             # ---------- Group Split ----------
#             if gender_dict[user] == 0:
#                 male_recalls.append(recall)
#                 male_ndcgs.append(ndcg)
#                 male_precisions.append(precision)
#             else:
#                 female_recalls.append(recall)
#                 female_ndcgs.append(ndcg)
#                 female_precisions.append(precision)

#         # ---------- Final Metrics ----------
#         metrics = {
#             'recall': np.mean(recalls) if recalls else 0,
#             'ndcg': np.mean(ndcgs) if ndcgs else 0,
#             'precision': np.mean(precisions) if precisions else 0,
#             'male_recall': np.mean(male_recalls) if male_recalls else 0,
#             'female_recall': np.mean(female_recalls) if female_recalls else 0,
#             'male_ndcg': np.mean(male_ndcgs) if male_ndcgs else 0,
#             'female_ndcg': np.mean(female_ndcgs) if female_ndcgs else 0,
#             'male_precision': np.mean(male_precisions) if male_precisions else 0,
#             'female_precision': np.mean(female_precisions) if female_precisions else 0,
#         }
        
#         metrics['gru_recall'] = abs(metrics['male_recall'] - metrics['female_recall'])
#         metrics['gru_ndcg'] = abs(metrics['male_ndcg'] - metrics['female_ndcg'])
#         metrics['gru_precision'] = abs(metrics['male_precision'] - metrics['female_precision'])

#         return metrics

from collections import defaultdict
import torch
import numpy as np

def evaluate_fairness(model, adj, train_interactions, test_interactions, gender_dict, item_category, k=20):

    model.eval()

    with torch.no_grad():

        users_emb, items_emb = model.propagate(adj)

        train_dict = {}
        test_dict = {}

        for u, i in train_interactions:
            train_dict.setdefault(u, set()).add(i)

        for u, i in test_interactions:
            test_dict.setdefault(u, set()).add(i)

        recalls, precisions, ndcgs = [], [], []

        male_recalls, female_recalls = [], []
        male_precisions, female_precisions = [], []
        male_ndcgs, female_ndcgs = [], []

        # Exposure
        all_genres = set()
        for genres_list in item_category.values():
            all_genres.update(genres_list)

        male_exposure = {g: 0 for g in all_genres}
        female_exposure = {g: 0 for g in all_genres}
        male_total = 0
        female_total = 0

        for user in test_dict.keys():

            user_embedding = users_emb[user]
            scores = torch.matmul(user_embedding, items_emb.T)

            train_items = train_dict.get(user, set())
            scores[list(train_items)] = -1e9

            _, top_k = torch.topk(scores, k)

            recommended = top_k.cpu().numpy()
            relevant = test_dict[user]

            # ---------- Recall ----------
            hit = len(set(recommended) & relevant)
            recall = hit / len(relevant)

            # ---------- Precision ----------
            precision = hit / k

            # ---------- NDCG ----------
            dcg = 0
            for idx, item in enumerate(recommended):
                if item in relevant:
                    dcg += 1 / np.log2(idx + 2)

            ideal_dcg = sum(
                1 / np.log2(i + 2)
                for i in range(min(len(relevant), k))
            )

            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0

            # Store overall
            recalls.append(recall)
            precisions.append(precision)
            ndcgs.append(ndcg)

            # ---------- Group split ----------
            if gender_dict[user] == 0:
                male_recalls.append(recall)
                male_precisions.append(precision)
                male_ndcgs.append(ndcg)
            else:
                female_recalls.append(recall)
                female_precisions.append(precision)
                female_ndcgs.append(ndcg)

            # ---------- Exposure ----------
            for item in recommended:
                categories = item_category[item]
                
                for category in categories:
                    if gender_dict[user] == 0:
                        male_exposure[category] += 1
                        male_total += 1
                    else:
                        female_exposure[category] += 1
                        female_total += 1

        # ---------- Final metrics ----------
        recall_avg = np.mean(recalls)
        precision_avg = np.mean(precisions)
        ndcg_avg = np.mean(ndcgs)

        male_recall = np.mean(male_recalls)
        female_recall = np.mean(female_recalls)

        male_precision = np.mean(male_precisions)
        female_precision = np.mean(female_precisions)

        male_ndcg = np.mean(male_ndcgs)
        female_ndcg = np.mean(female_ndcgs)

        # Gap (you can extend later for precision/ndcg too)
        gru = abs(male_recall - female_recall)

        # ---------- Exposure distribution ----------
        male_dist = {k: v / male_total for k, v in male_exposure.items()} if male_total > 0 else {k: 0 for k in all_genres}
        female_dist = {k: v / female_total for k, v in female_exposure.items()} if female_total > 0 else {k: 0 for k in all_genres}

        # Sort for visual clarity
        male_dist = dict(sorted(male_dist.items()))
        female_dist = dict(sorted(female_dist.items()))

        # ---------- PRINT ----------
        print("\n--- Overall Metrics ---")
        print(f"Recall@{k}: {recall_avg:.4f}")
        print(f"Precision@{k}: {precision_avg:.4f}")
        print(f"NDCG@{k}: {ndcg_avg:.4f}")

        print("\n--- Group Metrics ---")
        print(f"Male Recall: {male_recall:.4f}, Female Recall: {female_recall:.4f}")
        print(f"Male Precision: {male_precision:.4f}, Female Precision: {female_precision:.4f}")
        print(f"Male NDCG: {male_ndcg:.4f}, Female NDCG: {female_ndcg:.4f}")
        print(f"GRU (Recall Gap): {gru:.4f}")

        print("\n--- Exposure Distribution ---")
        print("Male:", male_dist)
        print("Female:", female_dist)

        return (
            recall_avg,
            precision_avg,
            ndcg_avg,
            male_recall,
            female_recall,
            male_precision,
            female_precision,
            male_ndcg,
            female_ndcg,
            gru
        )

def compute_exposure_from_model(
    model,
    adj,
    train_interactions,
    test_interactions,
    gender_dict,
    item_category,
    k=20
):

    model.eval()

    with torch.no_grad():

        users_emb, items_emb = model.propagate(adj)

        train_dict = {}
        for u, i in train_interactions:
            train_dict.setdefault(u, set()).add(i)

        # test users only
        test_dict = {}
        for u, i in test_interactions:
            test_dict.setdefault(u, set()).add(i)

        # get all genres
        all_genres = set()
        for genres_list in item_category.values():
            all_genres.update(genres_list)

        male_exposure = {g: 0 for g in all_genres}
        female_exposure = {g: 0 for g in all_genres}

        # CHANGE HERE (test users only)
        for user in test_dict.keys():

            user_embedding = users_emb[user]
            scores = torch.matmul(user_embedding, items_emb.T)

            train_items = train_dict.get(user, set())
            scores[list(train_items)] = -1e9

            _, top_k = torch.topk(scores, k)
            recommended = top_k.cpu().numpy()

            for item in recommended:
                categories = item_category[item]

                for category in categories:
                    if gender_dict[user] == 0:
                        male_exposure[category] += 1
                    else:
                        female_exposure[category] += 1

        return male_exposure, female_exposure