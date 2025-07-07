import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from data import ratings_df, user_item_matrix, items_df, cosine_sim, item_indices

# Collaborative Filtering
def recommend_cf(user_id, top_n=2):
    if user_id not in user_item_matrix.index:
        return []

    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
    similar_users = user_sim_df[user_id].drop(user_id).sort_values(ascending=False)

    weighted_scores = pd.Series(dtype=float)
    for other_user, sim in similar_users.items():
        other_ratings = user_item_matrix.loc[other_user]
        weighted_scores = weighted_scores.add(other_ratings * sim, fill_value=0)

    already_rated = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    weighted_scores = weighted_scores.drop(already_rated, errors='ignore')

    return list(weighted_scores.sort_values(ascending=False).head(top_n).index)

# Content-Based Filtering
def recommend_cb(item_name, top_n=2):
    if item_name not in item_indices:
        return []

    idx = item_indices[item_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    indices = [i[0] for i in sim_scores]
    return list(items_df.iloc[indices]['item'])

# Hybrid
def recommend_hybrid(user_id, content_weight=0.5, cf_weight=0.5, top_n=2):
    cf_scores = recommend_cf_scores(user_id, top_n=10)
    items_rated = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index.tolist()

    if not items_rated:
        return list(cf_scores.head(top_n).index)

    last_item = items_rated[-1]
    if last_item not in item_indices:
        return list(cf_scores.head(top_n).index)

    idx = item_indices[last_item]
    cb_scores = pd.Series(cosine_sim[idx], index=items_df['item'])
    cb_scores = cb_scores.drop(items_rated, errors='ignore')

    cf_scores = cf_scores / (cf_scores.max() or 1)
    cb_scores = cb_scores / (cb_scores.max() or 1)

    combined = (cf_scores * cf_weight).add(cb_scores * content_weight, fill_value=0)
    return list(combined.sort_values(ascending=False).head(top_n).index)

# Helper to get raw CF scores
def recommend_cf_scores(user_id, top_n=10):
    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
    similar_users = user_sim_df[user_id].drop(user_id).sort_values(ascending=False)

    weighted_scores = pd.Series(dtype=float)
    for other_user, sim in similar_users.items():
        other_ratings = user_item_matrix.loc[other_user]
        weighted_scores = weighted_scores.add(other_ratings * sim, fill_value=0)

    already_rated = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    weighted_scores = weighted_scores.drop(already_rated, errors='ignore')
    return weighted_scores.sort_values(ascending=False)
