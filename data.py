import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ratings: user-item
ratings_dict = {
    'user_id': [1, 1, 1, 2, 2, 3, 3],
    'item': ['A', 'B', 'C', 'A', 'B', 'B', 'C'],
    'rating': [5, 3, 2, 5, 1, 4, 5]
}
ratings_df = pd.DataFrame(ratings_dict)

# Item metadata
items_df = pd.DataFrame({
    'item': ['A', 'B', 'C'],
    'description': [
        'Action thriller with fast pace', 
        'Romantic drama with deep characters', 
        'Sci-fi adventure in space'
    ]
})

# Pivot user-item matrix
user_item_matrix = ratings_df.pivot(index='user_id', columns='item', values='rating').fillna(0)

# TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(items_df['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
item_indices = pd.Series(items_df.index, index=items_df['item'])
