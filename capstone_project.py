import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors

# ==================== Data Loading & Preprocessing ====================
data = pd.read_csv(r"D:\recomendation_project\recommendation_clean_dataset.csv")

item_counts = data['item_id'].value_counts()
data = data[data['item_id'].isin(item_counts[item_counts >= 5].index)]
data['timestamp'] = pd.to_datetime(data['stime'])
data = data.sort_values('timestamp')

split_time = data['timestamp'].quantile(0.8)
train = data[data['timestamp'] <= split_time]
test = data[data['timestamp'] > split_time]

cold_start_users = set(test['user_id']) - set(train['user_id'])
cold_start_rows = test[test['user_id'].isin(cold_start_users)]
test_df = test[~test['user_id'].isin(cold_start_users)]
train_df = pd.concat([train, cold_start_rows], ignore_index=True)

train_matrix = train_df.groupby(['user_id', 'item_id'])['event_weight'].max().unstack(fill_value=0).astype(np.float32)
test_matrix = test_df.groupby(['user_id', 'item_id'])['event_weight'].max().unstack(fill_value=0).reindex(columns=train_matrix.columns, fill_value=0).astype(np.float32)

# ==================== Collaborative Filtering ====================
train_data = train_matrix.values
user_means = np.mean(train_data, axis=1)
train_data_centered = train_data - user_means.reshape(-1, 1)
k = 100
U, sigma, Vt = svds(train_data_centered, k=k)
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_means.reshape(-1, 1)

def recommend_collaborative(user_id, n=5):
    if user_id not in train_matrix.index:
        return []
    user_idx = train_matrix.index.get_loc(user_id)
    pred_scores = predicted_ratings[user_idx]
    norm_scores = MinMaxScaler().fit_transform(pred_scores.reshape(-1, 1)).flatten()
    item_ids = train_matrix.columns
    top_n_idx = np.argsort(norm_scores)[::-1][:n]
    return list(zip(item_ids[top_n_idx], norm_scores[top_n_idx]))

# ==================== Content-Based Filtering ====================
content_columns = ['item_id', 'name', 'price', 'c0_name', 'c1_name', 'c2_name', 'brand_name', 'item_condition_name']
content_df = data[content_columns].drop_duplicates(subset=['item_id']).set_index('item_id')
content_df.fillna('Unknown', inplace=True)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(content_df['name'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=content_df.index, columns=tfidf.get_feature_names_out())

# One-hot encoding for categorical features
categorical_cols = ['c0_name', 'c1_name', 'c2_name', 'brand_name', 'item_condition_name']
content_df_encoded = pd.get_dummies(content_df[categorical_cols])
content_df['price_normalized'] = MinMaxScaler().fit_transform(content_df[['price']])

# Final content feature matrix
final_features = pd.concat([tfidf_df, content_df_encoded, content_df['price_normalized']], axis=1).fillna(0)
cosine_sim_matrix = cosine_similarity(final_features)

# ==================== Content Recommendation ====================
def build_feature_vector(product_details):
    tfidf_input = tfidf.transform([product_details.get('name', '')])
    tfidf_df_input = pd.DataFrame(tfidf_input.toarray(), columns=tfidf.get_feature_names_out())

    encoded_input = pd.DataFrame([0] * content_df_encoded.shape[1]).T
    encoded_input.columns = content_df_encoded.columns
    for col in categorical_cols:
        val = product_details.get(col)
        if val:
            col_name = f"{col}_{val}"
            if col_name in encoded_input.columns:
                encoded_input[col_name] = 1

    price_val = product_details.get('price', 0)
    try:
        price_scaled = MinMaxScaler().fit_transform([[float(price_val)]])[0][0]
    except:
        price_scaled = 0
    
    final_vector = pd.concat([tfidf_df_input, encoded_input, pd.DataFrame([price_scaled], columns=['price_normalized'])], axis=1)
    return final_vector.reindex(columns=final_features.columns, fill_value=0)

def recommend_content(product_details, top_n=5):
    input_vector = build_feature_vector(product_details)
    sim_scores = cosine_similarity(input_vector, final_features)[0]
    top_indices = np.argsort(sim_scores)[::-1][:top_n]
    top_items = final_features.index[top_indices]
    normalized_scores = MinMaxScaler().fit_transform(sim_scores[top_indices].reshape(-1, 1)).flatten()
    return list(zip(top_items, normalized_scores))

# ==================== Demographic Filtering ====================
user_features = data[['user_id', 'age', 'gender', 'location', 'income']].drop_duplicates()
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'income']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['gender', 'location'])
])
user_features_processed = preprocessor.fit_transform(user_features.drop('user_id', axis=1))
knn = NearestNeighbors(n_neighbors=3, metric='cosine').fit(user_features_processed)

def recommend_demographic(user_profile, top_n=5):
    new_user_df = pd.DataFrame([user_profile])
    new_user_processed = preprocessor.transform(new_user_df)
    distances, indices = knn.kneighbors(new_user_processed)
    similar_users = user_features.iloc[indices[0]]['user_id']
    similar_items = data[data['user_id'].isin(similar_users)]
    item_scores = similar_items.groupby('item_id')['event_weight'].mean().reset_index()
    item_scores['score'] = MinMaxScaler().fit_transform(item_scores[['event_weight']])
    return item_scores.sort_values('score', ascending=False).head(top_n)[['item_id', 'score']].values.tolist()

# ==================== Hybrid Recommendation ====================
def hybrid_recommend(user_id=None, product_details=None, new_user=None, top_n=5, weights=[0.4, 0.3, 0.3]):
    results = []
    if user_id is not None:
        results += [(iid, s, 'collab') for iid, s in recommend_collaborative(user_id, top_n)]
    if product_details:
        results += [(iid, s, 'content') for iid, s in recommend_content(product_details, top_n)]
    if new_user:
        results += [(iid, s, 'demo') for iid, s in recommend_demographic(new_user, top_n)]

    df_all = pd.DataFrame(results, columns=['item_id', 'score', 'model'])
    df_all['score'] = MinMaxScaler().fit_transform(df_all[['score']])
    model_weights = {'collab': weights[0], 'content': weights[1], 'demo': weights[2]}
    df_all['weighted'] = df_all.apply(lambda x: x['score'] * model_weights.get(x['model'], 0), axis=1)

    final = df_all.groupby('item_id')['weighted'].sum().reset_index()
    return final.sort_values('weighted', ascending=False).head(top_n)

# ==================== Input Utility ====================
def get_user_input():
    try:
        user_id = int(input("Enter user ID (or press Enter to skip): ") or -1)
        user_id = None if user_id == -1 else user_id
    except: user_id = None

    print("\n🛒 Product Details (optional):")
    product_details = {
        'name': input("Name: ").strip(),
        'price': input("Price: ").strip(),
        'c0_name': input("Category 0: ").strip(),
        'c1_name': input("Category 1: ").strip(),
        'c2_name': input("Category 2: ").strip(),
        'brand_name': input("Brand: ").strip(),
        'item_condition_name': input("Condition: ").strip()
    }
    product_details = {k: v for k, v in product_details.items() if v}
    try:
        product_details['price'] = float(product_details.get('price', 0))
    except:
        product_details.pop('price', None)

    print("\n👤 User Demographics:")
    try:
        age = int(input("Age: "))
    except:
        age = 30
    gender = input("Gender: ") or 'Unknown'
    location = input("Location: ") or 'Unknown'
    try:
        income = int(input("Income: "))
    except:
        income = 50000

    user_profile = {'age': age, 'gender': gender, 'location': location, 'income': income}
    return user_id, product_details, user_profile

# ==================== Run ====================
if __name__ == '__main__':
    user_id, product_info, demo_profile = get_user_input()
    results = hybrid_recommend(user_id=user_id, product_details=product_info, new_user=demo_profile, top_n=5)
    print("\n✅ Top Recommendations:")
    print(results)