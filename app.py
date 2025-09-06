import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from flask import Flask, render_template, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# ==================== Data Loading & Preprocessing ====================
print("Loading and preprocessing data...")
data = pd.read_csv(r"F:\Hybrid_Recommendation_System\recommendation_clean_dataset.csv")

# Basic data cleaning
item_counts = data['item_id'].value_counts()
data = data[data['item_id'].isin(item_counts[item_counts >= 5].index)]
data['timestamp'] = pd.to_datetime(data['stime'])
data = data.sort_values('timestamp')

# Train-test split
split_time = data['timestamp'].quantile(0.8)
train = data[data['timestamp'] <= split_time]
test = data[data['timestamp'] > split_time]

# Handle cold start users
cold_start_users = set(test['user_id']) - set(train['user_id'])
cold_start_rows = test[test['user_id'].isin(cold_start_users)]
test_df = test[~test['user_id'].isin(cold_start_users)]
train_df = pd.concat([train, cold_start_rows], ignore_index=True)

# Create matrices
train_matrix = train_df.groupby(['user_id', 'item_id'])['event_weight'].max().unstack(fill_value=0).astype(np.float32)
test_matrix = test_df.groupby(['user_id', 'item_id'])['event_weight'].max().unstack(fill_value=0).reindex(columns=train_matrix.columns, fill_value=0).astype(np.float32)

# ==================== Collaborative Filtering ====================
print("Training collaborative filtering model...")
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
    # Ensure scores are positive and meaningful
    pred_scores = np.clip(pred_scores, 0, None)  # Remove negative scores
    item_ids = train_matrix.columns
    top_n_idx = np.argsort(pred_scores)[::-1][:n]
    return list(zip(item_ids[top_n_idx], pred_scores[top_n_idx]))

# ==================== Content-Based Filtering ====================
print("Preparing content-based features...")
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
print("Preparing demographic features...")
user_features = data[['user_id', 'age', 'gender', 'location', 'income']].drop_duplicates()

# Handle missing values
user_features['age'].fillna(user_features['age'].median(), inplace=True)
user_features['income'].fillna(user_features['income'].median(), inplace=True)
user_features['gender'].fillna('Unknown', inplace=True)
user_features['location'].fillna('Unknown', inplace=True)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'income']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['gender', 'location'])
])

user_features_processed = preprocessor.fit_transform(user_features.drop('user_id', axis=1))
knn = NearestNeighbors(n_neighbors=3, metric='cosine').fit(user_features_processed)
def recommend_demographic(user_profile, top_n=5):
    try:
        new_user_df = pd.DataFrame([{
            'age': int(user_profile.get('age', 30)),
            'gender': user_profile.get('gender', 'Unknown'),
            'location': user_profile.get('location', 'Unknown'),
            'income': int(user_profile.get('income', 50000))
        }])
        
        new_user_processed = preprocessor.transform(new_user_df)
        distances, indices = knn.kneighbors(new_user_processed)
        similar_users = user_features.iloc[indices[0]]['user_id']
        similar_items = data[data['user_id'].isin(similar_users)]
        
        if similar_items.empty:
            return []
            
        item_scores = similar_items.groupby('item_id')['event_weight'].mean().reset_index()
        # Ensure scores are properly scaled
        item_scores['score'] = item_scores['event_weight'] / item_scores['event_weight'].max()
        return item_scores.sort_values('score', ascending=False).head(top_n)[['item_id', 'score']].values.tolist()
    except Exception as e:
        print(f"Error in demographic recommendation: {e}")
        return []

# ==================== Hybrid Recommendation ====================
def hybrid_recommend(user_id=None, product_details=None, new_user=None, top_n=5, weights=[0.4, 0.3, 0.3]):
    results = []
    
    # Collaborative filtering
    if user_id is not None and weights[0] > 0:
        try:
            collab_recs = recommend_collaborative(user_id, top_n)
            results += [(iid, s * weights[0], 'collab') for iid, s in collab_recs]
        except Exception as e:
            print(f"Collaborative filtering error: {e}")

    # Content-based filtering
    if product_details and weights[1] > 0:
        try:
            content_recs = recommend_content(product_details, top_n)
            results += [(iid, s * weights[1], 'content') for iid, s in content_recs]
        except Exception as e:
            print(f"Content-based filtering error: {e}")

    # Demographic filtering
    if new_user and weights[2] > 0:
        try:
            demo_recs = recommend_demographic(new_user, top_n)
            results += [(iid, s * weights[2], 'demo') for iid, s in demo_recs]
        except Exception as e:
            print(f"Demographic filtering error: {e}")

    if not results:
        return pd.DataFrame(columns=['item_id', 'weighted_score', 'name', 'price', 'brand_name'])

    df_all = pd.DataFrame(results, columns=['item_id', 'raw_score', 'model'])
    
    # Normalize scores to 0-1 range
    if not df_all['raw_score'].empty:
        min_score = df_all['raw_score'].min()
        max_score = df_all['raw_score'].max()
        if max_score > min_score:  # Avoid division by zero
            df_all['weighted_score'] = (df_all['raw_score'] - min_score) / (max_score - min_score)
        else:
            df_all['weighted_score'] = 0.5  # Default score if all scores are equal
    
    # Merge with product information
    final = df_all.groupby('item_id')['weighted_score'].max().reset_index()
    final = final.merge(content_df[['name', 'price', 'brand_name']], 
                      left_on='item_id', 
                      right_index=True,
                      how='left')
    
    return final.sort_values('weighted_score', ascending=False).head(top_n)

# ==================== Flask Routes ====================
@app.route('/')
def home():
    if 'user_id' in session:
        return render_template('home.html', 
                            username=f"User {session['user_id']}",
                            logged_in=True)
    return render_template('home.html', logged_in=False)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        
        try:
            user_id = int(user_id) if user_id else None
        except:
            flash("User ID must be a number", "error")
            return redirect(url_for('login'))
        
        if user_id is not None:
            user_id = int(user_id)
            # Check if user exists in the training data
            if user_id in train_matrix.index:
                session['user_id'] = user_id
                session['logged_in'] = True
                
                # Get collaborative recommendations
                recommendations = hybrid_recommend(
                    user_id=user_id,
                    weights=[1, 0, 0]  # 100% collaborative
                )
                
                return render_template('results.html',
                                    recommendations=recommendations.to_dict('records'),
                                    weights=[1, 0, 0],
                                    logged_in=True)
            else:
                flash("User ID not found. Please register or try another ID.", "error")
                return redirect(url_for('login'))
        else:
            flash("Please enter a user ID", "error")
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    global user_features
    if request.method == 'POST':
        try:
            user_id = int(request.form.get('user_id'))
            
            # Check if user ID already exists
            if user_id in user_features['user_id'].values:
                flash("User ID already exists. Please login or use a different ID.", "error")
                return redirect(url_for('register'))
            
            age = int(request.form.get('age', 30))
            gender = request.form.get('gender', 'Unknown')
            location = request.form.get('location', 'Unknown')
            income = int(request.form.get('income', 50000))
            
            # Create user profile
            new_user = {
                'age': age,
                'gender': gender,
                'location': location,
                'income': income
            }
            
            # Add new user to user_features
            
            new_user_df = pd.DataFrame([{
                'user_id': user_id,
                'age': age,
                'gender': gender,
                'location': location,
                'income': income
            }])
            user_features = pd.concat([user_features, new_user_df], ignore_index=True)
            
            # Get demographic recommendations
            recommendations = hybrid_recommend(
                new_user=new_user,
                weights=[0, 0, 1]  # 100% demographic
            )
            
            session['user_id'] = user_id
            session['logged_in'] = True
            session['demo_info'] = new_user
            
            return render_template('results.html',
                                recommendations=recommendations.to_dict('records'),
                                weights=[0, 0, 1],
                                logged_in=True)
        
        except ValueError as e:
            flash("Please enter valid numbers for ID, age and income", "error")
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        product_details = {
            'name': request.form.get('name', '').strip(),
            'price': request.form.get('price', '').strip(),
            'c0_name': request.form.get('c0_name', '').strip(),
            'c1_name': request.form.get('c1_name', '').strip(),
            'c2_name': request.form.get('c2_name', '').strip(),
            'brand_name': request.form.get('brand_name', '').strip(),
            'item_condition_name': request.form.get('item_condition_name', '').strip()
        }
        
        # Clean product details
        product_details = {k: v for k, v in product_details.items() if v}
        try:
            product_details['price'] = float(product_details.get('price', 0))
        except:
            product_details.pop('price', None)
        
        # Determine weights based on scenario
        user_exists = 'user_id' in session and session['user_id'] in train_matrix.index
        has_product_search = len(product_details) > 0
        
        if user_exists and has_product_search:
            weights = [0.3, 0.4, 0.3]  # Existing user with search
        elif user_exists:
            weights = [0.8, 0.1, 0.1]  # Existing user without search
        elif has_product_search:
            weights = [0, 0.7, 0.3]  # New user with search
        else:
            weights = [0, 0, 1]  # New user without search
        
        # Get recommendations
        recommendations = hybrid_recommend(
            user_id=session.get('user_id') if user_exists else None,
            product_details=product_details if has_product_search else None,
            new_user=session.get('demo_info') if not user_exists else None,
            weights=weights,
            top_n=5
        )
        
        return render_template('results.html', 
                             recommendations=recommendations.to_dict('records'),
                             weights=weights,
                             logged_in='user_id' in session)
    
    return render_template('search.html', logged_in='user_id' in session)

if __name__ == '__main__':
    app.run(debug=True)