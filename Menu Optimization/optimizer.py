import numpy as np 
import pandas as pd
import warnings
import random
from faker import Faker
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore', category=DeprecationWarning)

np.random.seed(16)
random.seed(16)

fake = Faker()

def generate_orders_data(num_orders):
    orders = []
    for _ in range(num_orders):
        customer_id = fake.uuid4()
        dish = fake.word()
        price = round(random.uniform(5, 20), 2)
        customer_rating = random.randint(1, 5)
        orders.append([customer_id, dish, price, customer_rating])

    df = pd.DataFrame(orders, columns=['customer_id', 'dish', 'price', 'customer_rating'])
    df_encoded = df.pivot_table(
        fill_value=0,
        index='customer_id',
        columns='dish',
        aggfunc=lambda x: 1
    )

    return df_encoded

def perform_kmeans_with_progress(df, n_clusters=3, update_interval=5):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=16, max_iter=100)

    with tqdm(total=kmeans.max_iter, desc="K-Means Clustering") as pbar:
        for i in range(kmeans.max_iter):
            kmeans.fit(df_scaled)
            pbar.update(1)
            if i  % update_interval == 0:
                yield kmeans.labels_
            if kmeans.n_iter_ <= i + 1:
                break
    return kmeans.labels_

def simple_apriori_algo(df, min_support=0.01, min_confidence=0.6):
    def support(item_set):
        return (df[list(item_set)].sum(axis=1) == len(item_set)).mean()

    items = set(df.columns)
    item_sets = [frozenset([item]) for item in items]

    rules = []

    for k in range(2, len(items) + 1):
        item_sets = [s for s in combinations(items, k) if support(s) >= min_support]

        # print(f"Item sets of size {k}: {item_sets}") 
        
        for item_set in item_sets:
            item_set = frozenset(item_set)

            for i in range(1, len(item_set)):
                for antecedent in combinations(item_set, i):
                    antecedent = frozenset(antecedent)
                    consequent = item_set - antecedent
                    if support(antecedent) > 0:  # Prevent division by zero
                        confidence = support(item_set) / support(antecedent)
                        if confidence >= min_confidence:
                            if support(consequent) > 0:  # Ensure consequent is not zero
                                lift = confidence / support(consequent)
                                rules.append({
                                    'antecedents': ', '.join(map(str, antecedent)),
                                    'consequents': ','.join(map(str, consequent)),
                                    'support': support(item_set),
                                    'confidence': confidence,
                                    'lift': lift
                                })

        if len(rules) >= 10:
            break
    return pd.DataFrame(rules).sort_values('lift', ascending=False)

def visualize_apriori_rules(rules, top_n=10):
    top_rules = rules.head(top_n)

    fig = px.scatter_3d(
        top_rules, x='support', y='confidence', z='lift',
        color='lift', size='support',
        hover_name="antecedents", hover_data={"consequents": True},
        labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
        title=f"Top {top_n} Association Rules"
    )

    return fig

def visualize_kmeans_clusters(df, cluster_labels):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df)

    fig = px.scatter_3d(
        x=pca_result[:, 0], y=pca_result[:, 1], z=pca_result[:, 2],
        color = cluster_labels,
        labels={'color': 'Cluster'},
        title="3D Scatter Plot of K-Means Clustering"
    )

    return fig

def main():
    labels = None
    print("Generating Synthetic Data...")
    df_encoded = generate_orders_data(num_orders=50)
    print("Data generation completed.")
    print(f"Data Shape: {df_encoded.shape}")

    print("Performing the Apriori Algorithm...")
    rules = simple_apriori_algo(df_encoded, min_support=0.01, min_confidence=0.6)
    
    if not rules.empty:
        print(f"Apriori algorithm complete. Found {len(rules)} Rules")
        viz = visualize_apriori_rules(rules)
        viz.write_html("apriori_rules_3d.html")
        print("Apriori rules visualization saved as 'apriori_rules_3d.html'.")
    else:
        print("Apriori Algorithm failed to generate rules.")
    
    print("Performing K-Means Clustering...")
    kmeans_generator = perform_kmeans_with_progress(df_encoded, n_clusters=3, update_interval=5)
    # print("K-means generator initialized.")

    for i, labels in enumerate(kmeans_generator):
        print(f"K-means iteration {i * 5}")
        # print(f"Labels: {labels}")  
        viz = visualize_kmeans_clusters(df_encoded, labels)
        viz.write_html(f"kmeans_scatter_3d_step_{i}.html")
        print(f"Intermediate visualization saved as 'kmeans_scatter_3d_step_{i}.html'")
       
    final_labels = labels

    print("K-Means Clustering Completed.")
    final_viz = visualize_kmeans_clusters(df_encoded, final_labels)
    final_viz.write_html("kmeans_final_3d.html")
    print("Final 3D K-means cluster saved as 'kmeans_final_3d.html'.")

    print("Optimization completed.")

if __name__ == "__main__":
    main()
