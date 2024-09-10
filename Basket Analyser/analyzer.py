import numpy as np
import pandas as pd 
from faker import Faker
import random 
import plotly.graph_objects as go
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.decomposition import PCA
import plotly.express as px
import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)

np.random.seed(16)
random.seed(16)

fake = Faker()

def generate_data(num_products=10, num_customers=100, num_transactions=500):
    products = [fake.word() for _ in range(num_products)]
    transcations = []

    for _ in range(num_transactions):
        customer_id = random.randint(1, num_customers)
        basket_size = random.randint(1, 5)
        basket = random.sample(products, basket_size)
        transcations.append({
            'customer_id':customer_id,
            'products': basket
        })

    df=pd.DataFrame(transcations)
    df_encoded = df.explode('products').pivot_table(
        index='customer_id',
        columns='products',
        aggfunc=lambda x: 1,
        fill_value=0
    )

    return df_encoded
        # APRIORI Algo!

def simple_apriori(df, min_support=0.1, min_confidence=0.5):
    def support(item_set):
        return (df[list(item_set)].sum(axis=1) == len(item_set)).mean()
    items = set(df.columns)
    item_sets = [frozenset(items) for item in items]
    rules =[]


    for k in range(2, len(items)+1):
        item_sets = [s for s in combinations(items, k) if support(s) >= min_support]

        for item_set in item_sets:
            item_set = frozenset(item_set)
            for i in range(1, len(item_set)):
                for antecedent in combinations(item_set, i):
                    antecedent = frozenset(antecedent)
                    consequent = item_set - antecedent
                    confidence = support(item_set) / support(consequent)
                    if confidence >= min_confidence:
                        lift = confidence / support(consequent)
                        rules.append({
                            'antecedents': ','.join(antecedent),
                            'consequents':','.join(consequent),
                            'support': support(item_set),
                            'confidence': confidence,
                            'lift': lift
                        })
        if len(rules) >= 10:
            break

    return pd.DataFrame(rules).sort_values('lift', ascending=False)
                    
# K-means Algo!

def perform_kmeans_with_progress(df, n_clusters=3, update_interval=5):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters= n_clusters, random_state=16, max_iter=100)
    
    with tqdm(total=kmeans.max_iter, desc="k_means clustering") as pbar:
        for i in range(kmeans.max_iter):
            kmeans.fit(df_scaled)
            pbar.update(1)
            if i  % update_interval == 0:
                yield kmeans.labels_
            if kmeans.n_iter_ <= i + 1:
                break
    return kmeans.labels_
        

# visualize data
        
def visualize_apriori(rules, top_n=10):
    top_rules = rules.head(top_n)

    fig = px.scatter_3d(
        top_rules,x='support', y='confidence', z='lift',
        color = 'lift', size='support',
        hover_name="antecedents", hover_data=["consequents"],
        labels={"support":"Support", "confidence":"Confidence", "lift":"Lift"},
        title=f"Top {top_n} Association Rules"

    )
    return fig

def visualize_kmeans_clusters(df, cluster_labels):

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df)

    fig = px.scatter_3d(
        x=pca_result[:, 0], y=pca_result[:, 1], z=pca_result[:, 2],
        color=cluster_labels,
        labels={"color": 'Cluster'},
        title="Customer Cluster Visualization"
    )
    
    return fig
                                    
def main():
    print("Gathering Synthetic Data...")
    df_encoded = generate_data(num_products=10, num_customers=100, num_transactions=500)
    print("Data gathering completed!")
    print(f"Dataset shape:{df_encoded.shape}")
    print("Performing Apriori Algo...")
    rules= simple_apriori(df_encoded, min_support=0.1, min_confidence=0.5)

    if not rules.empty:
        print(f"Apriori algo complete. found{len(rules)} rules.")
        viz = visualize_apriori(rules)
        viz.write_html("apriori.html")
        print("Apriori rules visuals saved as 'apriori3d.html'.")
    else:
        print("apriori algo failed!")
    
    print("Performing K-Means")
    kmeans_generater = perform_kmeans_with_progress(df_encoded, n_clusters=3, update_interval=5)
    
    for i, labels in enumerate(kmeans_generater):
        print(f"kmeans iteration {i*5}")
        viz =visualize_kmeans_clusters(df_encoded, labels)
        viz.write_html(f"customer_cluster_3d_step_{i}.html")
        print(f"Intermediate visuals saved as customer_cluster_3d_step_{i}.html")

        final_labels = labels
        print("k_means clustering complete.")

        final_viz = visualize_kmeans_clusters(df_encoded, final_labels)
        final_viz.write_html("customer_cluster3dfinal.html")
        print("final customer cluster saved.")

    print("Analysis completed")

if __name__ == "__main__":
    main()
                                    
