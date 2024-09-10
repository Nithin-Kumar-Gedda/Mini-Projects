import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from faker import Faker
from sklearn.decomposition import PCA
import random
import plotly.express as px

warnings.filterwarnings('ignore', category=DeprecationWarning)

fake = Faker()

def generate_data(num_samples):
    
        soil_data = [{
            'soil_type': random.choice(['loamy', 'sandy', 'clay']),
            'ph_level': round(random.uniform(5.5, 6.5), 1),
            'nutrients': {'N': round(random.uniform(0.5, 1.5), 1),
                        'P': round(random.uniform(0.5, 1.5), 1),
                        'K': round(random.uniform(0.5, 1.5), 1)},
        }for _ in range(num_samples)]

        market_data = [{
            'crop': random.choice(['wheat', 'vegetables', 'corn', 'rice']),
            'market_price' : round(random.uniform(100, 200), 2),     
        } for _ in range(num_samples)]

        weather_report = [{
                'location': fake.city(),
                'rainfall': round(random.uniform(0, 150), 1),
                'temperature': round(random.uniform(25, 40), 1),
                'humidity': round(random.uniform(35, 85), 1)
        } for _ in range(num_samples)]


        soil_df = pd.DataFrame(soil_data)
        market_df = pd.DataFrame(market_data)
        weather_df = pd.DataFrame(weather_report)
        # merging 3 synthetic datasets to one main dataset.
        dataset = pd.concat([soil_df, weather_df, market_df], axis= 1)

        return dataset


def perform_kmeans(df, num_clusters):
        
        imp_data = df[['ph_level', 'temperature', 'rainfall', 'humidity']]

        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(imp_data)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)

        with tqdm(total=1, desc="K-means Clustering") as pbar:
            kmeans.fit(df_scaled)
            pbar.update(1)  
        return kmeans.labels_, kmeans, scaler

def visualize_kmeans_cluster(df, cluster_labels):
       pca_data = df[['ph_level', 'temperature', 'rainfall', 'humidity']]
       pca = PCA(n_components=3)
       pca_result = pca.fit_transform(pca_data)

       fig = px.scatter_3d(
            x = pca_result[:, 0], y = pca_result[:, 1], z = pca_result[:, 2],
            color = cluster_labels,
            labels= {'color': 'Cluster'},
            title = "3D PCA K-Means Clustering "
        )
       return fig


def predict_crop(in_data, df, kmeans, scaler):
       
       input_scaled = scaler.transform([in_data])

       cluster = kmeans.predict(input_scaled)[0]

       cluster_data = df[df['cluster']== cluster]

       best_crop = cluster_data.loc[cluster_data['market_price'].idxmax()]['crop']

       return best_crop


def main():
    print("Gathering Synthethic Data...")
    df_encoded = generate_data(num_samples=100)
    print("Data Gethered...")
    print(f"Data Shape: {df_encoded.shape}")
    print("Performing K-Means Clustering!")
    final_labels, kmeans, scaler = perform_kmeans(df_encoded, num_clusters=3)
    df_encoded['cluster'] = final_labels
    viz = visualize_kmeans_cluster(df_encoded, final_labels)
    viz.write_html("final_3d.html")
    print("K-Means 3D Visualization saved as'final_3d.html'.")

    print("Please enter the following information:")
    ph_level = float(input("Soil pH level (e.g., 6.7): "))
    temperature = float(input("Temperature (°C, e.g., 22.0): "))
    rainfall = float(input("Rainfall (mm, e.g., 100.0): "))
    humidity = float(input("Humidity (% e.g., 60.0): "))

    # Prepare input data
    input_data = [ph_level, temperature, rainfall, humidity]

    # Predict the best crop
    best_crop = predict_crop(input_data, df_encoded, kmeans, scaler)

    print(f"The best crop to plant is: {best_crop}")

if __name__ == "__main__":
    main()
        