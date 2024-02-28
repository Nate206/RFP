import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

# Loading the data
userData = pd.read_csv(r"CSVFiles\customer_profiles.csv")
ad_events = pd.read_csv(r"CSVFiles\ad_events.csv")
ad_details = pd.read_csv(r"CSVFiles\ad_details.csv")

# Merge ad_events with ad_details
ad_events = ad_events.merge(
    ad_details, left_on="event_info", right_on="ad id", how="left"
)

# Preprocessing event_info to extract the offer id
ad_events["offer_id"] = ad_events["event_info"].apply(lambda x: eval(x)["offer id"])

# Creating a binary feature for each event type
event_type_dummies = pd.get_dummies(ad_events["event"], prefix="event")

# Creating features for counts of each event type
events_count = ad_events.groupby("customer").agg({"event": "count"})

# Merge with userData
userData = userData.merge(events_count, left_on="id", right_on="customer", how="left")

# Feature engineering for ad response
ad_response_features = userData[["id", "event_count"]]  # Add other features you created

# Preprocess the data: scale numerical data and encode categorical data
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            ["age", "income"],
        ),  # Add more numerical features if available
        (
            "cat",
            OneHotEncoder(),
            ["gender", "ever_married"],
        ),  # Add more categorical features if available
    ]
)

# Define the clustering model
clustering_model = AgglomerativeClustering(
    n_clusters=None, distance_threshold=0, affinity="euclidean", linkage="ward"
)

# Create a pipeline that preprocesses the data and then applies clustering
pipeline = make_pipeline(preprocessor, clustering_model)

# Fit the pipeline to the user data with ad responses
pipeline.fit(userData)

# The labels_ attribute will give you the cluster labels for each user
clusters = pipeline.named_steps["agglomerativeclustering"].labels_

# Add the cluster labels to the user data
userData["cluster"] = clusters

# Calculate the silhouette score
silhouette_avg = silhouette_score(userData, clusters)
print(f"Silhouette Score: {silhouette_avg}")

# Now you can analyze the clusters
print(userData.groupby("cluster").mean())
