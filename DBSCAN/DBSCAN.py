import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD


def getCsvFiles():
    customer_profiles = pd.read_csv(r"CSVFiles\Ads\customer_profiles.csv")
    ad_events = pd.read_csv(r"CSVFiles\Ads\ad_events.csv")
    ad_details = pd.read_csv(r"CSVFiles\Ads\ad_details.csv")
    return customer_profiles, ad_events, ad_details


def encodeAndScaleData(df, numeric_columns, categorical_columns):
    # Combine OneHotEncoder for categorical columns and MinMaxScaler for numeric columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numeric_columns),
            ("cat", OneHotEncoder(), categorical_columns),
        ]
    )

    # Apply transformations
    processed_data = preprocessor.fit_transform(df)
    return processed_data


def dbScan(processed_data, eps=0.5, min_samples=5):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(processed_data)
    return clustering.labels_


def find_optimal_eps(data, min_samples, eps_values):
    best_eps = None
    best_score = -1

    for eps in eps_values:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        labels = clustering.labels_

        if len(set(labels)) > 1 and np.count_nonzero(labels != -1) > 1:
            score = silhouette_score(data, labels)
            print(f"Eps: {eps} - Silhouette Score: {score}")

            if score > best_score:
                best_score = score
                best_eps = eps

    return best_eps, best_score


def find_optimal_min_samples(data, best_eps, min_samples_values):
    best_min_samples = None
    best_score = -1

    for min_samples in min_samples_values:
        clustering = DBSCAN(eps=best_eps, min_samples=min_samples).fit(data)
        labels = clustering.labels_

        # Calculate Silhouette Score only if more than one cluster is found
        if len(set(labels)) > 1 and np.count_nonzero(labels != -1) > 1:
            score = silhouette_score(data, labels)
            print(f"Min_samples: {min_samples} - Silhouette Score: {score}")

            if score > best_score:
                best_score = score
                best_min_samples = min_samples

    return best_min_samples, best_score


def plot_clusters(reduced_data, labels, clusters_to_plot=None):
    plt.figure(figsize=(10, 6))
    unique_labels = set(labels)

    # If a subset of clusters is specified, filter the unique labels
    if clusters_to_plot is not None:
        unique_labels = [x for x in unique_labels if x in clusters_to_plot]

    for label in unique_labels:
        if label == -1 and clusters_to_plot is not None:
            continue  # skip noise if we are plotting specific clusters
        # Assign black to noise if not skipping
        color = (
            "black" if label == -1 else plt.cm.jet(float(label) / max(unique_labels))
        )
        plt.scatter(
            reduced_data[labels == label, 0],
            reduced_data[labels == label, 1],
            c=[color],
            label=f"Cluster {label}",
        )

    plt.title("DBSCAN Clustering")
    plt.xlabel("PCA Feature 1")
    plt.ylabel("PCA Feature 2")
    plt.legend()
    plt.show()


def analyze_clusters(df, labels, categorical_columns, numeric_columns):
    df["Cluster"] = labels

    # Filter out noise if present
    df = df[df["Cluster"] != -1]

    # Initialize a dictionary to hold cluster profiles
    cluster_profiles = {}

    # Loop through each cluster and calculate statistics
    for cluster in sorted(df["Cluster"].unique()):
        cluster_data = df[df["Cluster"] == cluster]
        profile = {
            "size": len(cluster_data),
            "numeric_stats": cluster_data[numeric_columns].mean().to_dict(),
            "categorical_stats": {},
        }
        for col in categorical_columns:
            profile["categorical_stats"][col] = cluster_data[col].mode()[0]
        cluster_profiles[cluster] = profile

    return cluster_profiles


def reduce_dimensions(data, n_components=2):
    svd = TruncatedSVD(n_components=n_components)
    reduced_data = svd.fit_transform(data)
    return reduced_data


def main():
    customer_profiles, ad_events, ad_details = getCsvFiles()

    # Adjusting column configuration as per your data
    numeric_columns = ["age", "income"]  # Assuming 'age' and 'income' are numeric
    categorical_columns = [
        "gender",
        "ever_married",
        "home_state",  # Assuming these are categorical
        # Add other categorical columns if applicable
    ]

    # Process the data
    processed_customer_profiles = encodeAndScaleData(
        customer_profiles, numeric_columns, categorical_columns
    )

    # DBSCAN parameters
    min_samples_values = range(2, 10)
    eps_values = np.linspace(0.1, 2.0, 20)

    # Find the best epsilon value
    best_eps, eps_score = find_optimal_eps(processed_customer_profiles, 5, eps_values)
    print(f"Best Eps: {best_eps} with Silhouette Score: {eps_score}")

    # Test various min_samples values with the best eps
    if best_eps is not None:
        best_min_samples, min_samples_score = find_optimal_min_samples(
            processed_customer_profiles, best_eps, min_samples_values
        )
        print(
            f"Best Min_samples: {best_min_samples} with Silhouette Score: {min_samples_score}"
        )

        if best_min_samples is not None:
            cluster_labels = dbScan(
                processed_customer_profiles, eps=best_eps, min_samples=best_min_samples
            )

            if len(cluster_labels) == len(customer_profiles):
                customer_profiles["Cluster"] = cluster_labels

                # Count the number of clusters
                unique_clusters = np.unique(cluster_labels)
                # Exclude noise if present
                num_clusters = len(unique_clusters) - (
                    1 if -1 in unique_clusters else 0
                )
                print(f"Number of clusters: {num_clusters}")

                # Identify the three largest clusters
                cluster_sizes = customer_profiles["Cluster"].value_counts()
                # Exclude noise (cluster label -1) if present
                cluster_sizes = cluster_sizes[cluster_sizes.index != -1]
                largest_clusters = cluster_sizes.nlargest(3).index

                # Reduce dimensions for visualization
                reduced_data = reduce_dimensions(processed_customer_profiles)

                # Plot a subset of clusters for analysis (e.g., largest clusters)
                plot_clusters(
                    reduced_data, cluster_labels, clusters_to_plot=largest_clusters
                )

                # Filter and export data for each of the three largest clusters
                for cluster in largest_clusters:
                    cluster_data = customer_profiles[
                        customer_profiles["Cluster"] == cluster
                    ]
                    filename = f"cluster_{cluster}_customers.csv"
                    cluster_data.to_csv(filename, index=False)
                    print(f"Data for Cluster {cluster} exported to {filename}")
            else:
                print(
                    "Error: Mismatch in the number of cluster labels and the number of rows in the DataFrame."
                )
    else:
        print("No suitable eps value found.")


# Run the main function
main()
