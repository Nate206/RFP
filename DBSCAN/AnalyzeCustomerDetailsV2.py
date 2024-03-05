import pandas as pd
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def analyze_cluster(cluster_data):
    # Calculating basic statistics
    average_age = cluster_data["age"].mean()
    average_income = cluster_data["income"].mean()
    most_common_gender = cluster_data["gender"].mode()[0]
    average_kids = cluster_data["kids"].mean()
    most_common_state = cluster_data["home_state"].mode()[0]  # Get the most common home state

    return {
        "Average Age": average_age,
        "Average Income": average_income,
        "Most Common Gender": most_common_gender,
        "Average Number of Kids": average_kids,
        "Most Common State": most_common_state  # Adding the most common state to the analysis
    }

def main():
    directory = r"CSVFiles\ClusterFiles"  # Specify your directory here
    customer_profiles_path = r"C:\Users\tyler\OneDrive\Desktop\GitHub Repositories\RFP\CSVFiles\Ads\customer_profiles.csv"  # Adjust the path as necessary
    total_profiles = len(load_data(customer_profiles_path))

    # List all cluster CSV files in the directory
    cluster_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    for file_name in cluster_files:
        file_path = os.path.join(directory, file_name)
        cluster_data = load_data(file_path)
        analysis = analyze_cluster(cluster_data)

        cluster_size = len(cluster_data)
        cluster_percentage = (cluster_size / total_profiles) * 100

        print(f"\nAnalysis for {file_name}:")
        for key, value in analysis.items():
            print(f"{key}: {value}")
        print(f"Percentage of Total Data: {cluster_percentage:.2f}%")
        #print out the most_common_state for each cluster
        print(f"Most Common State: {analysis['Most Common State']}")

if __name__ == "__main__":
    main()
