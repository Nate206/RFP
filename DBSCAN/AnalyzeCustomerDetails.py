import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)


def analyze_cluster(cluster_data):
    # Calculating basic statistics
    average_age = cluster_data["age"].mean()
    average_income = cluster_data["income"].mean()
    most_common_gender = cluster_data["gender"].mode()[
        0
    ]  # Assuming 'gender' is a categorical column
    average_kids = cluster_data["kids"].mean()

    return {
        "Average Age": average_age,
        "Average Income": average_income,
        "Most Common Gender": most_common_gender,
        "Average Number of Kids": average_kids,
    }


def main():
    cluster_files = [
        r"CSVFiles\ClusterFiles\cluster_1_customers.csv",
        r"CSVFiles\ClusterFiles\cluster_2_customers.csv",
        r"CSVFiles\ClusterFiles\cluster_21_customers.csv",
    ]

    for i, file_path in enumerate(cluster_files, start=1):
        cluster_data = load_data(file_path)
        analysis = analyze_cluster(cluster_data)

        print(f"\nCluster {i} Analysis:")
        for key, value in analysis.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
