import pandas as pd


def getDataframes():
    cluster1 = pd.read_csv(r"CSVFiles\ClusterFiles\cluster_1_customers.csv")
    cluster2 = pd.read_csv(r"CSVFiles\ClusterFiles\cluster_2_customers.csv")
    cluster3 = pd.read_csv(r"CSVFiles\ClusterFiles\cluster_21_customers.csv")
    adEvents = pd.read_csv(r"CSVFiles\Ads\ad_events.csv")
    adDetails = pd.read_csv(r"CSVFiles\Ads\ad_details.csv")

    return cluster1, cluster2, cluster3, adEvents, adDetails


def extractOfferID(adEvents):
    def get_offer_id(event_info):
        if "offer id" in event_info:
            return event_info.split("'")[3]
        elif "offer_id" in event_info:  # handle different format
            return event_info.split("'")[3]
        return None

    adEvents["extracted_offer_id"] = adEvents["event_info"].apply(get_offer_id)
    return adEvents


def mergeDataframes(cluster, adEvents, adDetails):
    adEvents = extractOfferID(adEvents)
    cluster = cluster.rename(columns={"id": "customer"})

    merged_data = pd.merge(adEvents, cluster, on="customer", how="inner")
    merged_data = pd.merge(
        merged_data,
        adDetails,
        left_on="extracted_offer_id",
        right_on="ad id",
        how="left",
    )

    # Remove rows where 'offer_type' or 'ad id' is NaN
    merged_data = merged_data.dropna(subset=["offer_type", "ad id"])

    return merged_data


def analyzeClusterResponses(merged_data):
    # Debugging: Check the count of each type of event
    print("Event Counts:")
    print(merged_data["event"].value_counts())

    # Filter to only include 'offer completed' events
    completed_offers = merged_data[merged_data["event"] == "offer completed"]

    if not completed_offers.empty:
        # Calculate the percentage of each offer type
        offer_type_counts = completed_offers["offer_type"].value_counts()
        total_completed_offers = offer_type_counts.sum()
        offer_type_percentages = (offer_type_counts / total_completed_offers) * 100

        return offer_type_percentages
    else:
        print("No completed offers found.")
        return pd.Series([], dtype=float)


def main():
    cluster1, cluster2, cluster3, adEvents, adDetails = getDataframes()

    # Debug: Check if 'offer completed' events exist in adEvents
    print("Checking 'offer completed' events in adEvents:")
    print(adEvents[adEvents["event"] == "offer completed"].head())

    # Analyze each cluster
    for i, cluster in enumerate([cluster1, cluster2, cluster3], start=1):
        print(f"\nProcessing Cluster {i}")
        merged_cluster = mergeDataframes(cluster, adEvents, adDetails)

        # Debugging: Check if 'offer completed' events are present after merging
        if "offer completed" in merged_cluster["event"].values:
            print("Offer completed events found in merged data.")
        else:
            print("No offer completed events found in merged data.")

        response_cluster = analyzeClusterResponses(merged_cluster)
        print(f"Percentage Responses for Cluster {i}:")
        print(response_cluster)


# Run the main function
main()
