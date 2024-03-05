import pandas as pd

def getDataframes():
    cluster1 = pd.read_csv(r"CSVFiles\ClusterFiles\cluster_1_customers.csv")
    cluster2 = pd.read_csv(r"CSVFiles\ClusterFiles\cluster_2_customers.csv")
    cluster3 = pd.read_csv(r"CSVFiles\ClusterFiles\cluster_6_customers.csv")
    adEvents = pd.read_csv(r"CSVFiles\Ads\ad_events.csv")
    adDetails = pd.read_csv(r"CSVFiles\Ads\ad_details.csv")

    return cluster1, cluster2, cluster3, adEvents, adDetails

def extractOfferID(adEvents):
    def get_offer_id(event_info):
        if "offer id" in event_info or "offer_id" in event_info:
            return event_info.split("'")[3]
        return None

    adEvents["extracted_offer_id"] = adEvents["event_info"].apply(get_offer_id)
    return adEvents

def mergeDataframes(cluster, adEvents, adDetails):
    adEvents = extractOfferID(adEvents)
    cluster = cluster.rename(columns={"id": "customer"})
    merged_data = pd.merge(adEvents, cluster, on="customer", how="inner")
    merged_data = pd.merge(merged_data, adDetails, left_on="extracted_offer_id", right_on="ad id", how="left")
    merged_data = merged_data.dropna(subset=["offer_type", "ad id"])
    return merged_data

def analyzeClusterResponses(merged_data):
    completed_offers = merged_data[merged_data["event"] == "offer completed"]

    if not completed_offers.empty:
        # Calculate the response for each type of offer
        offer_response = completed_offers.groupby('offer_type')['offer_type'].count()
        return offer_response
    else:
        print("No completed offers found.")
        return pd.Series([], dtype=int)

def main():
    cluster1, cluster2, cluster3, adEvents, adDetails = getDataframes()

    for i, cluster in enumerate([cluster1, cluster2, cluster3], start=1):
        print(f"\nProcessing Cluster {i}")
        merged_cluster = mergeDataframes(cluster, adEvents, adDetails)
        response_cluster = analyzeClusterResponses(merged_cluster)
        print(f"Marketing Response for Cluster {i}:")
        print(response_cluster)

if __name__ == "__main__":
    main()
