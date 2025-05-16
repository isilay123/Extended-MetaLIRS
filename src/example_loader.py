from dataset_loader import DatasetLoader

EXAMPLE = {
    "name": "eamonn",
    "kind": "csv",
    "url": "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv"
}

if __name__ == "__main__":
    dsl = DatasetLoader()
    example_file = dsl.load_dataset(EXAMPLE)
    print(f"EXAMPLE FILE DOWNLOADED: {example_file}")
