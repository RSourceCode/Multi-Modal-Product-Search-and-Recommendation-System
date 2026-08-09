from datasets import load_dataset, Dataset, Features, Value, List
from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv

# Data is quite large so moved it away from the working directory
BASE_DIR =  "E:/Datasets/mm-prod-search-and-reco/" # Path.absolute(__file__).parent
PROD_DATA_DIR = f"{BASE_DIR}/data/prod"
REVIEW_DATA_DIR = f"{BASE_DIR}/data/review"

load_dotenv()

class Data():
    def __init__(self):
        os.makedirs(PROD_DATA_DIR, exist_ok=True)
        os.makedirs(REVIEW_DATA_DIR, exist_ok=True)
        self.esci_data_loader()
        self.sqid_data_loader()
        self.amazon_review_data_loader()

    def esci_data_loader(self):
        print("Loading ESIC DATA...")
        tables = ["queries", "products", "sources"]
        for table in tables:
            if not os.path.exists(f"{PROD_DATA_DIR}/{table}_train.parquet"):
                temp_table = load_dataset(path="milistu/amazon-esci-data", name=table, split=["train", "test"])
                temp_table[0].to_parquet(f"{PROD_DATA_DIR}/{table}_train.parquet")
                temp_table[1].to_parquet(f"{PROD_DATA_DIR}/{table}_test.parquet")
        print("ESIC Downloaded succesfully!")

    def sqid_data_loader(self):
        print("Loading SQID DATA...")
        tables = ["product_image_urls", "product_features", "query_features"]
        for table in tables:
            if not os.path.exists(f"{PROD_DATA_DIR}/{table}.parquet"):
                temp_table = load_dataset(path="crossingminds/shopping-queries-image-dataset", name=table, split="train")
                temp_table.to_parquet(f"{PROD_DATA_DIR}/{table}.parquet")
        print("SQID Downloaded succesfully!")

    def amazon_review_data_loader(self):
        print("Loading Amazon Review DATA...")
        tables = [
                "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing", "Automotive",
                "Baby_Products", "Beauty_and_Personal_Care", "Books",
                "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry",
                "Digital_Music", "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food",
                "Handmade_Products", "Health_and_Household", "Health_and_Personal_Care", "Home_and_Kitchen",
                "Industrial_and_Scientific", "Kindle_Store", "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments",
                "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies", "Software", "Sports_and_Outdoors", "Subscription_Boxes",
                "Tools_and_Home_Improvement", "Toys_and_Games", "Video_Games",
                ]

        meta_features = Features({
                "main_category": Value("string"),
                "title": Value("string"),
                "average_rating": Value("float64"),
                "rating_number": Value("int64"),
                "features": List(Value("string")),
                "description": List(Value("string")),
                "price": Value("string"),  # should be float64 but since some empty prices are marked with "_", initally taken as string, need to later convert price back to float64
                "images": List({
                    "360w": Value("string"),
                    "480w": Value("string"),
                    "720w": Value("string"),
                    "1080w": Value("string"),
                    "1440w": Value("string"),
                    "1920w": Value("string"),
                    "variant": Value("string"),
                    "thumb": Value("string"),
                    "large": Value("string"),
                    "hi_res": Value("string"),
                }),
                "videos": List({
                    "title": Value("string"),
                    "url": Value("string"),
                    "user_id": Value("string"),
                }),
                "store": Value("string"),
                "categories": List(Value("string")),
                "details": Value("string"),
                "parent_asin": Value("string"),
                "bought_together": Value("null"),

                # Extra columns
                "subtitle": Value("string"),
                "author": {
                    "avatar": Value("string"),
                    "name": Value("string"),
                    "about": List(Value("string")),
                },
            })
        
        for table in tables:
            if not os.path.exists(f"{REVIEW_DATA_DIR}/{table}_review.parquet"):
                # Download only the top 1000 entries from each category
                review_table = load_dataset("json", data_files=f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{table}.jsonl", split='train', streaming=True)
                sub_review_table = Dataset.from_list(list(review_table.take(1000)))
                sub_review_table.to_parquet(f"{REVIEW_DATA_DIR}/{table}_review.parquet")
            if not os.path.exists(f"{REVIEW_DATA_DIR}/{table}_meta.parquet"):
                # Download only the top 1000 entries from each category
                meta_table = load_dataset("json", data_files=f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_{table}.jsonl", split='train', streaming=True, features=meta_features)
                sub_meta_table = Dataset.from_list(list(meta_table.take(1000)))
                sub_meta_table.to_parquet(f"{REVIEW_DATA_DIR}/{table}_meta.parquet")

        print("Amazon Review Data Downloaded succesfully!")

if __name__ == "__main__":
    data = Data()