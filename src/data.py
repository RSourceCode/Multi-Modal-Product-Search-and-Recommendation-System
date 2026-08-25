import os
import re
import string
import pandas as pd
from typing import Literal
from dotenv import load_dotenv
from datasets import load_dataset, Dataset, Features, Value, List

# Data is quite large so moved it away from the working directory
BASE_DIR =  "../.." # Path.absolute(__file__).parent
PROD_DATA_DIR = f"{BASE_DIR}/data/prod"
REVIEW_DATA_DIR = f"{BASE_DIR}/data/review"

load_dotenv()

class Data():
    def __init__(self, train_size = 0.7, val_size = 0.1, test_size = 0.2):
        os.makedirs(PROD_DATA_DIR, exist_ok=True)
        os.makedirs(REVIEW_DATA_DIR, exist_ok=True)
        self.__esci_data_loader()
        self.__sqid_data_loader()
        self.__amazon_review_data_loader()
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def __esci_data_loader(self):
        print("Loading ESIC DATA...")
        tables = ["queries", "products", "sources"]
        for table in tables:
            if not os.path.exists(f"{PROD_DATA_DIR}/{table}_train.parquet"):
                temp_table = load_dataset(path="milistu/amazon-esci-data", name=table, split=["train", "test"])
                temp_table[0].to_parquet(f"{PROD_DATA_DIR}/{table}_train.parquet")
                temp_table[1].to_parquet(f"{PROD_DATA_DIR}/{table}_test.parquet")
        print("ESIC Downloaded succesfully!")

    def __sqid_data_loader(self):
        print("Loading SQID DATA...")
        tables = ["product_image_urls", "product_features", "query_features"]
        for table in tables:
            if not os.path.exists(f"{PROD_DATA_DIR}/{table}.parquet"):
                temp_table = load_dataset(path="crossingminds/shopping-queries-image-dataset", name=table, split="train")
                temp_table.to_parquet(f"{PROD_DATA_DIR}/{table}.parquet")
        print("SQID Downloaded succesfully!")

    def __amazon_review_data_loader(self, num_rows = 1000):
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
                # Download only the top num_rows entries from each category
                review_table = load_dataset("json", data_files=f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{table}.jsonl", split='train', streaming=True)
                sub_review_table = Dataset.from_list(list(review_table.take(num_rows)))
                sub_review_table.to_parquet(f"{REVIEW_DATA_DIR}/{table}_review.parquet")
            if not os.path.exists(f"{REVIEW_DATA_DIR}/{table}_meta.parquet"):
                # Download only the top num_rows entries from each category
                meta_table = load_dataset("json", data_files=f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_{table}.jsonl", split='train', streaming=True, features=meta_features)
                sub_meta_table = Dataset.from_list(list(meta_table.take(num_rows)))
                sub_meta_table.to_parquet(f"{REVIEW_DATA_DIR}/{table}_meta.parquet")

        print("Amazon Review Data Downloaded succesfully!")

    def get_complete_data(self, choice1 : Literal['train', 'validation', 'test'] = 'train'):
        product_image_urls_table = pd.read_parquet(f'{PROD_DATA_DIR}/product_image_urls.parquet')
        products_test_table = pd.read_parquet(f'{PROD_DATA_DIR}/products_test.parquet')
        products_train_table = pd.read_parquet(f'{PROD_DATA_DIR}/products_train.parquet')
        queries_table = pd.read_parquet(f'{PROD_DATA_DIR}/queries_train.parquet')

        product_table = pd.concat([products_train_table, products_test_table], ignore_index=True)
        comb_table =  queries_table.merge(product_table, on='product_id', how='left').merge(product_image_urls_table, on='product_id', how='left')
        eng_table =  comb_table[comb_table['product_locale_x'] == 'us'].dropna(how='any').reset_index(drop=True)

        if(choice1 == 'train'):
            return eng_table[ : int(len(eng_table) * self.train_size)]

        if(choice1 == 'validation'):
            return eng_table[ int(len(eng_table) * self.train_size) : int(len(eng_table) * (self.val_size + self.train_size))]

        if(choice1 == 'test'):
            return eng_table[int(len(eng_table) * (self.val_size + self.train_size)) : ]        

    def get_prod_data(self):
        product_image_urls_table = pd.read_parquet(f'{PROD_DATA_DIR}/product_image_urls.parquet')
        product_train_table = pd.read_parquet(f'{PROD_DATA_DIR}/products_train.parquet')

        prod_table = product_image_urls_table.merge(product_train_table, how='inner', on='product_id')

        # Clean data
        cols = [
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
            "product_color"
        ]

        d_clean_prod = prod_table

        for col in cols:
            # Remove URLs, Remove HTML tags, lowercase and remove punctuation
            d_clean_prod[col]  = d_clean_prod[col].str.lower().str.replace(r"http\S+|www\S+|https\S+", "", regex=True).str.replace(r"<.*?>", "", regex=True).str.translate(str.maketrans("", "", string.punctuation))

        return d_clean_prod[d_clean_prod['product_locale'] == 'us'].dropna().reset_index(drop=True)

    def get_data(self):
        product_features_table = pd.read_parquet(f'{PROD_DATA_DIR}/product_features.parquet')
        product_image_urls_table = pd.read_parquet(f'{PROD_DATA_DIR}/product_image_urls.parquet')
        products_test_table = pd.read_parquet(f'{PROD_DATA_DIR}/products_test.parquet')
        product_train_table = pd.read_parquet(f'{PROD_DATA_DIR}/products_train.parquet')
        queries_train_table = pd.read_parquet(f'{PROD_DATA_DIR}/queries_train.parquet')
        query_features_table = pd.read_parquet(f'{PROD_DATA_DIR}/query_features.parquet')

        return product_features_table, product_image_urls_table, products_test_table, product_train_table, queries_train_table, query_features_table

    def get_review_data(self):
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
        review_tables = {}
        for table in tables:
            review = pd.read_parquet(f"{REVIEW_DATA_DIR}/{table}_review.parquet")
            meta = pd.read_parquet(f"{REVIEW_DATA_DIR}/{table}_meta.parquet")
            review_tables[table] = meta.merge(review, how='inner', on='parent_asin')

        return review_tables

    def clean_data(self, data):
        # product_title , product_description , product_bullet_point , product_brand , product_color <- clean these columns
        pass


if __name__ == "__main__":
    data = Data()