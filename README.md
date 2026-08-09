# Multi-Modal-Product-Search-and-Recommendation-System

## Data
### For Product Search:
Looked into various existing datasets:
1.  Amazon ESIC dataset ```https://github.com/amazon-science/esci-data/blob/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet``` : This was a really good dataset having text queries and a good list of products which are relevant for those queries. Problem was that it did not have support image queries nor did the products have their images.
2. Shopping Queries Image Dataset (SQID) ```https://github.com/Crossing-Minds/shopping-queries-image-dataset``` : This was a augmentation to the ESIC dataset, which provided product images for most of the available products. Though the problem of image queries could still not be solved.
3. M-BEIR ``` https://huggingface.co/datasets/TIGER-Lab/M-BEIR ``` : This is a really good dataset with both image and text queries along with a list of positive and negative candidates for each query. The problem with this dataset was that it's scope was not exactly related to products, it was more focused on content retrieval from say newspapers or wikipedia article which did not fit with the other half of the project corresponding to recommendation. 
4. M5Product  ``` https://arxiv.org/pdf/2109.04275v1``` : This had even more modalities such as audio and tables, but as with M-BEIR, it was also quite misaligned with our goals
5. MMShopBench ``` https://arxiv.org/html/2607.29002v1 ``` : One of the very recent datasets in this context which felt to be relevant as well, but was not properly uploaded on huggingface at he date we very looking for the data.

There were many other datasets we looked into such as FashionIQ, H&M, etc. but found them to be too narrowly scoped for our purpose and were hence rejected.

Finally, we decided to use 1 + 2 with an innovative idea to get image queries. The idea is pretty simple but effective. For a given text query, we would have a list of exact, similar, complementary and irrelevant products. Now the products which are the exact match of the query, will in some sense share a similar relation as the text query with this list of products. Hence, what we decided is to use the image query as the image of the exact match and the same set of products and tags as we had for that text query. This completed the only missing piece we had in this dataset.

### For Recommendation:
We decided to use Amazon Review 23 ``` https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 ``` : This contains a large list of user reviews for various products. Being 5-core, each user has atleast 5 reviews and each product in this data also has atleast 5 reviews.