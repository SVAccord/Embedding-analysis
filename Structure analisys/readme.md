**Description**:
Here's the code for data preparation, training, and processing.

1. ds_partition.py - Form a set of excerpts from the training text according to Bert's settings. This facilitates the subsequent search for the necessary words for generating embeddings.

2. rnsp_bert.py - train Bert from scrtach on my data.

3. findwordforms_dict.py - Find all similar forms for the words specified in the list in the dictionary. This is necessary to search only for forms represented by a single token.

4. emball.py - Create a database (ChromaDB) of embeddings for the selected forms.

5. PCA_hdbscan.ipynb - Analyze the resulting embeddings.