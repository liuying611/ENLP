from mordenNN import *
import pandas as pd
from sklearn.model_selection import train_test_split
from EDA import *
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW

df = pd.read_csv('./data/df_10_category_balance_30k_pca.csv')
#df.drop(columns=['sa'], inplace=True)
# instantiate labelencoder object
le = LabelEncoder()
# apply le on categorical feature columns
df['category'] = le.fit_transform(df['category'])

da = DataAnlysisPCA(df)
max_len, length_list = da.get_max_word_length(plot=False)
word_count_df, df_most_common, df_pca = da.get_most_common_words(plot=False)
word_count_df = pd.Series(word_count_df['count'].values, index=word_count_df['word'].values)

train, test = train_test_split(df, test_size=0.2, random_state=42)
val, test = train_test_split(test, test_size=0.5, random_state=42)
print(f'Train set length: {len(train)} ; categories: {train["category"].nunique()}')
print(f'Val set length: {len(val)} ; categories: {val["category"].nunique()}')
print(f'Test set length: {len(test)} ; categories: {test["category"].nunique()}')

word2index, index2word = prepare_text_dict(word_count_df)

train_data = Data(train, word2index)
val_data = Data(val, word2index)
test_data = Data(test, word2index)

train_loader = DataLoader(train_data, batch_size=32, collate_fn=collate_fn_padded)
val_loader = DataLoader(val_data, batch_size=32, collate_fn=collate_fn_padded)
test_loader = DataLoader(test_data, batch_size=32, collate_fn=collate_fn_padded)

features_cols = df.columns.drop(['category', 'text', 'sa'])

label2encoding = dict(zip(le.classes_, le.transform(le.classes_)))
encoding2label = dict(zip(le.transform(le.classes_), le.classes_))

train_loader = DataLoader(train_data, batch_size=32, collate_fn=collate_fn_padded)
val_loader = DataLoader(val_data, batch_size=32, collate_fn=collate_fn_padded)
test_loader = DataLoader(test_data, batch_size=32, collate_fn=collate_fn_padded)

vector_size = 50
word2vec_embeddings = get_mapping_dict(df, word2index, vector_size= vector_size)
embedding_matrix = create_embedding_matrix( word2index, word2vec_embeddings, vector_size)
