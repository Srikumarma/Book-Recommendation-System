import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,sigmoid_kernel

df=pd.read_json('goodreads_book_series.json',lines=True)

df=df.dropna(how='all')

df=df[df['description'] !='']

df.drop(df.tail(140000).index,
        inplace = True)
df.reset_index(drop = True, inplace = True)

df=df.drop_duplicates(subset='title', keep="last")

df.reset_index(drop = True, inplace = True)

df = df.drop(df.columns[[0,1]], axis=1)

tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df = 5, stop_words='english',token_pattern=r'\w{1,}')

tfidf_matrix= tfidf.fit_transform(df['description'])

tfidf_matrix.shape

tfidf_matrix.toarray()

cosine_similarities = cosine_similarity(tfidf_matrix,tfidf_matrix)

data = pd.DataFrame(cosine_similarities,index=df.title)

#cosine_similarities[724]

cosine_similarities

#sig=sigmoid_kernel(tfidf_matrix,tfidf_matrix)

indices = pd.Series(df.index,index = df['title']).drop_duplicates()

from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(algorithm='brute')

model_knn=model_knn.fit(cosine_similarities)

def recommendation(title, data,topn=None):
    distances, indices = model_knn.kneighbors(data[data.index == title].values.reshape(1,-1),n_neighbors=6)
    result=data.index[indices.flatten()][1:6]
    return list(result)

recommendation('''The Norton History of Modern Europe''',data)

import pickle
pickle.dump(df,open("books.pkl","wb"))
pickle.dump(model_knn,open("knn.pkl","wb"))