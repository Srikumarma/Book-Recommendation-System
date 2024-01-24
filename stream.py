import pickle
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,sigmoid_kernel
# Streamlit web app
books_dict = pickle.load(open("books.pkl","rb"))
model_knn = pickle.load(open("knn.pkl","rb"))
books = pd.DataFrame(books_dict)

tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 10), min_df = 2, stop_words='english',token_pattern=r'\w{1,}')


tfidf_matrix= tfidf.fit_transform(books_dict['description'])

cosine_similarities = cosine_similarity(tfidf_matrix,tfidf_matrix)

data = pd.DataFrame(cosine_similarities,index=books_dict.title)


def recommendation(title, data,topn=None):
    distances, indices = model_knn.kneighbors(data[data.index == title].values.reshape(1,-1),n_neighbors=6)
    result=data.index[indices.flatten()][1:6]
    return list(result)


st.title("Book Recommendation System")

selected_books_list = st.selectbox(
" Select the book",
books["title"].values)

if st.button("Show Recommendation"):
    result_data = recommendation(selected_books_list,data)
    st.text(result_data)