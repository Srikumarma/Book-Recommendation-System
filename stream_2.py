import pickle
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,sigmoid_kernel
# Streamlit web app
books_dict = pickle.load(open("books.pkl","rb"))
data = pickle.load(open("data.pkl","rb"))
books = pd.DataFrame(books_dict)

cosine_similarities = cosine_similarity(data,data)

#data = pd.DataFrame(cosine_similarities,index=books.title)

indices = pd.Series(books.index,index = books['title']).drop_duplicates()

def recommed2(title, cos=cosine_similarities):
    idx=indices[title]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]
    title = books['title'].iloc[book_indices]
    result=pd.DataFrame({'Title':title})
    #result.drop(index=df.index[0], axis=0, inplace=True)
    result_final = result.reset_index()
    result_final=result_final.iloc[:,1]
    return result_final


st.title("Book Recommendation System")

selected_books_list = st.selectbox(
" Select the book",
books["title"].values)

if st.button("Show Recommendation"):
    result_data = recommed2(selected_books_list)
    st.text(result_data)