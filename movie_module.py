# Import necessary libraries
import streamlit as st
import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the new dataset
file_path = 'imdb_top_1000.csv'
movies = pd.read_csv(file_path)

# Prepare movie tags for similarity calculation
movies['tags'] = movies['Genre'] + ' ' + movies['Overview'] + ' ' + movies['Director'] + ' ' + movies['Star1'] + ' ' + movies['Star2'] + ' ' + movies['Star3'] + ' ' + movies['Star4']

cv = CountVectorizer(stop_words='english')
tag_matrix = cv.fit_transform(movies['tags'])
similarity = cosine_similarity(tag_matrix)

# Movie Recommendation Function
def recommend(movie):
    movie = movie.lower()
    index = movies[movies['Series_Title'].str.lower() == movie].index

    if index.empty:
        return []

    index = index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    recommended_movies = []
    for i in distances[1:11]:  # Top 10 recommendations
        row = movies.iloc[i[0]]
        cast = f"{row['Star1']}, {row['Star2']}, {row['Star3']}, {row['Star4']}"
        recommended_movies.append((row['Series_Title'], row['IMDB_Rating'], row['Poster_Link'].replace("_SX300", "_SX700"), row['Overview'], row['Gross'], row['Runtime'], cast))

    return recommended_movies

# Genre-based Recommendation Function
def recommend_by_genre(genre):
    genre = genre.lower()
    genre_matches = movies[movies['Genre'].str.contains(genre, case=False, na=False)]

    recommended_movies = []
    for _, row in genre_matches.head(10).iterrows():
        cast = f"{row['Star1']}, {row['Star2']}, {row['Star3']}, {row['Star4']}"
        recommended_movies.append((row['Series_Title'], row['IMDB_Rating'], row['Poster_Link'].replace("_SX300", "_SX700"), row['Overview'], row['Gross'], row['Runtime'], cast))

    return recommended_movies

# Get Top 5 Most Viewed Movies
def get_top_5_movies():
    top_movies = movies.sort_values(by='No_of_Votes', ascending=False).head(5)

    top_5_movies = []
    for _, row in top_movies.iterrows():
        cast = f"{row['Star1']}, {row['Star2']}, {row['Star3']}, {row['Star4']}"
        top_5_movies.append((row['Series_Title'], row['IMDB_Rating'], row['Poster_Link'].replace("_SX300", "_SX700"), row['Overview'], row['Gross'], row['Runtime'], cast))

    return top_5_movies

# Get Random Movies
def get_random_movies():
    random_movies = movies.sample(5)

    random_movie_list = []
    for _, row in random_movies.iterrows():
        cast = f"{row['Star1']}, {row['Star2']}, {row['Star3']}, {row['Star4']}"
        random_movie_list.append((row['Series_Title'], row['IMDB_Rating'], row['Poster_Link'].replace("_SX300", "_SX700"), row['Overview'], row['Gross'], row['Runtime'], cast))

    return random_movie_list

# Streamlit App Header
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🎬 Movie Recommender System 🍿</h1>", unsafe_allow_html=True)

# Display Top 5 Most Viewed Movies
top_5_movies = get_top_5_movies()
if top_5_movies:
    st.markdown("<h2 style='color: #FFD700;'>🔥 Top 5 Most Viewed Movies:</h2>", unsafe_allow_html=True)

    cols = st.columns(5)

    for idx, (movie, rating, poster, overview, gross, runtime, cast) in enumerate(top_5_movies):
        with cols[idx]:
            st.image(poster, use_column_width=True)
            st.markdown(f"**{movie}**")
            st.markdown(f"⭐ **Rating:** {rating}")

# Display Random Movies
random_movies = get_random_movies()
if random_movies:
    st.markdown("<h2 style='color: #00FFFF;'>🎲 Popular Movies:</h2>", unsafe_allow_html=True)

    cols = st.columns(5)

    for idx, (movie, rating, poster, overview, gross, runtime, cast) in enumerate(random_movies):
        with cols[idx]:
            st.image(poster, use_column_width=True)
            st.markdown(f"**{movie}**")

# Choose recommendation method
option = st.selectbox("🔍 How would you like to find movies?", ("By Movie Name", "By Genre"))

if option == "By Movie Name":
    title_list = movies['Series_Title'].values
    selected_movie = st.selectbox("🎥 Type or select a movie from the dropdown:", title_list)

    if st.button('🚀 Show Recommendations'):
        recommended_movies = recommend(selected_movie)

        st.markdown("<h2 style='color: #4CAF50;'>✨ Recommended Movies:</h2>", unsafe_allow_html=True)

        for movie, rating, poster, overview, gross, runtime, cast in recommended_movies:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(poster, use_column_width=True)

            with col2:
                st.markdown(f"**{movie}**")
                st.markdown(f"⭐ **Rating:** {rating}")
                st.markdown(f"_Summary:_ {overview}")
                st.markdown(f"💰 **Revenue:** ${gross}")
                st.markdown(f"⏳ **Duration:** {runtime}")
                st.markdown(f"🎭 **Cast:** {cast}")

            st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

elif option == "By Genre":
    genre_input = st.text_input("🎭 Enter a genre (e.g., Action, Comedy, Drama):")

    if genre_input:
        recommended_movies = recommend_by_genre(genre_input)

        st.markdown("<h2 style='color: #2196F3;'>🔎 Movies in Genre:</h2>", unsafe_allow_html=True)

        for movie, rating, poster, overview, gross, runtime, cast in recommended_movies:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(poster, use_column_width=True)

            with col2:
                st.markdown(f"**{movie}**")
                st.markdown(f"⭐ **Rating:** {rating}")
                st.markdown(f"_Summary:_ {overview}")
                st.markdown(f"💰 **Revenue:** ${gross}")
                st.markdown(f"⏳ **Duration:** {runtime}")
                st.markdown(f"🎭 **Cast:** {cast}")

            st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)
