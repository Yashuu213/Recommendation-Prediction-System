# import streamlit as st
# import pandas as pd
# import numpy as np

# # Load datasets
# books_df = pd.read_csv("Books.csv", low_memory=False)
# ratings_df = pd.read_csv("Ratings.csv", low_memory=False)

# # Select relevant columns (Removing 'Summary')
# books_df = books_df[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-L']]
# ratings_df = ratings_df[['ISBN', 'Book-Rating']]

# # Merge datasets
# books_with_ratings = books_df.merge(ratings_df, on="ISBN", how="left")

# # Calculate average rating and total rating count for popular books
# popular_books = books_with_ratings.groupby(["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-L"]).agg({
#     "Book-Rating": ["count", "mean"]
# }).reset_index()

# # Rename columns
# popular_books.columns = ["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-L", "Total-Ratings", "Avg-Rating"]

# # Sort by popularity
# top_5_books = popular_books.sort_values(by=["Total-Ratings", "Avg-Rating"], ascending=[False, False]).head(5)

# # Genres for filtering
# genres = ["Romance", "Mystery", "Sci-Fi", "Fictional", "Non-Fictional", "Paranormal", "Thriller"]

# # Streamlit UI
# st.title("📚 Book Recommendation System")

# # Popular Books Section
# st.subheader("🔥 Top 5 Popular Books")
# for index, row in top_5_books.iterrows():
#     st.image(row["Image-URL-L"], width=100)
#     st.write(f"**{row['Book-Title']}**")
#     st.write(f"👨‍💼 *Author:* {row['Book-Author']}")
#     st.write(f"📅 *Published:* {row['Year-Of-Publication']}")
#     st.write(f"🏢 *Publisher:* {row['Publisher']}")
#     st.write(f"⭐ *Rating:* {round(row['Avg-Rating'], 2)} ({row['Total-Ratings']} reviews)")
#     st.markdown("---")

# # Genre Selection
# st.subheader("🎭 Choose a Genre for Recommendations")
# selected_genre = st.selectbox("Select Genre", genres)

# # Dummy genre-based recommendation (replace with real logic)
# recommended_books = popular_books.sample(5)

# st.subheader(f"📖 Recommended Books in {selected_genre}")
# for index, row in recommended_books.iterrows():
#     st.image(row["Image-URL-L"], width=100)
#     st.write(f"**{row['Book-Title']}**")
#     st.write(f"👨‍💼 *Author:* {row['Book-Author']}")
#     st.write(f"📅 *Published:* {row['Year-Of-Publication']}")
#     st.write(f"🏢 *Publisher:* {row['Publisher']}")
#     st.write(f"⭐ *Rating:* {round(row['Avg-Rating'], 2)} ({row['Total-Ratings']} reviews)")
#     st.markdown("---")
# 

# import streamlit as st
# import pandas as pd
# import numpy as np

# # Load datasets
# books_df = pd.read_csv("Books.csv", low_memory=False)
# ratings_df = pd.read_csv("Ratings.csv", low_memory=False)

# # Select relevant columns
# books_df = books_df[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-L']]
# ratings_df = ratings_df[['ISBN', 'Book-Rating']]

# # Merge books and ratings
# books_with_ratings = books_df.merge(ratings_df, on="ISBN", how="left")

# # Calculate average rating and total rating count for popular books
# popular_books = books_with_ratings.groupby(["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-L"]).agg({
#     "Book-Rating": ["count", "mean"]
# }).reset_index()

# # Rename columns
# popular_books.columns = ["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-L", "Total-Ratings", "Avg-Rating"]

# # Sort by popularity
# top_5_books = popular_books.sort_values(by=["Total-Ratings", "Avg-Rating"], ascending=[False, False]).head(5)

# # Genres for filtering
# genres = ["Romance", "Mystery", "Sci-Fi", "Fictional", "Non-Fictional", "Paranormal", "Thriller"]

# # Streamlit UI
# st.title("📚 Book Recommendation System")

# # Display Popular Books
# st.subheader("🔥 Top 5 Popular Books")
# cols = st.columns(3)
# for i, row in top_5_books.iterrows():
#     with cols[i % 3]:  # Distribute books in columns correctly
#         st.image(row["Image-URL-L"], width=120)
#         st.write(f"**{row['Book-Title']}**")
#         st.write(f"👨‍💼 *Author:* {row['Book-Author']}")
#         st.write(f"📅 *Published:* {row['Year-Of-Publication']}")
#         st.write(f"🏢 *Publisher:* {row['Publisher']}")
#         st.write(f"⭐ *Rating:* {round(row['Avg-Rating'], 2)} ({row['Total-Ratings']} reviews)")
#         st.markdown("---")

# # Genre Selection
# st.subheader("🎭 Choose a Genre for Recommendations")
# selected_genre = st.selectbox("Select Genre", genres)

# if st.button("Show Genre-Based Recommendations"):
#     recommended_genre_books = popular_books.sample(5)

#     st.subheader(f"📖 Recommended Books in {selected_genre}")
#     cols = st.columns(3)
#     for i, row in recommended_genre_books.iterrows():
#         with cols[i % 3]:  # Distribute books properly
#             st.image(row["Image-URL-L"], width=120)
#             st.write(f"**{row['Book-Title']}**")
#             st.write(f"👨‍💼 *Author:* {row['Book-Author']}")
#             st.write(f"📅 *Published:* {row['Year-Of-Publication']}")
#             st.write(f"🏢 *Publisher:* {row['Publisher']}")
#             st.write(f"⭐ *Rating:* {round(row['Avg-Rating'], 2)} ({row['Total-Ratings']} reviews)")
#             st.markdown("---")

# # Book Selection for Recommendation
# st.subheader("🔍 Select a Book for Similar Recommendations")
# selected_book = st.selectbox("Select a Book", books_df["Book-Title"].unique())

# if st.button("Show Book Recommendations"):
#     recommended_books = popular_books.sample(5)

#     st.subheader(f"📌 Books Similar to {selected_book}")
#     cols = st.columns(3)
#     for i, row in recommended_books.iterrows():
#         with cols[i % 3]:  # Ensure books appear in columns
#             st.image(row["Image-URL-L"], width=120)
#             st.write(f"**{row['Book-Title']}**")
#             st.write(f"👨‍💼 *Author:* {row['Book-Author']}")
#             st.write(f"📅 *Published:* {row['Year-Of-Publication']}")
#             st.write(f"🏢 *Publisher:* {row['Publisher']}")
#             st.write(f"⭐ *Rating:* {round(row['Avg-Rating'], 2)} ({row['Total-Ratings']} reviews)")
#             st.markdown("---")



# 

# 
import streamlit as st
import pandas as pd
import numpy as np


def book_app():
    st.title("📚 Book Recommendation System")
    
# Load datasets
books_df = pd.read_csv("datasets/Books.csv", low_memory=False)
ratings_df = pd.read_csv("datasets/Ratings.csv", low_memory=False)

# Select relevant columns
books_df = books_df[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-L']]
ratings_df = ratings_df[['ISBN', 'Book-Rating']]

# Merge books and ratings
books_with_ratings = books_df.merge(ratings_df, on="ISBN", how="left")

# Calculate average rating and total rating count for popular books
popular_books = books_with_ratings.groupby(["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-L"]).agg({
    "Book-Rating": ["count", "mean"]
}).reset_index()

# Rename columns
popular_books.columns = ["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-L", "Total-Ratings", "Avg-Rating"]

# Sort by popularity
top_5_books = popular_books.sort_values(by=["Total-Ratings", "Avg-Rating"], ascending=[False, False]).head(5)

# Genres for filtering
genres = ["Romance", "Mystery", "Sci-Fi", "Fictional", "Non-Fictional", "Paranormal", "Thriller"]

# Streamlit UI
st.title("📚 Book Recommendation System")

# Display Popular Books in Row-Wise Format
st.subheader("🔥 Top 5 Popular Books")
for _, row in top_5_books.iterrows():
    col1, col2 = st.columns([1, 3])  # Image (1) | Details (3)
    with col1:
        st.image(row["Image-URL-L"], width=120)
    with col2:
        st.write(f"### {row['Book-Title']}")
        st.write(f"👨‍💼 **Author:** {row['Book-Author']}")
        st.write(f"📅 **Published:** {row['Year-Of-Publication']}")
        st.write(f"🏢 **Publisher:** {row['Publisher']}")
        st.write(f"⭐ **Rating:** {round(row['Avg-Rating'], 2)} ({row['Total-Ratings']} reviews)")
    st.markdown("---")  # Separator

# Genre Selection
st.subheader("🎭 Choose a Genre for Recommendations")
selected_genre = st.selectbox("Select Genre", genres)

if st.button("Show Genre-Based Recommendations"):
    recommended_genre_books = popular_books.sample(5)

    st.subheader(f"📖 Recommended Books in {selected_genre}")
    for _, row in recommended_genre_books.iterrows():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(row["Image-URL-L"], width=120)
        with col2:
            st.write(f"### {row['Book-Title']}")
            st.write(f"👨‍💼 **Author:** {row['Book-Author']}")
            st.write(f"📅 **Published:** {row['Year-Of-Publication']}")
            st.write(f"🏢 **Publisher:** {row['Publisher']}")
            st.write(f"⭐ **Rating:** {round(row['Avg-Rating'], 2)} ({row['Total-Ratings']} reviews)")
        st.markdown("---")

# Book Selection for Recommendation
st.subheader("🔍 Select a Book for Similar Recommendations")
selected_book = st.selectbox("Select a Book", books_df["Book-Title"].unique())

if st.button("Show Book Recommendations"):
    recommended_books = popular_books.sample(5)

    st.subheader(f"📌 Books Similar to {selected_book}")
    for _, row in recommended_books.iterrows():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(row["Image-URL-L"], width=120)
        with col2:
            st.write(f"### {row['Book-Title']}")
            st.write(f"👨‍💼 **Author:** {row['Book-Author']}")
            st.write(f"📅 **Published:** {row['Year-Of-Publication']}")
            st.write(f"🏢 **Publisher:** {row['Publisher']}")
            st.write(f"⭐ **Rating:** {round(row['Avg-Rating'], 2)} ({row['Total-Ratings']} reviews)")
        st.markdown("---")  # Separator

st.write("📌 **Select a genre & book to get recommendations!** 🎉")

def book_app():
    st.title("📚 Book Recommendation System")