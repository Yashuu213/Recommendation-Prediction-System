import streamlit as st
import pandas as pd
import requests
import base64
from datetime import datetime

# Streamlit UI
def music_app():
    st.title("🎵 Music Recommendation")
    
# Load dataset
df = pd.read_csv("datasets/dataset.csv")

# Filter only English songs
df = df[df["track_name"].str.contains(r'^[A-Za-z0-9\s.,!?"\']+$', na=False, regex=True)]

# Spotify API Credentials
CLIENT_ID = "3c340935882e4eb186f1326de0b2fa00"
CLIENT_SECRET = "6f7d1549bd9d42298d063968ad183bf8"

# Get Spotify Access Token
def get_spotify_token():
    auth_url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    }
    data = {"grant_type": "client_credentials"}
    response = requests.post(auth_url, headers=headers, data=data)
    return response.json().get("access_token")

access_token = get_spotify_token()

# Fetch album cover from Spotify
def get_album_cover(track_name, artist):
    search_url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"q": f"track:{track_name} artist:{artist}", "type": "track", "limit": 1}

    response = requests.get(search_url, headers=headers, params=params)
    if response.status_code == 200:
        items = response.json().get("tracks", {}).get("items", [])
        if items:
            return items[0]["album"]["images"][0]["url"]
    return None



# Time-based mapping
time_mapping = {
    "Morning": ["dance", "pop", "workout"],
    "Afternoon": ["pop", "hip-hop", "indie"],
    "Evening": ["r&b", "acoustic", "chill"],
    "Night": ["jazz", "classical", "lo-fi"]
}

# Get current hour
current_hour = datetime.now().hour
if 5 <= current_hour < 11:
    time_of_day = "Morning"
elif 11 <= current_hour < 17:
    time_of_day = "Afternoon"
elif 17 <= current_hour < 21:
    time_of_day = "Evening"
else:
    time_of_day = "Night"

st.subheader(f"🎶 {time_of_day} Vibes")

# Filter dataset based on genre
selected_genres = time_mapping[time_of_day]
df_filtered = df[df["track_genre"].str.contains('|'.join(selected_genres), case=False, na=False)]
time_songs = df_filtered.sample(n=min(5, len(df_filtered)))

# Display song cards
def display_song(row):
    cover_url = get_album_cover(row['track_name'], row['artists'])
    col1, col2 = st.columns([2, 2])
    if cover_url:
        col1.image(cover_url, width=200)
    with col2:
        st.write(f"**{row['track_name']}** - {row['artists']}")
        st.write(f"⭐ Popularity: {row['popularity']}")
        st.write(f"⏱ Duration: {row['duration_ms'] // 60000}:{(row['duration_ms'] // 1000) % 60:02d}")
        st.write("---")

for _, row in time_songs.iterrows():
    display_song(row)

# # User-selected genre-based recommendation
# st.subheader("🎵 Explore by Genre")
# selected_genre = st.selectbox("Pick a Genre", df["track_genre"].unique())
# genre_songs = df[df["track_genre"] == selected_genre].sample(n=min(5, len(df[df["track_genre"] == selected_genre])))

# for _, row in genre_songs.iterrows():
#     display_song(row)

# # User-selected song-based recommendations
# st.subheader("🔍 Discover Similar Songs")
# selected_song = st.selectbox("Pick a Song", df["track_name"].unique())
# similar_songs = df[df["track_name"] != selected_song].sample(n=min(5, len(df[df["track_name"] != selected_song])))

# for _, row in similar_songs.iterrows():
#     display_song(row)

# Additional Explore Options
st.subheader("🔎 Explore More")
explore_choice = st.radio("Select an Option", ["Explore by Genre", "Discover Similar Songs"])

if explore_choice == "Explore by Genre":
    selected_genre = st.selectbox("Pick a Genre", df["track_genre"].unique(), key="explore_genre")
    explore_songs = df[df["track_genre"] == selected_genre].sample(n=min(5, len(df[df["track_genre"] == selected_genre])))
    for _, row in explore_songs.iterrows():
        display_song(row)

elif explore_choice == "Discover Similar Songs":
    selected_song = st.selectbox("Pick a Song", df["track_name"].unique(), key="explore_song")
    explore_songs = df[df["track_name"] != selected_song].sample(n=min(5, len(df[df["track_name"] != selected_song])))
    for _, row in explore_songs.iterrows():
        display_song(row)
