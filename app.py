TMDB_API_KEY = "77f1ecefaed93237081eaf95292b76dc"
import streamlit as st
import pickle
import joblib
import requests
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Movie Recommendation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CSS ---------------- #

st.markdown("""
<style>

html,body,[class*="css"]{
    background-color:#0E1117;
    color:white;
}

h1{
    text-align:center;
    color:#E50914;
    font-size:55px;
}

h3{
    color:white;
}

div.stButton>button{
    width:100%;
    height:55px;
    background:#E50914;
    color:white;
    border:none;
    border-radius:12px;
    font-size:20px;
    font-weight:bold;
}

div.stButton>button:hover{
    background:#B20710;
}

.stSelectbox label{
    color:white;
    font-size:18px;
}

.movie-card{

    background:#1b1b1b;

    padding:15px;

    border-radius:15px;

    box-shadow:0px 0px 15px rgba(0,0,0,0.4);

    transition:0.3s;
}

.movie-card:hover{

    transform:scale(1.04);
}

footer{
visibility:hidden;
}

</style>

""",unsafe_allow_html=True)

# ---------------- API ---------------- #

TMDB_API_KEY = "77f1ecefaed93237081eaf95292b76dc"

IMAGE_URL="https://image.tmdb.org/t/p/w500"

# ---------------- CACHE ---------------- #

@st.cache_resource
def load_data():

    with open("model.pkl","rb") as f:

        movies=pickle.load(f)

    similarity=joblib.load("similarity_matrix.joblib")

    return movies,similarity

movies,similarity=load_data()

# ---------------- TMDB ---------------- #

@st.cache_data(show_spinner=False)
def fetch_movie_details(movie_id):

    url=f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"

    response=requests.get(url)

    if response.status_code!=200:

        return None

    return response.json()


@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):

    data=fetch_movie_details(movie_id)

    if data is None:

        return None

    if data["poster_path"]:

        return IMAGE_URL+data["poster_path"]

    return None


@st.cache_data(show_spinner=False)
def fetch_trailer(movie_id):

    url=f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"

    data=requests.get(url).json()

    if "results" not in data:

        return None

    for video in data["results"]:

        if video["site"]=="YouTube":

            return "https://www.youtube.com/watch?v="+video["key"]

    return None

# ---------------- RECOMMEND ---------------- #

def recommend(movie):

    movie=movie.lower()

    idx=movies[movies["title"]==movie].index[0]

    distances=similarity[idx]

    indices=np.argsort(distances)[::-1][1:6]

    return movies.iloc[indices]

# ---------------- HEADER ---------------- #

st.markdown("<h1>🎬 AI Movie Recommendation</h1>",unsafe_allow_html=True)

st.markdown(
"""
<h4 style='text-align:center;color:gray;'>

Discover similar movies using Machine Learning

</h4>

""",
unsafe_allow_html=True
)

st.write("")

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.image("https://cdn-icons-png.flaticon.com/512/744/744922.png",width=80)

    st.title("Movie Recommendation")

    st.write("---")

    st.write("Developed by")

    st.write("**Pallab Pratihar**")

    st.write("NIT Jamshedpur")

    st.write("---")

    st.info("Built using\n\n- Python\n- Streamlit\n- TF-IDF\n- Cosine Similarity")

movie_list=sorted(movies["title"].str.title().values)

selected_movie=st.selectbox(
    "🎥 Search Your Favourite Movie",
    movie_list
)

recommend_button=st.button("Recommend Movies")

# -------- PART 2 STARTS HERE --------
if recommend_button:

    with st.spinner("Finding Similar Movies..."):

        recommended_movies = recommend(selected_movie)

    st.markdown("## ⭐ Recommended Movies")

    cols = st.columns(5)

    for col, (_, movie) in zip(cols, recommended_movies.iterrows()):

        movie_id = movie["id"]

        details = fetch_movie_details(movie_id)

        poster = fetch_poster(movie_id)

        trailer = fetch_trailer(movie_id)

        with col:

            st.markdown("<div class='movie-card'>", unsafe_allow_html=True)

            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.image(
                    "https://via.placeholder.com/500x750?text=No+Poster",
                    use_container_width=True,
                )

            st.markdown(
                f"""
                <h4 style='text-align:center;color:white;'>
                {movie['title'].title()}
                </h4>
                """,
                unsafe_allow_html=True,
            )

            if details:

                rating = details.get("vote_average", "N/A")
                release = details.get("release_date", "N/A")
                runtime = details.get("runtime", "N/A")

                genres = ", ".join(
                    [g["name"] for g in details.get("genres", [])]
                )

                overview = details.get("overview", "")

                if len(overview) > 150:
                    overview = overview[:150] + "..."

                st.markdown(f"⭐ **Rating:** {rating}")

                st.markdown(f"📅 **Release:** {release}")

                st.markdown(f"⏱ **Runtime:** {runtime} min")

                st.markdown(f"🎭 **Genre:** {genres}")

                st.markdown("---")

                st.caption(overview)

            if trailer:

                st.link_button(
                    "▶ Watch Trailer",
                    trailer,
                    use_container_width=True
                )

            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
"""
<hr>

<center>

<h5 style="color:gray;">
🎬 AI Movie Recommendation System
</h5>

<p style="color:gray;">
Built with ❤️ using Streamlit, Scikit-Learn and TMDB API
</p>

<p style="color:gray;">
Developed by <b>Pallab Pratihar</b>
</p>

</center>
""",
unsafe_allow_html=True,
)