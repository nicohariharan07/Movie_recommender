"""
Interactive Streamlit demo for the CineMatch recommender.

Run with:  streamlit run app/streamlit_app.py
"""

import os
import sys

import streamlit as st

# allow `import src...` when run from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.recommender import Recommender, TitleNotFoundError

ARTIFACT_DIR = "models"

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="centered")
st.title("🎬 CineMatch — Content-Based Recommender")
st.caption("TF-IDF + cosine similarity over overview / genre / cast — no user ratings needed.")


@st.cache_resource
def load_recommender() -> Recommender:
    return Recommender.load(ARTIFACT_DIR)


if not os.path.exists(f"{ARTIFACT_DIR}/similarity.npy"):
    st.error(
        "No trained model found. Run this first:\n\n"
        "```\npython scripts/make_sample_dataset.py\npython main.py --config config/config.yaml\n```"
    )
    st.stop()

recommender = load_recommender()
all_titles = sorted(recommender.titles.tolist())

col1, col2 = st.columns([3, 1])
with col1:
    title = st.selectbox("Pick a title you like", options=all_titles, index=0)
with col2:
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)

if st.button("Recommend", type="primary"):
    try:
        results = recommender.recommend(title, k=int(top_k))
    except TitleNotFoundError as e:
        st.warning(str(e))
    else:
        st.subheader(f"Because you liked *{title}*:")
        for rank, (rec_title, score) in enumerate(results, start=1):
            shared = recommender.explain(title, rec_title)
            with st.container(border=True):
                left, right = st.columns([4, 1])
                left.markdown(f"**{rank}. {rec_title}**")
                if shared:
                    left.caption("Shared: " + ", ".join(shared[:6]))
                right.metric("Similarity", f"{score:.2f}")

st.divider()
with st.expander("How this works"):
    st.markdown(
        """
        1. Each title's `overview + genres + keywords + cast + director` is combined into one text profile (**tags**).
        2. `tags` is vectorized with **TF-IDF** into a sparse matrix of word-importance scores.
        3. **Cosine similarity** is computed between every pair of titles.
        4. Recommending a title just looks up its row in the similarity matrix and returns the highest-scoring neighbors.

        **Limitation:** a brand-new title with no `overview`/genre data yet can't be scored — this is the
        classic *cold-start problem*. See the README for how collaborative filtering complements this.
        """
    )
