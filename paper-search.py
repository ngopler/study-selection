import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re
from io import StringIO

# Lebarkan layout halaman dan atur padding agar judul lebih ke atas
st.set_page_config(page_title="Research Paper Search System", layout="wide")

# Custom CSS untuk mengatur ulang padding
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem !important;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 95%;
    }
    h1 {
        margin-top: 0rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Function to format APA 7th citation
def format_apa_citation(row):
    authors = row['Authors']
    year = row['Year']
    title = row['Title']
    source_title = row['Source title']
    volume = row['Volume']
    issue = row['Issue']
    page_start = row['Page start']
    page_end = row['Page end']
    doi = row['DOI']

    if pd.notna(authors):
        author_list = authors.split(', ')
        if len(author_list) > 1:
            formatted_authors = ', '.join(author_list[:-1]) + ', & ' + author_list[-1]
        else:
            formatted_authors = authors
    else:
        formatted_authors = ''

    formatted_title = title + '.' if title else ''
    formatted_source = source_title if source_title else ''

    volume_issue = ''
    if pd.notna(volume):
        volume_issue += f"{volume}"
    if pd.notna(issue):
        volume_issue += f"({issue})"
    if volume_issue:
        volume_issue += ', '

    pages = ''
    if pd.notna(page_start):
        pages = f"{page_start}"
        if pd.notna(page_end):
            pages += f"-{page_end}"

    doi_str = f" https://doi.org/{doi}" if pd.notna(doi) else ''

    citation = f"{formatted_authors} ({year}). {formatted_title} {formatted_source}, {volume_issue}{pages}.{doi_str}"
    citation = re.sub(' +', ' ', citation).strip()
    return citation

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def search_with_threshold(index, query_embedding, threshold, k=1000):
    distances, indices = index.search(query_embedding, k)
    mask = distances[0] >= threshold
    filtered_indices = indices[0][mask]
    filtered_distances = distances[0][mask]
    sorted_order = np.argsort(-filtered_distances)
    return filtered_indices[sorted_order], filtered_distances[sorted_order]

# Judul utama
st.markdown("<h1>Study Selection using <em>Sentence Embeddings</em></h1>", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Pemilihan model dan pemrosesan embeddings
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state["df"] = df

    st.subheader("Pilih Model Embedding")
    model_options = {
        "all-MiniLM-L6-v2": "Cepat dan ringan (384 dimensi)",
        "all-MiniLM-L12-v2": "Lebih akurat, tetap ringan (384 dimensi)",
        "all-distilroberta-v1": "Lebih akurat (768 dimensi)",
        "all-mpnet-base-v2": "Sangat akurat, lebih lambat (768 dimensi)",
        "paraphrase-multilingual-MiniLM-L12-v2": "Mendukung multibahasa (384 dimensi)"
    }

    # Ambil model sebelumnya jika ada, default ke pertama
    default_model = st.session_state.get("selected_model", list(model_options.keys())[0])
    selected_model_name = st.selectbox("Model:", list(model_options.keys()), index=list(model_options.keys()).index(default_model))
    st.session_state["selected_model"] = selected_model_name
    st.caption(model_options[selected_model_name])

    # Tombol untuk proses embedding
    if st.button("Go Embeddings"):
        with st.spinner(f"Membuat embeddings dengan model {selected_model_name}..."):
            model = SentenceTransformer(selected_model_name)
            text_to_embed = df['Title'].fillna('') + " " + df['Abstract'].fillna('') + " " + df['Author Keywords'].fillna('')
            paper_embeddings = model.encode(text_to_embed.tolist(), show_progress_bar=True)
            index = create_faiss_index(paper_embeddings)
            st.session_state["paper_embeddings"] = paper_embeddings
            st.session_state["faiss_index"] = index
        st.success("Embeddings berhasil dibuat.")


# Ambil dari session_state
df = st.session_state.get("df", None)
index = st.session_state.get("faiss_index", None)
paper_embeddings = st.session_state.get("paper_embeddings", None)

# Pencarian
if df is not None and index is not None:
    st.header("Search Papers")
    research_question = st.text_area("Research Question", height=100)
    keywords = st.text_area("Keywords", height=50)
    threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.60, step=0.05)

    if st.button("Proses"):
        if not research_question and not keywords:
            st.warning("Please enter either a research question or keywords.")
        else:
            with st.spinner("Searching..."):
                query_text = f"{research_question} {keywords}"
                model = SentenceTransformer(st.session_state["selected_model"])
                query_embedding = model.encode([query_text])
                result_indices, result_distances = search_with_threshold(index, query_embedding, threshold)

                if len(result_indices) == 0:
                    st.warning("No papers found above the specified threshold.")
                else:
                    st.success(f"Found {len(result_indices)} papers with similarity ≥ {threshold:.2f}")
                    results = []
                    for i, (idx, score) in enumerate(zip(result_indices, result_distances)):
                        paper = df.iloc[idx]
                        results.append({
                            "No": i + 1,
                            "Paper": format_apa_citation(paper),
                            "Cited by": paper['Cited by'],
                            "Similarity": f"{score:.3f}"
                        })

                    results_df = pd.DataFrame(results)
                    st.markdown("""
                    <style>
                    table { width: 100%; table-layout: fixed; }
                    thead th:nth-child(1), tbody td:nth-child(1) {
                        width: 40px; text-align: center;
                    }
                    thead th:nth-child(3), tbody td:nth-child(3) {
                        width: 70px; text-align: center;
                    }
                    thead th:nth-child(2), tbody td:nth-child(2) {
                        width: 600px; word-wrap: break-word; text-align: left;
                    }
                    thead th:nth-child(4), tbody td:nth-child(4) {
                        width: 100px; word-wrap: break-word; text-align: center;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    st.markdown(results_df[["No", "Paper", "Cited by", "Similarity"]].to_html(index=False, escape=False), unsafe_allow_html=True)

# Sidebar petunjuk dan kredit
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload a CSV file with research papers
2. Pilih model embeddings lalu klik "Go Embeddings"
3. Masukkan pertanyaan riset dan/atau keywords
4. Atur threshold kemiripan (0.0-1.0)
5. Klik "Proses" untuk menemukan paper relevan

**Output columns:**
- No: Result number
- Paper: Formatted in APA 7th style
- Cited by: Citation count
- Similarity: Match score (higher is better)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: left; font-size: 90%;'>
© <em>tailored by</em>  
<a href='mailto:adiwjj@uima.ac.id'>adiwjj</a> 2025
</div>
""", unsafe_allow_html=True)
