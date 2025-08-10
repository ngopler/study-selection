import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re
from io import StringIO
import traceback

hide_github_icon = """
    <style>
        [data-testid="stToolbar"] {visibility: hidden;}
    </style>
"""
st.markdown(hide_github_icon, unsafe_allow_html=True)

st.set_page_config(page_title="Research Paper Search System", layout="wide")
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

def format_apa_citation(row):
    try:
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
    except Exception as e:
        st.error(f"Error formatting citation: {str(e)}")
        return "Citation formatting error"

def create_faiss_index(embeddings):
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return index
    except Exception as e:
        st.error(f"Error creating FAISS index: {str(e)}")
        return None

def search_with_threshold(index, query_embedding, threshold, k=1000):
    try:
        distances, indices = index.search(query_embedding, k)
        mask = distances[0] >= threshold
        filtered_indices = indices[0][mask]
        filtered_distances = distances[0][mask]
        sorted_order = np.argsort(-filtered_distances)
        return filtered_indices[sorted_order], filtered_distances[sorted_order]
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return [], []

st.markdown("""
<h1>Study Selection using <em>Sentence Embeddings</em></h1>
""", unsafe_allow_html=True)

# CSS untuk membatasi lebar selectbox
st.markdown("""
    <style>
    div[data-baseweb="select"] {
        max-width: 350px !important;  /* Ubah sesuai kebutuhan */
    }
    </style>
    """, unsafe_allow_html=True)

colA, colB = st.columns([0.5, 5])  # kolom 1 untuk label, kolom 2 untuk selectbox
with colA:
    st.markdown("**Data Source:**")  # label manual
with colB:
    data_source = st.selectbox(
        label="",  # tidak digunakan karena disembunyikan
        options=["Scopus", "Lens.org"],
        index=0,  # default ke "Scopus"
        key="data_source_select",
        label_visibility="collapsed"
    )

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.fillna('', inplace=True)
        
        # Rename kolom agar sesuai dengan skema internal
        if data_source == "Lens.org":
            df = df.rename(columns={
                'Author/s': 'Authors',
                'Publication Year': 'Year',
                'Source Title': 'Source title',
                'Issue Number': 'Issue',
                'Start Page': 'Page start',
                'End Page': 'Page end',
                'Keywords': 'Author Keywords',
                'Citing Works Count': 'Cited by'  # disamakan agar tidak perlu ubah di banyak tempat
            })
        
        st.session_state["df"] = df
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        st.error("Detail error:")
        st.code(traceback.format_exc())
        st.stop()

    st.subheader("Pilih Model Embedding")
    model_options = {
        "all-MiniLM-L6-v2": "Cepat dan ringan (384 dimensi)",
        "all-MiniLM-L12-v2": "Lebih akurat, tetap ringan (384 dimensi)",
        "all-distilroberta-v1": "Lebih akurat (768 dimensi)",
        "all-mpnet-base-v2": "Sangat akurat, lebih lambat (768 dimensi)",
        "paraphrase-multilingual-MiniLM-L12-v2": "Mendukung multibahasa (384 dimensi)"
    }
    
    default_model = st.session_state.get("selected_model", list(model_options.keys())[0])
    col1, col2 = st.columns([1, 4])    
    with col1:
        selected_model_name = st.selectbox(
            label="hidden",
            options=list(model_options.keys()), 
            index=list(model_options.keys()).index(default_model),
            key="selected_model",
            label_visibility="collapsed"
        )
    
    with col2:
        col2a, col2b = st.columns([1, 3])
        with col2a:
            run_embed = st.button("Go Embeddings", use_container_width=True)
        
        with col2b:
            if run_embed:
                try:
                    with st.spinner(f"Membuat embeddings dengan model {selected_model_name}..."):
                        model = SentenceTransformer(selected_model_name)
                        text_to_embed = df['Title'].fillna('') + " " + df['Abstract'].fillna('') + " " + df['Author Keywords'].fillna('')
                        paper_embeddings = model.encode(text_to_embed.tolist(), show_progress_bar=True)
                        index = create_faiss_index(paper_embeddings)
                        
                        if index is not None:
                            st.session_state["paper_embeddings"] = paper_embeddings
                            st.session_state["faiss_index"] = index
                            st.session_state["embed_success"] = True
                        else:
                            st.session_state["embed_success"] = False
                except Exception as e:
                    st.error(f"Error creating embeddings: {str(e)}")
                    st.error("Detail error:")
                    st.code(traceback.format_exc())
                    st.session_state["embed_success"] = False
            
            if st.session_state.get("embed_success", False):
                st.markdown(
                    "<div style='padding: 0.4rem 0.75rem; background-color: #d4edda; color: #155724; "
                    "border: 1px solid #c3e6cb; border-radius: 0.25rem; width:100%; display: block; "
                    "font-size: 1rem; vertical-align: middle;'>Embeddings berhasil dibuat.</div>",
                    unsafe_allow_html=True
                )
    
    st.caption(model_options[selected_model_name])

# Ambil state dengan pengecekan error
try:
    df = st.session_state.get("df", None)
    index = st.session_state.get("faiss_index", None)
    paper_embeddings = st.session_state.get("paper_embeddings", None)
    selected_model = st.session_state.get("selected_model", "all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Error accessing session state: {str(e)}")
    st.error("Detail error:")
    st.code(traceback.format_exc())
    df, index, paper_embeddings, selected_model = None, None, None, "all-MiniLM-L6-v2"

if df is not None and index is not None:
    st.header("Search Papers")
    col1, col2 = st.columns([1.5, 1])    
    with col1:
        inclusion_criteria = st.text_area("Kriteria Inklusi", height=200)
    with col2:
        exclusion_criteria = st.text_area("Kriteria Eksklusi (opsional)", height=200)
    
    threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.60, step=0.01)
    
    if st.button("Proses"):
        if not inclusion_criteria:
            st.warning("Mohon isi setidaknya kriteria inklusi.")
        else:
            try:
                with st.spinner("Searching..."):
                    model = SentenceTransformer(selected_model)
                    
                    # Proses inklusi
                    query_embedding_inclusion = model.encode([inclusion_criteria])
                    faiss.normalize_L2(query_embedding_inclusion)
                    inclusion_indices, inclusion_scores = search_with_threshold(index, query_embedding_inclusion, threshold)
                    
                    if len(inclusion_indices) == 0:
                        st.session_state["results_df"] = pd.DataFrame()
                        st.session_state["bibtex_output"] = ""
                        st.warning("Tidak ditemukan paper yang sesuai kriteria inklusi dan threshold.")
                    else:
                        inclusion_set = set(inclusion_indices)
                        
                        # Proses eksklusi jika ada
                        if exclusion_criteria.strip():
                            query_embedding_exclusion = model.encode([exclusion_criteria])
                            faiss.normalize_L2(query_embedding_exclusion)
                            exclusion_indices, _ = search_with_threshold(index, query_embedding_exclusion, threshold)
                            exclusion_set = set(exclusion_indices)
                            final_indices = list(inclusion_set - exclusion_set)
                            final_scores = [inclusion_scores[list(inclusion_indices).index(i)] for i in final_indices]
                        else:
                            final_indices = list(inclusion_indices)
                            final_scores = list(inclusion_scores)
                        
                        if len(final_indices) == 0:
                            st.session_state["results_df"] = pd.DataFrame()
                            st.session_state["bibtex_output"] = ""
                            st.warning("Semua hasil yang relevan dengan inklusi juga cocok dengan eksklusi, tidak ada hasil ditampilkan.")
                        else:
                            results = []
                            for i, (idx, score) in enumerate(zip(final_indices, final_scores)):
                                try:
                                    paper = df.iloc[idx]
                                    results.append({
                                        "No": i + 1,
                                        "Paper": format_apa_citation(paper),
                                        "Cited by": paper['Cited by'],
                                        "Similarity": f"{score:.3f}"
                                    })
                                except Exception as e:
                                    st.error(f"Error processing paper at index {idx}: {str(e)}")
                                    continue
                            
                            results_df = pd.DataFrame(results)
                            doi_list = [df.iloc[idx]['DOI'] for idx in final_indices if pd.notna(df.iloc[idx]['DOI'])]
                            doi_output = ', '.join(doi_list)
                            st.session_state["results_df"] = results_df
                            st.session_state["bibtex_output"] = doi_output
            except Exception as e:
                st.error(f"Error during search process: {str(e)}")
                st.error("Detail error:")
                st.code(traceback.format_exc())
                st.session_state["results_df"] = pd.DataFrame()
                st.session_state["bibtex_output"] = ""
    
    if "results_df" in st.session_state and not st.session_state["results_df"].empty:
        st.success(f"Ditemukan {len(st.session_state['results_df'])} paper relevan.")
        st.download_button(
            label="\U0001F4E5 Download DOIs",
            data=st.session_state["bibtex_output"],
            file_name="filtered_papers_doi.txt",
            mime="text/plain"
        )
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
        st.markdown(st.session_state["results_df"][["No", "Paper", "Cited by", "Similarity"]].to_html(index=False, escape=False), unsafe_allow_html=True)

st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload CSV file  
2. Pilih model dan klik **Go Embeddings**  
3. Masukkan kriteria inklusi dan eksklusi  
4. Atur threshold dan klik **Proses**  
5. Download hasil DOI dan lihat tabel hasil
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
