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
    /* Kurangi padding atas pada container utama */
    .main .block-container {
        padding-top: 1rem !important;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 95%;
    }

    /* Hilangkan margin default pada h1 (judul) */
    h1 {
        margin-top: 0rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)



# Initialize the sentence transformer model
@st.cache_resource
def load_model():
    #return SentenceTransformer('all-MiniLM-L6-v2')
    return SentenceTransformer('all-MiniLM-L12-v2')

model = load_model()

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
    
    # Format authors (simple handling - in reality would need to parse properly)
    if pd.notna(authors):
        author_list = authors.split(', ')
        if len(author_list) > 1:
            formatted_authors = ', '.join(author_list[:-1]) + ', & ' + author_list[-1]
        else:
            formatted_authors = authors
    else:
        formatted_authors = ''
    
    # Format title
    formatted_title = title + '.' if title else ''
    
    # Format source title
    formatted_source = source_title if source_title else ''
    
    # Format volume and issue
    volume_issue = ''
    if pd.notna(volume):
        volume_issue += f"{volume}"
    if pd.notna(issue):
        volume_issue += f"({issue})"
    if volume_issue:
        volume_issue += ', '
    
    # Format pages
    pages = ''
    if pd.notna(page_start):
        pages = f"{page_start}"
        if pd.notna(page_end):
            pages += f"-{page_end}"
    
    # Format DOI
    doi_str = ''
    if pd.notna(doi):
        doi_str = f" https://doi.org/{doi}"
    
    citation = f"{formatted_authors} ({year}). {formatted_title} {formatted_source}, {volume_issue}{pages}.{doi_str}"
    
    # Clean up multiple spaces
    citation = re.sub(' +', ' ', citation).strip()
    return citation

# Function to create and search FAISS index
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def search_with_threshold(index, query_embedding, threshold, k=1000):
    # We set k to a high value to ensure we get all possible matches above threshold
    distances, indices = index.search(query_embedding, k)
    
    # Filter results based on threshold
    mask = distances[0] >= threshold
    filtered_indices = indices[0][mask]
    filtered_distances = distances[0][mask]
    
    # Sort by distance (descending)
    sorted_order = np.argsort(-filtered_distances)
    return filtered_indices[sorted_order], filtered_distances[sorted_order]

# Streamlit app
#st.title("Study Selection using Sentence Embeddings")
st.markdown("<h1>Study Selection using <em>Sentence Embeddings</em></h1>", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None and "faiss_index" not in st.session_state:
    # Read the CSV file
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    df = pd.read_csv(stringio)
    
    st.session_state["df"] = df
    
    with st.spinner("Creating embeddings..."):
        text_to_embed = df['Title'].fillna('') + " " + df['Abstract'].fillna('') + " " + df['Author Keywords'].fillna('')
        paper_embeddings = model.encode(text_to_embed.tolist(), show_progress_bar=True)
        index = create_faiss_index(paper_embeddings)

        # Simpan di session_state
        st.session_state["paper_embeddings"] = paper_embeddings
        st.session_state["faiss_index"] = index

    st.success("Embeddings created and FAISS index built.")

# Gunakan variabel dari session_state
df = st.session_state.get("df", None)
index = st.session_state.get("faiss_index", None)
paper_embeddings = st.session_state.get("paper_embeddings", None)


# Search interface
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
                # Combine research question and keywords for query
                query_text = f"{research_question} {keywords}"
                query_embedding = model.encode([query_text])
                
                # Search with threshold
                result_indices, result_distances = search_with_threshold(index, query_embedding, threshold)
                
                if len(result_indices) == 0:
                    st.warning("No papers found above the specified threshold.")
                else:
                    st.success(f"Found {len(result_indices)} papers with similarity ≥ {threshold:.2f}")
                    
                    # Prepare results
                    results = []
                    for i, (idx, score) in enumerate(zip(result_indices, result_distances)):
                        paper = df.iloc[idx]
                        results.append({
                            "No": i + 1,
                            "Paper": format_apa_citation(paper),
                            #"Author Keywords": paper['Author Keywords'],
                            "Cited by": paper['Cited by'],
                            #"DOI": paper['DOI'],
                            "Similarity": f"{score:.3f}"
                        })
                    
                    # Display as table
                    results_df = pd.DataFrame(results)
                    #st.dataframe(
                        ##results_df[["No", "Paper", "Author Keywords", "Cited by", "DOI", "Similarity"]],
                        #results_df[["No", "Paper", "Cited by", "DOI", "Similarity"]],
                        #column_config={
                            #"Paper": "Paper (APA 7th Format)",
                            ##"Author Keywords": "Keywords",
                            #"Cited by": "Citations",
                            #"Similarity": "Similarity Score"
                        #},
                        #use_container_width=True,
                        #hide_index=True
                    #)

                    # CSS kustom untuk kontrol lebar kolom
                    st.markdown("""
                    <style>
                    table {
                        width: 100%;
                        table-layout: fixed;
                    }
                    thead th:nth-child(1), tbody td:nth-child(1) {
                        width: 40px;
                        text-align: center;
                    }
                    thead th:nth-child(3), tbody td:nth-child(3) {
                        width: 70px;
                        text-align: center;
                    }
                    thead th:nth-child(2), tbody td:nth-child(2) {
                        width: 600px;
                        word-wrap: break-word;
                        text-align: left;
                    }
                    thead th:nth-child(4), tbody td:nth-child(4) {
                        width: 100px;
                        word-wrap: break-word;
                        text-align: center;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Tampilkan hasil dengan HTML
                    st.markdown(
                        #results_df[["No", "Paper", "Cited by", "DOI", "Similarity"]]
                        results_df[["No", "Paper", "Cited by", "Similarity"]]
                        .to_html(index=False, escape=False), unsafe_allow_html=True
                    )

                    

# Instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload a CSV file with research papers
2. The app will create embeddings for the papers
3. Enter your research question and/or keywords
4. Set a similarity threshold (0.0-1.0)
5. Click "Proses" to find relevant papers

**Output columns:**
- No: Result number
- Paper: Formatted in APA 7th style
- Cited by: Citation count
- DOI: Digital Object Identifier
- Similarity: Match score (higher is better)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: left; font-size: 90%;'>
© <em>tailored by</em>  
<a href='mailto:adiwjj@uima.ac.id'>adiwjj</a> 2025
</div>
""", unsafe_allow_html=True)