import streamlit as st
import pandas as pd
import pdfplumber
import google.generativeai as genai
from dotenv import load_dotenv
import os
import html
import re


# ====== Konfigurasi API ======
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm_model = genai.GenerativeModel(model_name="gemini-2.0-flash")


# ====== Fungsi Parsing Kriteria ======
def parse_criteria(text):
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    bullet_pattern = re.compile(r"^(\-|\*|\d+\)|\d+\.)\s*(.+)$")
    
    bullets = []
    narasi_awal = None
    
    for line in lines:
        match = bullet_pattern.match(line)
        if match:
            bullets.append(match.group(2).strip())
        else:
            if narasi_awal is None:
                narasi_awal = line
            else:
                narasi_awal += " " + line
    
    if narasi_awal and bullets:
        return [f"{narasi_awal} {b}" for b in bullets]
    else:
        return bullets if bullets else lines


# ====== Fungsi Ekstraksi PDF ======
def extract_pdf_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# ====== Fungsi Tanya ke Gemini per Kriteria (tabel) ======
def ask_gemini_per_kriteria(text, criteria_list, language):
    results = []
    for crit in criteria_list:
        if language == "ID":
            prompt = f"""
            Kamu adalah asisten yang membantu melakukan ekstraksi data dari paper ilmiah.
            Teks berikut diambil dari sebuah paper:
            ---
            {text}
            ---
            Tolong jawab secara naratif dalam 1 paragraf untuk kriteria berikut:
            "{crit}"
            Jika tidak ada informasi, jawab "Tidak ditemukan".
            Jawaban dalam bahasa Indonesia. Singkat namun tetap informatif.
            """
        else:
            prompt = f"""
            You are an assistant that helps extract data from a scientific paper.
            The following text is from a paper:
            ---
            {text}
            ---
            Please answer in a single narrative paragraph for the following criterion:
            "{crit}"
            If no information is found, write "Not found".
            Answer in English. Concise yet informative.
            """
        
        response = llm_model.generate_content(prompt)
        results.append({"Kriteria": crit, "Hasil Ekstraksi": response.text.strip()})
    
    return pd.DataFrame(results)


# ====== Fungsi Tanya ke Gemini Narasi Tunggal ======
def ask_gemini_single_narrative(text, criteria_list, language):
    if language == "ID":
        prompt = f"""
        Kamu adalah asisten yang membantu melakukan ekstraksi data dari paper ilmiah.
        Teks berikut diambil dari sebuah paper:
        ---
        {text}
        ---
        Berikut adalah daftar kriteria yang harus kamu jawab:
        {chr(10).join(f"{i+1}. {crit}" for i, crit in enumerate(criteria_list))}
        
        Pertama, tuliskan sitasi paper ini dalam format APA 7th (hanya penulis & tahun), misalnya: "(Smith et al, 2020):"
        Setelah itu, buat jawaban dalam {len(criteria_list) + 2} paragraf:
        1. Paragraf pertama: pengantar singkat namun jelas.
        2. Paragraf 2 hingga {len(criteria_list)+1}: masing-masing menjawab satu kriteria sesuai urutan.
        3. Paragraf terakhir: ringkasan/penutup. Hindari term:  In summary, In conclusion dan sejenisnya.

        Jawaban dalam bahasa Indonesia. Singkat namun tetap informatif.
        """
    else:
        prompt = f"""
        You are an assistant that helps extract data from a scientific paper.
        The following text is from a paper:
        ---
        {text}
        ---
        Here are the criteria to address:
        {chr(10).join(f"{i+1}. {crit}" for i, crit in enumerate(criteria_list))}
        
        First, write the citation of this paper in APA 7th format (authors & year only), e.g., "(Smith et al, 2020):"
        then, write the answer in {len(criteria_list) + 2} paragraphs:
        1. First paragraph: brief but clear introduction.
        2. Paragraphs 2 to {len(criteria_list)+1}: each answers one criterion in order.
        3. Final paragraph: summary/conclusion. Avoid terms such as 'In summary', 'In conclusion', and similar.

        Answer in English. Concise yet informative.
        """
    
    response = llm_model.generate_content(prompt)
    return response.text.strip()


# ====== Fungsi Render Tabel HTML dengan Wrap ======
def render_table_with_wrap(df):
    table_html = """
    <style>
    table {
        border-collapse: collapse;
        width: 100%;
        table-layout: fixed;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
        vertical-align: top;
        white-space: normal;
        word-wrap: break-word;
    }
    th {
        background-color: #f2f2f2;
    }
    th.kriteria, td.kriteria {
        width: 25%;
    }
    th.hasil, td.hasil {
        width: 75%;
    }
    </style>
    <table>
        <tr>
            <th class="kriteria">Kriteria</th>
            <th class="hasil">Hasil Ekstraksi</th>
        </tr>
    """
    for _, row in df.iterrows():
        kriteria = html.escape(str(row["Kriteria"]))
        hasil = html.escape(str(row["Hasil Ekstraksi"]))
        table_html += f"<tr><td class='kriteria'>{kriteria}</td><td class='hasil'>{hasil}</td></tr>"
    table_html += "</table>"
    return table_html


# ====== UI Streamlit ======
st.set_page_config(page_title="SLR Extraction - Direct Gemini", layout="wide")
st.title("📄 SLR Data Extraction (via Gemini)")

hide_github_icon = """
    <style>
        [data-testid="stToolbar"] {visibility: hidden;}
    </style>
"""
st.markdown(hide_github_icon, unsafe_allow_html=True)


# Upload File + simpan di session_state
uploaded_file = st.file_uploader("Upload Paper (PDF)", type="pdf")
if uploaded_file:
    st.session_state.uploaded_file_content = extract_pdf_text(uploaded_file)

criteria_input = st.text_area(
    "Masukkan daftar kriteria. Jika kriteria lebih dari satu, pisahkan dengan bullet (- atau *), atau numbering.",
    "",
    height=150
)

col1, col2, _ = st.columns([1, 1, 4])
with col1:
    answer_lang = st.selectbox("Answer in", ["EN", "ID"])
with col2:
    mode = st.selectbox("Mode Output", ["Tabel", "Narasi Tunggal"])

if st.button("Ekstrak Data"):
    if "uploaded_file_content" not in st.session_state:
        st.warning("Silakan upload file terlebih dahulu.")
    elif not criteria_input.strip():
        st.warning("Silakan masukkan kriteria.")
    else:
        with st.spinner("Membaca dan mengirim ke Gemini..."):
            criteria_list = parse_criteria(criteria_input)
            
            if mode == "Tabel":
                df_results = ask_gemini_per_kriteria(
                    st.session_state.uploaded_file_content,
                    criteria_list,
                    answer_lang
                )
                st.markdown("### Hasil Ekstraksi")
                st.markdown(render_table_with_wrap(df_results), unsafe_allow_html=True)

                csv_data = df_results.to_csv(index=False)
                st.download_button(
                    label="💾 Download Hasil Ekstraksi (CSV)",
                    data=csv_data.encode("utf-8"),
                    file_name="hasil_ekstraksi_gemini.csv",
                    mime="text/csv"
                )
            else:
                hasil_narasi = ask_gemini_single_narrative(
                    st.session_state.uploaded_file_content,
                    criteria_list,
                    answer_lang
                )
                st.markdown("### Hasil Ekstraksi (Narasi Tunggal)")
                st.write(hasil_narasi)
