import sys
import streamlit as st
import pdfplumber
from Resume_scanner import compare
import pandas as pd
# from dotenv import load_dotenv

def extract_pdf_data(file_path):
    data = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                data += text
    return data


def extract_text_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

# Initialize session state for tab index
if 'tab_index' not in st.session_state:
    st.session_state['tab_index'] = 0

# Function to switch tabs
def switch_tab(index):
    st.session_state['tab_index'] = index

def main():
    flag = 'HuggingFace-BERT'
    tab1, tab2, tab3 = st.tabs(["**Home**", "**Results**", "**Graphs**"])

    # Tab Home
    with tab1:
        st.title("Match Resume Assistant 🤖")
        JD = st.text_area("**Enter the job description:**")
        document_count = st.text_input("Nro of 'RESUMES' to return")
        uploaded_files = st.file_uploader(
            '**Choose your resume.pdf file:** ', type="pdf", accept_multiple_files=True)
        comp_pressed = st.button(label="Analyze!", type="primary")

        if comp_pressed and uploaded_files:
            with st.spinner("Wait for it ..."):
                uploaded_file_paths = [extract_pdf_data(file) for file in uploaded_files]
                score, summary = compare(uploaded_file_paths, JD, flag)

        st.divider()
        documents = []
        if comp_pressed and uploaded_files:
            for i in range(len(score)):
                my_dict = {"filename": uploaded_files[i].name, "score": float(score[i]), "summary": summary[i]}
                documents.append(my_dict)
            sorted_list = sorted(documents, key=lambda x: x['score'], reverse=True)
            sorted_list_filtered = sorted_list[:int(document_count)]
            for document in sorted_list_filtered:
                with st.expander("File: " + document["filename"]):
                    st.info("**Match Score** : " + str(document["score"]))
                    st.write("**Summary** : ", document["summary"])

    with tab2:
        st.header("Results")
        if comp_pressed and uploaded_files:
            df = pd.DataFrame(sorted_list)
            # df_filtered= df[["filename","score"]]
            st.dataframe(df,
                column_config={
                    "filename": "Document Name",
                    "score": "Score",
                    "summary": "Summary Resume"
                },
                hide_index=True
            )
            for document in sorted_list:
                with st.expander("File: " + document["filename"]):
                    st.info("**Match Score** : " + str(document["score"]))
                    st.write("**Summary** : ", document["summary"])
            st.success("Hope I was able to save your time 🤩")
    with tab3:
        st.header("Graphs")
        if comp_pressed and uploaded_files:
            st.area_chart(df, x="filename", y="score")
            st.bar_chart(df, x="filename", y="score")
            st.scatter_chart(df, x="filename", y="score")

if __name__=='__main__':
    main()