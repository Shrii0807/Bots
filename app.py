import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA  # Correct import here

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    # Add a header
    st.header("Chat with multiple PDFs :books:")

    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Upload Your PDFs here and click on 'Process'", type="pdf", accept_multiple_files=True)

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    # Get the PDFs text
                    raw_text = get_pdf_text(pdf_docs)

                    # Display first 500 characters of raw text to prevent overwhelming the user
                    st.write("Extracted text preview:")
                    st.write(raw_text[:500])

                    # Get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    st.write(f"Text has been split into {len(text_chunks)} chunks.")

                    # Create a vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # Now allow users to ask questions
                    query = st.text_input("Ask a question about your documents:")

                    if query:
                        if vectorstore:
                            # Create the RetrievalQA chain
                            qa_chain = RetrievalQA.from_chain_type(llm="openai", retriever=vectorstore.as_retriever())

                            # Run the query
                            answer = qa_chain.run(query)
                            st.write("Answer:", answer)
            else:
                st.error("Please upload at least one PDF document.")

if __name__ == "__main__":
    main()
