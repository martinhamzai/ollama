import os
import tempfile

import ollama
import chromadb
import streamlit as st

from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# the prompt used to instruct the llm on how to recieve prompts and provide feedback
# Constrained it to only provide answers based on the given context and not any external knowledge or assumptions (RAG)
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

# obtain/create the vector db
def get_vector_collection() -> chromadb.Collection:
    # define the embedding function used for this vector db
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest"
    )

    # store the database in a local directory
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    # return the collection with the defined embedding function and distance formula
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"}
    )

# take each split in the documents and add to vector db
def add_to_vector_collection(all_splits: list[Document], file_name: str):
    # obtain the vector collection that the splits will be stored in
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    # add or update splits into the collection
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Data added to the vector store!")

# takes the uploaded file and splits into chunks
def process_document(uploaded_file: UploadedFile) -> list[Document]:
    # convert to temp file
    with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # load the pdf and store in docs
    loader = PyMuPDFLoader(temp_file_path)
    docs = loader.load()
    os.unlink(temp_file_path)  # Delete temp file

    # define the parameters for the text splitter
    # chunk sizes of 400 with 100 character overlap between chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    # split the documents into chunks which yields a list of document chunks
    return text_splitter.split_documents(docs)

# search through the vector db for the 10 most similar splits to our prompt
def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

# feeds context and prompt to an llm to generate response
def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2:3b",
        # flowing output
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}"
            }
        ]
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

# re ranks documents using a cross-encoder model and returns the text of the top 3 most relevant
def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids

# running script
if __name__ == "__main__":
    # sidebar creation with streamlit
    with st.sidebar:
        st.set_page_config(page_title="RAG Question Answer")
        
        # file drop area
        uploaded_file = st.file_uploader(
            "**Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False
        )

        process = st.button(
            "Process"
        )

        if uploaded_file and process:
            # change all space indicating characters to '_'
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            # split the document into chunks
            all_splits = process_document(uploaded_file)
            # using the document splits and normaized file name, add it to a vector database
            add_to_vector_collection(all_splits, normalize_uploaded_file_name)

    st.header("RAG Question Answer")
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button(
        "Ask"
    )

    if ask and prompt:
        # get the 10 most similar results
        results = query_collection(prompt)
        # provide the most similar result as the context for the llm
        context = results.get("documents")[0]
        # use cross-encoder to obtain more relevant text
        relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
        response = call_llm(context=relevant_text, prompt=prompt)
        st.write_stream(response)

        with st.expander("See retrieved documents"):
            st.write(results)

        with st.expander("See most relevant document ids"):
            st.write(relevant_text_ids)
            st.write(relevant_text)