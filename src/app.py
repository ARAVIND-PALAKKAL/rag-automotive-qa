import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
VECTORSTORE_PATH = "vectorstore"
LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
from chain import load_qa_chain
@st.cache_resource
def cached_chain():
    return load_qa_chain()
st.set_page_config(page_title="SD40-2 Manual QA", page_icon="🚂")
st.title("SD40-2 Locomotive Manual Assistant")
st.caption("Ask anything about the SD40-2 service manual")
question = st.text_input("Your question:", placeholder="e.g. What is the oil pressure specification?")
if question:
    chain = cached_chain()
    result = chain.invoke({"query": question})
    st.markdown("### Answer")
    st.write(result["result"])
    with st.expander("Source pages"):
        for doc in result["source_documents"]:
            st.markdown(f"**Page {doc.metadata.get('page', '?')}**")
            st.write(doc.page_content)