from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
VECTORSTORE_PATH = "vectorstore"
LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
def load_qa_chain():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url="http://127.0.0.1:11434")
    vectorstore = FAISS.load_local(VECTORSTORE_PATH,embeddings,allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = OllamaLLM(model=LLM_MODEL, base_url="http://127.0.0.1:11434")
    prompt_template = """You are a technical assistant for the SD40-2 Locomotive Service Manual.  Use the context below to answer the question. If the answer is not in the context, say so clearly.

Context: {context}

Question: {question}

Answer:"""
    prompt = PromptTemplate(template=prompt_template,input_variables=["context", "question"])   
    chain = RetrievalQA.from_chain_type(llm=llm,retriever=retriever,chain_type="stuff",chain_type_kwargs={"prompt": prompt},return_source_documents=True)
    return chain