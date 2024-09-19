import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import faiss

# Load environment variables from .env file
load_dotenv()

# Now retrieve the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("AZURE_OPENAI_API_ENDPOINT")
openai_api_version = os.getenv("OPENAI_API_VERSION")
model_name = os.getenv("MODEL_NAME")
deployment_name = os.getenv("DEPLOYMENT_NAME")
# Initialize AzureChatOpenAI LLM
llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    model_name=model_name
)

# Function to load PDF and extract text
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to chunk PDF text
def chunk_pdf(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.create_documents([text])
    return chunks

# Function to embed chunks using Azure OpenAI embeddings
def embed_chunks(chunks):
    embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-large"# Your specific deployment name
    )
    embedded_chunks = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    return embedded_chunks

# Function to store embeddings in FAISS vector store
def store_embeddings(embedded_chunks):
    dimension = len(embedded_chunks[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embedded_chunks)
    return index

# Function to embed the user query
def embed_query(query):
    embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-large"
    )
    embedded_query = embeddings.embed_query(query)
    return embedded_query

# Function to perform similarity search
def similarity_search(index, embedded_query, top_k=5):
    _, top_indices = index.search(embedded_query, top_k)
    return top_indices

# Function to retrieve relevant chunks
def retrieve_chunks(chunks, top_indices):
    retrieved_chunks = [chunks[i] for i in top_indices]
    context = " ".join([chunk.page_content for chunk in retrieved_chunks])
    return context

# Function to generate answer using AzureChatOpenAI and a PromptTemplate
def generate_answer_with_template(context, query):
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""\
        Use the following context to answer the question.

        Context: {context}

        Question: {question}

        Answer:
        """
    )
    
    # Format the prompt with context and question
    prompt = prompt_template.format(context=context, question=query)
    
    # Query the LLM and get the response
    response = llm([{"role": "user", "content": prompt}])
    return response['choices'][0]['message']['content']

# Streamlit UI
st.title("Document QA using RAG")

# File uploader for the PDF
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

# Text input for the question
query = st.text_input("Ask a question")

# If a PDF and a question are provided
if pdf_file and query:
    with st.spinner('Processing...'):
        # Load and process the PDF
        text = load_pdf(pdf_file)
        chunks = chunk_pdf(text)
        embedded_chunks = embed_chunks(chunks)
        index = store_embeddings(embedded_chunks)
        
        # Embed the user query and search for similar chunks
        embedded_query = embed_query(query)
        top_indices = similarity_search(index, embedded_query)
        
        # Retrieve the most relevant chunks
        context = retrieve_chunks(chunks, top_indices)
        
        # Generate the answer from the LLM
        answer = generate_answer_with_template(context, query)
        
        # Display the answer
        st.success("Answer:")
        st.write(answer)
