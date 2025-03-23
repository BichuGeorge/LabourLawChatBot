from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import os
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')

def load_pdf(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

def create_index(text_chunks):
    pc = pinecone(api_key=PINECONE_API_KEY)
    embeddings = download_hugging_face_embeddings()
    index_name = "labourlawbot"
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)