from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from langchain_groq import ChatGroq
import os

load_dotenv()
app = Flask(__name__)

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
groq_key = os.getenv("GROQ")

embeddings = download_hugging_face_embeddings()

groq_model = ChatGroq(
    model="qwen-2.5-32b",
    groq_api_key=groq_key)

index_name = "labourlawbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k":3
    }
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", system_prompt
        ),
        (
            "human", "{input}"
        )
    ]
)

question_answer_chain = create_stuff_documents_chain(
    groq_model,
    prompt
)
rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET","POST"])
def chat():
    msg = request.form['msg']
    input=msg
    print(input)
    response = rag_chain.invoke(
        {
            "input": msg
        }
    )
    print("Response : ", response['answer'])
    return str(response['answer'])

if __name__ == "__main__":
    app.run(
        host="localhost",
        port=5001,
        debug=True
    ) 