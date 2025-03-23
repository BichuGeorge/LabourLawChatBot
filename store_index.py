from dotenv import load_dotenv
load_dotenv()
import os
from src.helper import create_index
from src.helper import load_pdf, text_split
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')

extracted_data = load_pdf(data='D:/training/ml_algorithms/Level_2_Project_Streamlit/ChatBot/ChatbotProject/Data/')

text_chunks = text_split(extracted_data)

# create_index(text_chunks)
