import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import AstraDB
from dotenv import load_dotenv

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_ENDPOINT")
ASTRA_DB_COLLECTION = os.environ.get("ASTRA_DB_COLLECTION")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# Load, chunk and index the contents of the blog.
loader = CSVLoader(file_path='supportai-qa.csv', encoding='utf-8')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)


embedding = OpenAIEmbeddings()

vstore = AstraDB(
    embedding=embedding,
    collection_name=ASTRA_DB_COLLECTION,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
)

# vstore.delete_collection()
supportTickets = vstore.add_documents(splits)

