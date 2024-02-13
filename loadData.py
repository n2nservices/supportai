import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import AstraDB
from dotenv import load_dotenv

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_ENDPOINT")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# PROD environment
ASTRA_DB_PROD_KEYSPACE = os.environ.get("ASTRA_DB_PROD_KEYSPACE")
ASTRA_DB_PROD_CLICKUP_COLLECTION = os.environ.get("ASTRA_DB_PROD_CLICKUP_COLLECTION")

# DEV Environenet
# ASTRA_DB_DEV_KEYSPACE = os.environ.get("ASTRA_DB_DEV_KEYSPACE")
# ASTRA_DB_DEV_CLICKUP_COLLECTION = os.environ.get("ASTRA_DB_DEV_CLICKUP_COLLECTION")


# Load, chunk and index the contents of the blog.
loader = CSVLoader(file_path='./supportai-prod.csv', encoding='utf-8')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


embedding = OpenAIEmbeddings()

vstore = AstraDB(
    embedding=embedding,
    namespace=ASTRA_DB_PROD_KEYSPACE,
    collection_name=ASTRA_DB_PROD_CLICKUP_COLLECTION,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
)

# vstore.delete_collection()
supportTickets = vstore.add_documents(splits)
