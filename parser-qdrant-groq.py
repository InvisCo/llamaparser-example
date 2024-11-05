# ruff: noqa: E402
import os
import nest_asyncio

nest_asyncio.apply()

# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv

load_dotenv()

##### LLAMAPARSE #####
from llama_parse import LlamaParse

llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")


import pickle


# Define a function to load parsed data if available, or parse if not
def load_or_parse_data():
    data_file = "./data/parsed_data.pkl"

    if os.path.exists(data_file):
        # Load the parsed data from the file
        with open(data_file, "rb") as f:
            parsed_data = pickle.load(f)
    else:
        # Perform the parsing step and store the result in llama_parse_documents
        llama_parse_documents = LlamaParse(
            api_key=llamaparse_api_key, result_type="markdown"
        ).load_data(["./data/saasbook-1.2.2_2.pdf"])

        # Save the parsed data to a file
        with open(data_file, "wb") as f:
            pickle.dump(llama_parse_documents, f)

        # Set the parsed data to the variable
        parsed_data = llama_parse_documents

    return parsed_data


# Call the function to either load or parse the data
llama_parse_documents = load_or_parse_data()


len(llama_parse_documents)

llama_parse_documents[0].text[:100]

type(llama_parse_documents)

######## QDRANT ###########

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

import qdrant_client

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

######### FastEmbedEmbeddings #############

# by default llamaindex uses OpenAI models
from llama_index.embeddings.fastembed import FastEmbedEmbedding

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")

""" embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    #model_name="llama2",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
) """

#### Setting embed_model other than openAI ( by default used openAI's model)
from llama_index.core import Settings

Settings.embed_model = embed_model

######### Groq API ###########

from llama_index.llms.groq import Groq

groq_api_key = os.getenv("GROQ_API_KEY")

llm = Groq(model="mixtral-8x7b-32768", api_key=groq_api_key)
# llm = Groq(model="gemma-7b-it", api_key=groq_api_key)

######### Ollama ###########

# from llama_index.llms.ollama import Ollama
# llm = Ollama(model="llama2", request_timeout=30.0)

#### Setting llm other than openAI ( by default used openAI's model)
Settings.llm = llm

client = qdrant_client.QdrantClient(
    api_key=qdrant_api_key,
    url=qdrant_url,
)

vector_store = QdrantVectorStore(client=client, collection_name="qdrant_rag")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents=llama_parse_documents, storage_context=storage_context, show_progress=True
)

#### PERSIST INDEX #####
# index.storage_context.persist()

# storage_context = StorageContext.from_defaults(persist_dir="./storage")
# index = load_index_from_storage(storage_context)

# create a query engine for the index
query_engine = index.as_query_engine()

# query the engine
# query = "what is the common stock balance as of Balance as of March 31, 2022?"
query = "what is the letter of credit As of December 31, 2021 "
response = query_engine.query(query)
print(response)
