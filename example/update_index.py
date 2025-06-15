## This function  reads data from Azure Blob, preprocess and updates the index 
from openai import AzureOpenAI as AzureOpenAIo
from azure.identity import DefaultAzureCredential, get_bearer_token_provider, ManagedIdentityCredential
from azure.core.credentials import AzureKeyCredential
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore, IndexManagement, MetadataIndexFieldType
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from tenacity import retry, wait_random_exponential, stop_after_attempt  
from typing import Optional
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient  
from llama_index.readers.azstorage_blob import AzStorageBlobReader
# from azure.search.documents.models import Vector  # only in version azure-search-documents==11.4.0b8
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (  
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    SemanticField,  
    SearchField,  
    VectorSearch) 
import os
import time
import logging

logging.basicConfig(level=logging.DEBUG)
print("testing 55")


# Obtain credentials
service_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
index_name = os.environ["AZURE_SEARCH_INDEX"]
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_embedding_model = os.environ["AZURE_OPENAI_EMBEDDING_MODEL"]
azure_openai_embedding_deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
azure_openai_chatgpt_deployment = os.environ["MODEL_NAME"]
azure_openai_api_version = os.environ["API_VERSION"]
embedding_dimensions = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", 1536))
logging.info('obtain credential completed')


if os.environ["RUN_ENV"] == 'local':
    credential = AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]) 
    azure_openai_key = os.environ["AZURE_OPENAI_KEY"] 

    # Configure openai instance
    client = AzureOpenAIo(
        api_key = azure_openai_key,
        azure_endpoint = azure_openai_endpoint,
        api_version = azure_openai_api_version )
    logging.info('openai object')

    # Configure embeddings instance 
    embeddings = AzureOpenAIEmbedding(
        model_name=azure_openai_embedding_model,
        deployment_name=azure_openai_embedding_deployment,
        api_version=azure_openai_api_version,
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_key)
    logging.info('embeddings created')

    llm = AzureOpenAI(
        deployment_name=azure_openai_chatgpt_deployment,
        api_version=azure_openai_api_version,
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_key)
    logging.info('llm object created')

else: 
    uami_client_id = os.getenv('UAMI_CLIENT_ID') 
    token_provider = get_bearer_token_provider(DefaultAzureCredential(managed_identity_client_id=uami_client_id), "https://cognitiveservices.azure.com/.default")
    credential = DefaultAzureCredential(managed_identity_client_id=uami_client_id)
    # credential = ManagedIdentityCredential(client_id = uami_client_id)
    # credential = AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]) # This is temp only for testing cannot use in preprod 
    
    # Configure openai instance
    client = AzureOpenAIo(
        azure_ad_token_provider=token_provider,
        azure_endpoint = azure_openai_endpoint,
        api_version = azure_openai_api_version )
    logging.info('openai object')

    # Configure embeddings instance 
    embeddings = AzureOpenAIEmbedding(
        model_name=azure_openai_embedding_model,
        deployment_name=azure_openai_embedding_deployment,
        api_version=azure_openai_api_version,
        azure_endpoint=azure_openai_endpoint,
        azure_ad_token_provider=token_provider)
    logging.info('embeddings created')

    llm = AzureOpenAI(
        deployment_name=azure_openai_chatgpt_deployment,
        api_version=azure_openai_api_version,
        azure_endpoint=azure_openai_endpoint,
        azure_ad_token_provider=token_provider)
    logging.info('llm object created')




@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text):
    response = client.embeddings.create(input=text, model=azure_openai_embedding_deployment)
    time.sleep(10)
    embeddings = response.data[0].embedding
    return embeddings


def vector_similarity_search(query,service_endpoint, index_name, credential, top_n=2, hybrid=False, semantic=False):

    # Perform vector similarity search 
    search_client = SearchClient(service_endpoint, index_name, credential=credential)
    # vector = Vector(value=generate_embeddings(query), k=3, fields="contentVector")
    vector = VectorizedQuery(vector=generate_embeddings(query), k_nearest_neighbors=3, fields="content_vector")
    vector = [vector]

    if hybrid:
        search_query = query
    else:
        search_query = None
  

    results = search_client.search(  
        search_text=search_query,  
        vector_queries= vector,
        select=["content"],)  
  
    if semantic and results.get_answers() is not None:
        semantic_answers = results.get_answers()
        for answer in semantic_answers:
            if answer.highlights:
                print(f"Semantic Answer: {answer.highlights}")
            else:
                print(f"Semantic Answer: {answer.text}")
            print(f"Semantic Answer Score: {answer.score}\n")

    list_results = []
    count = 0
    for result in results:
        if count < top_n: 
            # print('score:', result['@search.score'])
            # print('content:', result['content'])
            # print("---------------------------------------------")
            list_results.append(result['content'])
        count = count + 1

    return list_results



def index_documents(container_name: Optional[str] = None): # pragma no cover
    
    logging.info("Processing documents .......")
        # Upload, vectorize, and index documents
    metadata_fields = {
            "author": "author",
            "theme": ("topic", MetadataIndexFieldType.STRING),
            "director": "director",
        }

    logging.info(f"index name: {index_name}")

    vector_store = AzureAISearchVectorStore(
        search_or_index_client=SearchIndexClient(endpoint=service_endpoint, credential=credential),
        filterable_metadata_field_keys=metadata_fields,
        index_name=index_name,
        index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
        id_field_key="id",
        chunk_field_key="content",
        embedding_field_key="content_vector",
        metadata_string_field_key="metadata",
        doc_id_field_key="doc_id",
        embedding_dimensionality=embedding_dimensions,
        language_analyzer="en.lucene",
        vector_algorithm_type="exhaustiveKnn")

    logging.info("vector store obj created")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    Settings.llm = llm
    Settings.embed_model = embeddings
    directory = os.getenv('LOCAL_PATH')
    logging.info("storage contect and settings defined")

    if os.environ["RUN_ENV"] == 'local':
        documents = SimpleDirectoryReader(directory).load_data()
    else:
        loader = AzStorageBlobReader(
            # container_name=os.getenv("BLOB_CONTAINER_NAME_2"),
            container_name=container_name,
            account_url=os.getenv('BLOB_ACC_URL'),
            credential=ManagedIdentityCredential(client_id = uami_client_id))
        logging.info("Azure loader created")

        documents = loader.load_data()
        logging.info("documents loaded")

    index = VectorStoreIndex.from_documents(documents=documents,storage_context=storage_context)
    logging.info(f"Index {index_name} created .......")

    return "indexing completed"


if __name__ == "__main__":
    response = index_documents()
