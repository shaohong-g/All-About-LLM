import sys
import pandas as pd
from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    Settings,
    SQLDatabase,
    get_response_synthesizer,
    PromptTemplate
)
from llama_index.core.node_parser import SentenceSplitter

from Config import Config, Mapping_Config, Custom_Prompt
from llama_utils import etl, PG_Database, Model, IncidentAgent


##################################
# Initialize
##################################
def initialize_pgvector():
    return PG_Database(Config.DB_HOSTNAME,Config.DB_PORT,Config.DB_USERNAME,Config.DB_PASSWORD,Config.DB_NAME, Config.AZURE_OPENAI_EMBEDDING_DIMENSIONS)

def initial_load(pg_db=None):
    if pg_db is None:
        pg_db = initialize_pgvector()
    df_sample_incidents = pd.read_excel(Config.SAMPLE_DATA_DIR + "test_sample_1_50_updated.xlsx", sheet_name=0) 
    df_sample_health_status = pd.read_excel(Config.SAMPLE_DATA_DIR + "health-status.xlsx", sheet_name=0)
    df_sample_app_linkage = pd.read_excel(Config.SAMPLE_DATA_DIR + "application-linkage.xlsx", sheet_name=0)

    # ETL
    incidents_etl_data = etl(df_sample_incidents, Mapping_Config.incidents_column_mapping, reset_index=False)
    app_linkage_etl_data = etl(df_sample_app_linkage, Mapping_Config.app_linkage_column_mapping, reset_index=False)
    health_status_etl_data = etl(df_sample_health_status, Mapping_Config.health_status_column_mapping, reset_index=False)

    pg_db.upsert_data(Config.INCIDENT_TABLE_NAME, incidents_etl_data, remove_update_keys=["created_on"])
    pg_db.upsert_data(Config.APP_LINKAGE_TABLE_NAME, app_linkage_etl_data, remove_update_keys=["created_on"])
    pg_db.upsert_data(Config.HEALTH_STATUS_TABLE_NAME, health_status_etl_data, remove_update_keys=["created_on"])
    return True

def initialize_incident_vector_store(pg_db=None, embedding_model= None):
    if pg_db is None:
        pg_db = initialize_pgvector()
    if embedding_model is None:
        model = Model()
        embedding_model = model.get_embedding_model(
            Config.AZURE_OPENAI_EMBEDDING_MODEL,
            Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            Config.AZURE_OPENAI_EMBEDDING_KEY,
            Config.AZURE_OPENAI_EMBEDDING_ENDPOINT,
            Config.AZURE_OPENAI_EMBEDDING_API_VERSION,
        )
    df_incidents = pg_db.read_table(Config.INCIDENT_TABLE_NAME)

    # Create Documents
    documents = []
    for _, row in df_incidents.iterrows():
        text_input = (
                f"Title of Incident=>{row.get('short_description', '')}\n"
                f"Description of Incident=>{row.get('description', '')}\n"
                f"Comments on general findings about the incident such as the person who found the incident, steps to replicate the issue, and potential solutions=>{row.get('work_notes', '')}"
            )
        documents.append(
            Document(
                text = text_input,
                metadata = {
                    "incident_number": row.get("incident_number", ""),
                    "service_impacted": row.get("service_impacted", ""),
                    "status": row.get("status", ""),
                    "assignment_group": row.get("assignment_group", ""),
                },
                metadata_template="{key}=>{value}",
                metadata_separator="::",
                text_template = "<INCIDENT>\n<METADATA>{metadata_str}</METADATA>\n\n<CONTENT>{content}</CONTENT>\n</INCIDENT>"
            )
        )

    # Store in Vectorstore
    vector_store_incidents = pg_db.get_vector_store(Config.INCIDENT_EMBEDDING_TABLE_NAME, new=True)
    storage_context_incidents = StorageContext.from_defaults(vector_store=vector_store_incidents)
    index_incidents = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context_incidents,
        embed_model=embedding_model,
        transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=50)],
        show_progress=False
    )
    return True

##################################
# Agent
##################################
def get_agent(only_class=False):
    agent_class = IncidentAgent(Config, Custom_Prompt)
    if only_class:
        return agent_class 
    return agent_class.get_agent()

