s
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import Table, MetaData, text
import tiktoken
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler, LlamaDebugHandler
from llama_index.core.schema import MetadataMode
from llama_index.core.agent.workflow import  ReActAgent
from llama_index.core.workflow import  Context
from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import (
    VectorStoreIndex,
    SQLDatabase,
    get_response_synthesizer,
    PromptTemplate
)
from llama_index.core.indices.struct_store.sql_query import (
    NLSQLTableQueryEngine
)

from llama_index.core.response_synthesizers import  BaseSynthesizer
from llama_index.core.retrievers import BaseRetriever, VectorIndexAutoRetriever
from llama_index.core.query_engine import  CustomQueryEngine

from Config import Config, Custom_Prompt
import logging
import sys,os
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# # logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



class Model():
    def __init__(self):
        self.embedding_model = None
        self.llm = None

        self.callback_manager = None
        self.token_counter = None
    def _get_callback_manager(self, model="gpt-4o", new=False):
        if self.get_callback_manager is None or new:
            # Token tracking
            self.token_counter = TokenCountingHandler(tokenizer=tiktoken.encoding_for_model(model).encode)
            self.llama_debug = LlamaDebugHandler(print_trace_on_end=True)
            self.callback_manager = CallbackManager([self.token_counter, self.llama_debug])
        return self.callback_manager
    
    def get_token_info(self, reset=False, model="gpt-4o"):
        if self.token_counter is None:
            self._get_callback_manager(model)
        token_counter = self.token_counter
        print(
            "Embedding Token: ",
            token_counter.total_embedding_token_count,
            "\n",
            "LLM Prompt Token: ",
            token_counter.prompt_llm_token_count,
            "\n",
            "LLM Completion Token: ",
            token_counter.completion_llm_token_count,
            "\n",
            "Total LLM Token Count: ",
            token_counter.total_llm_token_count,
            "\n"
        )
        if reset:
            token_counter.reset_counts()
    def get_embedding_model(self, AZURE_OPENAI_EMBEDDING_MODEL, AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME, AZURE_OPENAI_EMBEDDING_KEY, AZURE_OPENAI_EMBEDDING_ENDPOINT, AZURE_OPENAI_EMBEDDING_API_VERSION, new=False, enable_callback=False):
        if self.embedding_model is None or new:
            callback_manager = self._get_callback_manager(AZURE_OPENAI_EMBEDDING_MODEL, new=True) if enable_callback else None
            self.embedding_model = AzureOpenAIEmbedding(
                model=AZURE_OPENAI_EMBEDDING_MODEL,
                deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                api_key=AZURE_OPENAI_EMBEDDING_KEY,
                azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
                callback_manager=callback_manager,
                api_version=AZURE_OPENAI_EMBEDDING_API_VERSION,
            )
        return self.embedding_model

    def get_llm_model(self, AZURE_OPENAI_LLM_MODEL,AZURE_OPENAI_LLM_DEPLOYMENT_NAME,AZURE_OPENAI_LLM_KEY,AZURE_OPENAI_LLM_ENDPOINT,AZURE_OPENAI_LLM_API_VERSION,temperature = 0.1, new=False, enable_callback=False):
        if self.llm is None or new:
            callback_manager = self._get_callback_manager(AZURE_OPENAI_LLM_MODEL, new=True) if enable_callback else None
            self.llm = AzureOpenAI(
                model=AZURE_OPENAI_LLM_MODEL,
                deployment_name=AZURE_OPENAI_LLM_DEPLOYMENT_NAME,
                api_key=AZURE_OPENAI_LLM_KEY,
                azure_endpoint=AZURE_OPENAI_LLM_ENDPOINT,
                callback_manager=callback_manager,
                api_version=AZURE_OPENAI_LLM_API_VERSION,
                temperature = temperature
            )
        return self.llm


# DATABASE/VECTOR_STORE CONFIGURATION
class PG_Database():
    def __init__(self, host, port, user, password, dbname, schema:str=None, embedding_dim=3072):
        self.host=host
        self.port=port
        self.user=user
        self.password=password
        self.dbname=dbname
        self.conn = None
        self.conn_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.schema = schema if not schema else self.dbname

        self.embedding_dim = embedding_dim
        self.vector_store = {}
    def get_conn(self, new=False):
        if self.conn is None or new:
            self.conn = create_engine(self.conn_str)
            self._check_conn()
        return self.conn
    def _check_conn(self):
        try:
            self.conn.connect()
            logging.info(f"Successful connection to {self.dbname}!")
        except SQLAlchemyError as err:
            logging.info("Error connecting to DB", err.__cause__)
    def get_vector_store(self, table_name:str, embedding_dim:int=None, new=False):
        if embedding_dim:
            self.embedding_dim = embedding_dim
        if table_name not in self.vector_store or new:
            self.vector_store[table_name] = PGVectorStore.from_params(
                database=self.dbname,
                host=self.host,
                password=self.password,
                port=self.port,
                user=self.user,
                table_name=table_name,
                schema_name=self.schema,
                embed_dim=self.embedding_dim,
                hybrid_search=True,
                text_search_config="english",
                # use_halfvec=True,
                # {"ivfflat_probes": 10}
                # hnsw_kwargs={
                #     "hnsw_m": 16,
                #     "hnsw_ef_construction": 64,
                #     "hnsw_ef_search": 40,
                #     "hnsw_dist_method": "vector_cosine_ops",
                # }
            )
        return self.vector_store[table_name]
    def upsert_data(self, table_name:str, rows:list, primary_keys:list=None, additional_update_kwangs:dict={}, remove_update_keys:list=[]):
        engine = self.get_conn()
        with engine.connect() as connection:
            try:
                metadata = MetaData(self.schema)
                metadata.reflect(bind=engine)
                target_table = Table(table_name, metadata, autoload_with=engine)
                target_columns = [x.name for x in target_table.columns]
                
                for row in rows:
                    row = {key:row[key] for key in row if key in target_columns}
                    stmt = insert(target_table).values(row)
                    if not primary_keys:
                        primary_keys = [key.name for key in inspect(target_table).primary_key]

                    updated_dict = {c.name: c for c in stmt.excluded if not c.primary_key}
                    updated_dict.update(additional_update_kwangs)
                    for key in remove_update_keys:
                        updated_dict.pop(key, None)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=primary_keys,
                        set_= updated_dict
                    )
                    connection.execute(stmt)
                connection.commit()
                logging.info(f"Upsert completed ({table_name}): {len(rows)}")
            except Exception as e:
                connection.rollback()
                logging.info(f"Error during transaction:", e)
    def _create_simple_query(self, selected_table, **params):
        where_clauses = []
        updated_params = {}
        for key, value in params.items():
            where_clauses.append(f"{key} = :{key}")
            params[key] = value
        
        where_clause = " AND ".join(where_clauses)
        return text(f"SELECT * FROM {self.schema}.{selected_table} WHERE {where_clause}"), updated_params
    def read_table(self, table_name, schema=None):
        if not schema:
            schema = self.schema
        df = pd.read_sql_table(table_name=table_name, schema=schema, con=self.get_conn())
        return df
    def read_query(self, selected_table, **params):
        updated_q, params = self._create_simple_query(selected_table, **params)
        df = pd.read_sql_query(sql=updated_q, con=self.get_conn(), params=params)
        return df
    
##################################
# Agent Related
##################################
class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""
    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: AzureOpenAI
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        context_str = "\n\n".join([n.node.get_content(MetadataMode.LLM) for n in nodes])
        print(query_str)
        response = self.llm.complete(
            self.qa_prompt.format(context_str=context_str, query_str=query_str)
        )
        return str(response)
    
class IncidentAgent():
    def __init__(self, config=Config, custom_prompt = Custom_Prompt):
        self.config = config
        self.custom_prompt = custom_prompt
        self.pg_db = None
        self.index = None
        self.model = Model()
        self.embedding_model = None
        self.llm_model = None
        self.semantic_engine = None
        self.text_to_sql_engine = None
        self.agent=None
        self.ctx=None
        self.vector_store_info = VectorStoreInfo(
                                content_info="Incident Issues",
                                    metadata_info=[
                                        MetadataInfo(
                                            name="status",
                                            description="Whether the issue is `Assigned`, `Resolved`, `In Progress`",
                                            type="string",
                                        ),
                                        MetadataInfo(
                                            name="incident_number",
                                            description="Incident Number, indentifier or ticket.",
                                            type="string",
                                        ),
                                    ],
                                )
    def _get_pgvector(self, new=False):
        if self.pg_db is None or new:
            self.pg_db = PG_Database(self.config.DB_HOSTNAME,
                                    self.config.DB_PORT,
                                    self.config.DB_USERNAME,
                                    self.config.DB_PASSWORD,
                                    self.config.DB_NAME,
                                    self.config.AZURE_OPENAI_EMBEDDING_DIMENSIONS)
        return self.pg_db
    def _get_embedding_model(self, new=False):
        if self.embedding_model is None or new:
            self.embedding_model = self.model.get_embedding_model(
                self.config.AZURE_OPENAI_EMBEDDING_MODEL,
                self.config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                self.config.AZURE_OPENAI_EMBEDDING_KEY,
                self.config.AZURE_OPENAI_EMBEDDING_ENDPOINT,
                self.config.AZURE_OPENAI_EMBEDDING_API_VERSION,
                new=new
            )
        return self.embedding_model
    def _get_index(self, new=False):
        if self.index is None or new:
            pg_db = self._get_pgvector(new)
            embedding_model = self._get_embedding_model(new)
            self.index = VectorStoreIndex.from_vector_store(vector_store=pg_db.get_vector_store(self.config.INCIDENT_EMBEDDING_TABLE_NAME),embed_model=embedding_model)
        return self.index
    def _get_llm_model(self, new=False):
        if self.llm_model is None or new:
            self.llm_model = self.model.get_llm_model(
                self.config.AZURE_OPENAI_LLM_MODEL,
                self.config.AZURE_OPENAI_LLM_DEPLOYMENT_NAME,
                self.config.AZURE_OPENAI_LLM_KEY,
                self.config.AZURE_OPENAI_LLM_ENDPOINT,
                self.config.AZURE_OPENAI_LLM_API_VERSION,
                new=new
            )
        return self.llm_model
    def get_semantic_engine(self, new=False):
        if self.semantic_engine is None or new:
            vector_store_info = self.vector_store_info
            index = self._get_index(new)
            llm = self._get_llm_model(new)
            qa_prompt= self.custom_prompt.SEMANTIC_QA_PROMPT
            retriever = VectorIndexAutoRetriever(
                index,
                vector_store_info=vector_store_info,
                similarity_top_k=3,
                empty_query_top_k=10,  # if only metadata filters are specified, this is the limit
                verbose=True,
                vector_store_query_mode="hybrid", 
                vector_store_kwargs={"ivfflat_probes": 10},
                llm=llm
            )
            synthesizer = get_response_synthesizer(response_mode="compact", llm=llm)
            self.semantic_query_engine = RAGStringQueryEngine(llm=llm, retriever=retriever, response_synthesizer=synthesizer, qa_prompt=qa_prompt)
        return self.semantic_query_engine
    def get_text_to_sql_engine(self, new=False):
        if self.text_to_sql_engine is None or new:
            pg_db = self._get_pgvector(new)
            llm = self._get_llm_model(new)
            embed_model = self._get_embedding_model(new)
            INCIDENT_TABLE_NAME = self.config.INCIDENT_TABLE_NAME
            APP_LINKAGE_TABLE_NAME = self.config.APP_LINKAGE_TABLE_NAME
            HEALTH_STATUS_TABLE_NAME = self.config.HEALTH_STATUS_TABLE_NAME

            text_to_sql_prompt = self.custom_prompt._create_text_to_sql_prompt(INCIDENT_TABLE_NAME, APP_LINKAGE_TABLE_NAME, HEALTH_STATUS_TABLE_NAME)
            sql_database = SQLDatabase(pg_db.get_conn(), include_tables=[INCIDENT_TABLE_NAME, APP_LINKAGE_TABLE_NAME, HEALTH_STATUS_TABLE_NAME], schema=pg_db.schema,)
            self.text_to_sql_engine = NLSQLTableQueryEngine(
                            sql_database=sql_database,
                            tables=[INCIDENT_TABLE_NAME, APP_LINKAGE_TABLE_NAME, HEALTH_STATUS_TABLE_NAME], # Specify the table context for the engine
                            llm=llm,
                            embed_model=embed_model,
                            synthesize_response=False,
                            text_to_sql_prompt = PromptTemplate(text_to_sql_prompt)
                        )
        return self.text_to_sql_engine
    def get_agent(self, new=False):
        if self.agent is None or new:
            semantic_query_engine = self.get_semantic_engine(new)
            text_to_sql_engine = self.get_text_to_sql_engine(new)
            llm= self._get_llm_model(new)
            prompt_template_agent = self.custom_prompt.REACT_AGENT_PROMPT
            INCIDENT_TABLE_NAME = self.config.INCIDENT_TABLE_NAME
            APP_LINKAGE_TABLE_NAME = self.config.APP_LINKAGE_TABLE_NAME
            HEALTH_STATUS_TABLE_NAME = self.config.HEALTH_STATUS_TABLE_NAME

            semantic_search_tool = QueryEngineTool.from_defaults(
                query_engine=semantic_query_engine,
                name = "semantic_query_engine",
                description= "Use this tool for semantic search related task, such as incident descriptio, problem, possible solutions, and work notes."
            )
            sql_search_tool = QueryEngineTool.from_defaults(
                query_engine=text_to_sql_engine,
                name = "text_to_sql_engine",
                description= f"Use this tool to query {INCIDENT_TABLE_NAME}, {APP_LINKAGE_TABLE_NAME}, and {HEALTH_STATUS_TABLE_NAME} using sql. Use this for any quantitative tasks, tasks that expect definite data from the tables, tasks that require looking for upstream and downstream services (use {APP_LINKAGE_TABLE_NAME}), and the health status of the application (use {HEALTH_STATUS_TABLE_NAME})." 
            )
            f_agent = ReActAgent(
                tools = [semantic_search_tool, sql_search_tool],
                llm = llm)
            f_agent.update_prompts({"react_header": PromptTemplate(prompt_template_agent)})
            ctx = Context(f_agent)
            self.ctx = ctx
            self.agent = f_agent
        return self.agent
    def _sample_run(self, query):
        agent = self.get_agent()
        handler = agent.run(query, ctx=self.ctx)
        return handler
##################################
# Helper function
##################################
def etl(df: pd.DataFrame, column_mapping, reset_index=False):
    df = df.replace({np.nan: None})
    # Rename the DataFrame columns
    df = df.rename(columns=column_mapping)
    now = datetime.now()
    df['updated_on'] = now
    df['created_on'] = now
    if reset_index:
        df["indexed"] = False
    return df.to_dict("records")
