# All-About-LLM

0. Embeddings - [here](./embedding.ipynb)
1. Retrieval-Augmented Generation (RAG)
2. LLMs: Gemini (OpenAi - not free)
3. AI agents, LangChain, LangGraph
4. LLM with Knowledge graph
5. The Model Context Protocol (MCP)

## Environment
```sh
# Create python virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate.bat

# Activate virtual environment (Mac)
source .venv/bin/activate
```

## Pre-requisite
```sh
pip install -r requirements.txt
```

## Quick Start
1. PgVector
    - Initialize docker containers
    ```sh
    # Start Docker Service (detached)
    docker compose -f pgvector.yml up -d
    # End/Stop Docker Service
    docker compose -f pgvector.yml down -v
    docker compose -f pgvector.yml stop

    # Access container
    docker exec -it <container name> bash
    psql -U <POSTGRES_USER> <POSTGRES_DB>
    ```
    - Add server in PgAdmin (localhost:PORT)
        1. Under `Object` tab -> `Register` -> `Server`
        2. Fill in the followings:
            - `General` -> `Name` : <any name that you want>
            - `Connection` -> `Host name/address` : <db container name - pgvector-db>
            - `Connection` -> `Port` & `Username` & `Password`



## Tools
- [Tiktokenizer](https://tiktokenizer.vercel.app/)

# Tutorials
- [Introduction to Text Embeddings](https://www.datacamp.com/tutorial/introduction-to-text-embeddings-with-the-open-ai-api)

## Resources
- [How I use LLMs - Andrej Karpathy](https://www.youtube.com/watch?v=EWvNQjAaOHw)
- [RAG vs MCP makes sense? is RAG dead?](https://medium.com/@gejing/rag-vs-mcp-makes-sense-is-rag-dead-134856664cd6)
- [Vector Databases](https://medium.com/@soumitsr/a-broke-b-chs-guide-to-tech-start-up-choosing-vector-database-part-1-local-self-hosted-4ebe4eec3045)
- [Openai cheatsheet](https://www.datacamp.com/cheat-sheet/the-open-ai-api-in-python)
- [Openai - No longer free tier](https://community.openai.com/t/usage-tier-free-to-tier-1/919150)

## Embeddings
- [Sentence Transformer](https://www.sbert.net/examples/applications/computing-embeddings/README.html)
- [Tiktoken](https://github.com/openai/tiktoken/tree/main?tab=readme-ov-file)
- [Openai Embedding Models](https://platform.openai.com/docs/guides/embeddings#embedding-models)
- [Gemini Embedding](https://ai.google.dev/gemini-api/docs/embeddings)
- [Reduce document embedding dimension](https://stackoverflow.com/questions/53883945/how-to-reduce-the-dimension-of-the-document-embedding)

## RAG
- [Pgvector tutorial](https://www.datacamp.com/tutorial/pgvector-tutorial)
    - IVFFlat (Inverted File Flat) 
    ```sql
    CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    /*
    CREATE INDEX ON items: Starts the creation of an index on the items table.
    USING ivfflat: Specifies the use of the ivfflat indexing method, suitable for approximate nearest neighbor searches.
    (embedding vector_cosine_ops): Targets the embedding column and uses vector_cosine_ops for cosine similarity operations.
    WITH (lists = 100): Configures the index with 100 partition lists, balancing search speed and memory usage.
    */
    ```
    - HNSW (Hierarchical Navigable Small World) index
    ```sql
    CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64);
    /*
    CREATE INDEX ON items: Creates an index on the items table.
    USING hnsw: Specifies the use of the HNSW (Hierarchical Navigable Small World) algorithm for the index.
    (embedding vector_l2_ops): Targets the embedding column and uses vector_l2_ops for distance calculations.
    WITH (m = 16, ef_construction = 64): Sets parameters for the HNSW algorithm; m is the number of bi-directional links, ef_construction is the size of the dynamic list during index construction.
    */
    ```
- [Vector with different dimension](https://github.com/pgvector/pgvector/issues/426)
- [Faiss Tutorial](https://www.pinecone.io/learn/series/faiss/faiss-tutorial/)

https://github.com/google/generative-ai-docs/blob/main/site/en/gemini-api/tutorials/document_search.ipynb