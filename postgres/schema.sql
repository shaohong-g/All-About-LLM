-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create sample table
CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    item_data JSONB,
    content TEXT,
    embedding vector(1536), -- vector data (openai embedding model)
    embedding384 vector(384) -- vector data (sentence transformer)
);

-- Create index (Use vector_ip_ops for inner product and vector_cosine_ops for cosine distance)
CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON items USING ivfflat (embedding384 vector_cosine_ops) WITH (lists = 100);
