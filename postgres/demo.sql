-- Calculate Euclidean distance
SELECT * FROM items ORDER BY embedding <-> '[2,3,4]' LIMIT 1;

-- Calculate cosine distance
SELECT * FROM items ORDER BY embedding <=> '[2,3,4]' LIMIT 1;

-- Sieve records of diff dimension
SELECT * FROM items WHERE vector_dims(embedding) = 3 ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
