from graphrag import GraphBuilder, Schema

# Define your schema
schema = Schema(
    entities=["Person", "Organization", "Location"],
    relations=["WORKS_FOR", "LOCATED_IN"]
)

# Create a graph builder
builder = GraphBuilder(uri="bolt://localhost:7687", user="neo4j", password="password")

# Build the graph from your data
builder.build_graph(data="Your unstructured text data", schema=schema)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text):
    return model.encode(text)

