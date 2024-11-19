from langchain.graphs import Neo4jGraph
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Neo4jVector
from langchain.llms import OpenAI
from langchain.chains import GraphRAGChain

# Connect to Neo4j
graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")

# Set up embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store
vector_store = Neo4jVector.from_existing_graph(
    graph=graph,
    embedding=embeddings,
    node_properties=["name", "description"],
    relationship_properties=["type"]
)

# Set up LLM
llm = OpenAI(temperature=0)

# Create Graph RAG chain
graph_rag_chain = GraphRAGChain.from_llm(
    llm=llm,
    graph=graph,
    vector_store=vector_store,
    return_intermediate_steps=True
)

# Use the chain
query = "What organizations are located in San Francisco?"
result = graph_rag_chain({"query": query})
print(result["answer"])