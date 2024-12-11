# Example of combining GeoAI with GraphRAG in Neo4j
def create_geo_embeddings(location_data):
    # Convert geographic features to vector embeddings
    embeddings = spatial_encoder.encode(location_data)
    return embeddings

def store_in_neo4j(embeddings, location_metadata):
    # Create nodes with vector properties
    query = """
    CREATE (l:Location {
        coords: $coords,
        embedding: $embedding,
        metadata: $metadata
    })
    """
    graph.run(query, {
        'coords': location_metadata['coordinates'],
        'embedding': embeddings,
        'metadata': location_metadata
    })

def geo_aware_query(query_location, radius):
    # Combine vector similarity with spatial queries
    return """
    MATCH (l:Location)
    WHERE spatial.distance(l.coords, $query_coords) < $radius
    WITH l, vector.similarity(l.embedding, $query_embedding) as similarity
    RETURN l
    ORDER BY similarity DESC
    LIMIT 10
    """