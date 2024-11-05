import boto3
import numpy as np
from neo4j import GraphDatabase
from PIL import Image
import torch
from torchvision import models, transforms
from typing import List, Dict, Tuple
import logging

class GraphVectorOrchestrator:
    def __init__(self, 
                 neo4j_uri: str,
                 neo4j_user: str,
                 neo4j_password: str,
                 neptune_endpoint: str,
                 region_name: str):
        """
        Initialize connections to Neo4j and Neptune ML
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize Neo4j connection
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri, 
            auth=(neo4j_user, neo4j_password)
        )
        
        # Initialize Neptune ML connection
        self.neptune = boto3.client('neptune',
            endpoint_url=neptune_endpoint,
            region_name=region_name
        )
        
        # Initialize vision model (using ResNet as example)
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def generate_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Generate vector embedding for an image using pretrained model
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model.forward(image_tensor)
                
            return embedding.numpy()
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            raise

    def store_in_neo4j(self, 
                       image_id: str,
                       metadata: Dict,
                       relationships: List[Tuple[str, str, str]]):
        """
        Store image metadata and relationships in Neo4j
        """
        def create_node(tx, image_id, metadata):
            query = """
            CREATE (i:Image {id: $image_id})
            SET i += $metadata
            RETURN i
            """
            tx.run(query, image_id=image_id, metadata=metadata)

        def create_relationships(tx, relationships):
            for source, relation, target in relationships:
                query = """
                MATCH (a), (b)
                WHERE a.id = $source AND b.id = $target
                CREATE (a)-[r:$relation]->(b)
                """
                tx.run(query, source=source, relation=relation, target=target)

        with self.neo4j_driver.session() as session:
            session.write_transaction(create_node, image_id, metadata)
            session.write_transaction(create_relationships, relationships)

    def store_in_neptune(self, 
                        image_id: str,
                        embedding: np.ndarray):
        """
        Store vector embedding in Neptune ML
        """
        try:
            # Convert embedding to list for JSON serialization
            embedding_list = embedding.tolist()
            
            # Store embedding using Neptune ML API
            response = self.neptune.create_vector_embedding(
                vectorId=image_id,
                values=embedding_list,
                dimension=len(embedding_list)
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error storing in Neptune: {str(e)}")
            raise

    def query_similar_images(self, 
                           query_embedding: np.ndarray,
                           k: int = 5) -> List[Dict]:
        """
        Find similar images using Neptune ML vector similarity search
        """
        try:
            response = self.neptune.query_vector_embeddings(
                values=query_embedding.tolist(),
                k=k
            )
            
            # Get metadata from Neo4j for matched images
            similar_images = []
            
            with self.neo4j_driver.session() as session:
                for match in response['matches']:
                    query = """
                    MATCH (i:Image {id: $image_id})
                    RETURN i
                    """
                    result = session.run(query, image_id=match['vectorId'])
                    image_data = result.single()
                    
                    if image_data:
                        similar_images.append({
                            'image_id': match['vectorId'],
                            'similarity': match['score'],
                            'metadata': image_data['i']
                        })
            
            return similar_images
            
        except Exception as e:
            self.logger.error(f"Error querying similar images: {str(e)}")
            raise

    def close(self):
        """
        Close connections
        """
        self.neo4j_driver.close()

# Example usage
if __name__ == "__main__":
    orchestrator = GraphVectorOrchestrator(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        neptune_endpoint="https://your-neptune-endpoint",
        region_name="us-east-1"
    )
    
    # Process new image
    image_path = "example.jpg"
    image_id = "img_001"
    
    # Generate embedding
    embedding = orchestrator.generate_image_embedding(image_path)
    
    # Store metadata in Neo4j
    metadata = {
        "filename": "example.jpg",
        "timestamp": "2024-11-02",
        "category": "landscape"
    }
    relationships = [
        ("img_001", "SIMILAR_TO", "img_002"),
        ("img_001", "CONTAINS", "mountain")
    ]
    orchestrator.store_in_neo4j(image_id, metadata, relationships)
    
    # Store embedding in Neptune ML
    orchestrator.store_in_neptune(image_id, embedding)
    
    # Query similar images
    similar = orchestrator.query_similar_images(embedding, k=5)
    
    orchestrator.close()