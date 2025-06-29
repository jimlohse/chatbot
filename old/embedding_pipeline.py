#!/usr/bin/env python3
"""
Embedding Pipeline for Virginia & Truckee Railroad RAG System
Uses local sentence-transformers for embeddings and ChromaDB for storage
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
import os
from datetime import datetime

class VTRailroadEmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./chroma_db"):
        """
        Initialize the embedding pipeline

        Args:
            model_name: Hugging Face model name (all-MiniLM-L6-v2 is good balance of speed/quality)
            persist_directory: Where to store the ChromaDB database
        """
        self.model_name = model_name
        self.persist_directory = persist_directory

        # Initialize sentence transformer model
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        print(f"Model loaded. Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = "vt_railroad_content"

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Using existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Virginia & Truckee Railroad website content"}
            )
            print(f"Created new collection: {self.collection_name}")

    def create_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Create embeddings for a batch of texts"""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch, convert_to_tensor=False)
            embeddings.extend(batch_embeddings.tolist())
            print(f"Processed embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

        return embeddings

    def load_scraped_content(self, filename: str = "vt_railroad_content.json") -> List[Dict]:
        """Load the scraped content from JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = json.load(f)
            print(f"Loaded {len(content)} content chunks from {filename}")
            return content
        except FileNotFoundError:
            print(f"ERROR: Could not find {filename}. Make sure you've run the scraper first.")
            return []
        except json.JSONDecodeError:
            print(f"ERROR: Invalid JSON in {filename}")
            return []

    def process_and_store_content(self, content_file: str = "vt_railroad_content.json"):
        """Process scraped content and store embeddings in ChromaDB"""
        # Load scraped content
        content_chunks = self.load_scraped_content(content_file)
        if not content_chunks:
            return False

        # Prepare data for embedding
        texts = []
        metadatas = []
        ids = []

        for chunk in content_chunks:
            # Combine title and content for better context
            text_to_embed = f"{chunk['title']}\n\n{chunk['content']}"
            texts.append(text_to_embed)

            # Prepare metadata
            metadata = {
                "url": chunk["url"],
                "title": chunk["title"],
                "page_path": chunk["page_path"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "scraped_at": chunk["scraped_at"],
                "embedded_at": datetime.now().isoformat()
            }
            metadatas.append(metadata)
            ids.append(chunk["id"])

        print(f"Creating embeddings for {len(texts)} content chunks...")
        embeddings = self.create_embeddings_batch(texts)

        # Store in ChromaDB
        print("Storing embeddings in ChromaDB...")
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"SUCCESS: Stored {len(embeddings)} embeddings in database")
            return True
        except Exception as e:
            print(f"ERROR storing embeddings: {e}")
            return False

    def query_similar_content(self, query: str, n_results: int = 3) -> Dict:
        """
        Find similar content based on query

        Args:
            query: User's question or search term
            n_results: Number of similar chunks to return

        Returns:
            Dictionary with similar content and metadata
        """
        # Create embedding for the query
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)[0].tolist()

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = {
            "query": query,
            "results": []
        }

        for i in range(len(results["documents"][0])):
            result = {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity_score": 1 - results["distances"][0][i],  # Convert distance to similarity
                "url": results["metadatas"][0][i]["url"],
                "title": results["metadatas"][0][i]["title"]
            }
            formatted_results["results"].append(result)

        return formatted_results

    def get_database_stats(self) -> Dict:
        """Get statistics about the database"""
        count = self.collection.count()

        # Get a sample to see what's in there
        sample = self.collection.peek(limit=5)

        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension(),
            "sample_titles": [meta["title"] for meta in sample["metadatas"]] if sample["metadatas"] else []
        }

    def rebuild_database(self, content_file: str = "vt_railroad_content.json"):
        """Clear and rebuild the entire database"""
        print("Rebuilding database...")

        # Delete existing collection
        try:
            self.client.delete_collection(name=self.collection_name)
            print("Deleted existing collection")
        except:
            pass

        # Create new collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Virginia & Truckee Railroad website content"}
        )

        # Process and store content
        return self.process_and_store_content(content_file)

def main():
    """Test the embedding pipeline"""
    print("Virginia & Truckee Railroad Embedding Pipeline")
    print("=" * 60)

    # Initialize pipeline
    pipeline = VTRailroadEmbeddingPipeline()

    # Process scraped content
    print("\nProcessing scraped content...")
    success = pipeline.process_and_store_content()

    if success:
        # Show database stats
        stats = pipeline.get_database_stats()
        print(f"\nDATABASE STATISTICS:")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Model: {stats['model_name']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print(f"Sample titles: {stats['sample_titles'][:3]}")

        # Test queries
        test_queries = [
            "What time do trains run?",
            "How much do tickets cost?",
            "Where is the depot located?",
            "What can I see on the train ride?"
        ]

        print(f"\nTEST QUERIES:")
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = pipeline.query_similar_content(query, n_results=2)
            for i, result in enumerate(results["results"]):
                print(f"  Result {i+1} (similarity: {result['similarity_score']:.3f})")
                print(f"    Title: {result['title']}")
                print(f"    Content: {result['content'][:100]}...")

        print(f"\nSUCCESS: Embedding pipeline is ready!")
        print(f"Next step: Set up the RAG query system with Claude API")

    else:
        print("FAILED: Could not process content. Check that vt_railroad_content.json exists.")

if __name__ == "__main__":
    main()