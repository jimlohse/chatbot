#!/usr/bin/env python3
"""
Hybrid RAG System for Virginia & Truckee Railroad
1. First checks local knowledge base (fast, free)
2. Falls back to Ollama LLM if no local answer found (slower, still free)
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import requests
import asyncio
import aiohttp
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

class LocalKnowledgeBase:
    """Handles local file-based knowledge before calling Ollama"""

    def __init__(self, knowledge_files_path: str = "./knowledge"):
        self.knowledge_path = knowledge_files_path
        self.knowledge_base = {}
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize ChromaDB for local knowledge
        self.client = chromadb.PersistentClient(path="./local_chroma_db")
        try:
            self.collection = self.client.get_collection("local_knowledge")
        except:
            self.collection = self.client.create_collection("local_knowledge")

        self.load_local_knowledge()

    def load_local_knowledge(self):
        """Load local knowledge files (FAQ, schedules, etc.)"""
        os.makedirs(self.knowledge_path, exist_ok=True)

        # Create default knowledge files if they don't exist
        self.create_default_knowledge_files()

        # Load all knowledge files
        for filename in os.listdir(self.knowledge_path):
            if filename.endswith('.json'):
                filepath = os.path.join(self.knowledge_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.knowledge_base[filename] = data
                        print(f"Loaded local knowledge: {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        # Index knowledge in ChromaDB
        self.index_local_knowledge()

    def create_default_knowledge_files(self):
        """Create default knowledge files for Virginia & Truckee Railroad"""

        # FAQ file
        faq_file = os.path.join(self.knowledge_path, "faq.json")
        if not os.path.exists(faq_file):
            faq_data = {
                "faq": [
                    {
                        "question": "What time do trains run?",
                        "answer": "Trains run daily with multiple departures. Please check our schedule page for current times or call (775) 847-0380 for today's schedule.",
                        "keywords": ["time", "schedule", "when", "hours", "departure"]
                    },
                    {
                        "question": "How much do tickets cost?",
                        "answer": "Adult tickets are $29, Children (4-12) are $19, and Children 3 and under ride free. Special event trains may have different pricing.",
                        "keywords": ["cost", "price", "ticket", "fare", "money", "adult", "child"]
                    },
                    {
                        "question": "Where is the depot located?",
                        "answer": "The historic Virginia City depot is located at 165 F Street, Virginia City, Nevada, just below Main Street.",
                        "keywords": ["location", "address", "depot", "station", "where", "directions"]
                    },
                    {
                        "question": "Can I see wild horses?",
                        "answer": "Yes! You may see wild mustangs running alongside the tracks. These horses have roamed the area for over 400 years.",
                        "keywords": ["horses", "mustang", "wild", "animals", "see"]
                    },
                    {
                        "question": "How long is the train ride?",
                        "answer": "The scenic train ride to Gold Hill is approximately 35 minutes round trip.",
                        "keywords": ["long", "duration", "time", "minutes", "trip", "ride"]
                    },
                    {
                        "question": "Do you have parking?",
                        "answer": "Yes, free parking is available near the depot in Virginia City.",
                        "keywords": ["parking", "park", "car", "vehicle"]
                    },
                    {
                        "question": "What is the phone number?",
                        "answer": "You can call us at (775) 847-0380 for reservations and information.",
                        "keywords": ["phone", "number", "call", "contact", "telephone"]
                    }
                ]
            }

            with open(faq_file, 'w', encoding='utf-8') as f:
                json.dump(faq_data, f, indent=2)

        # Contact info file
        contact_file = os.path.join(self.knowledge_path, "contact.json")
        if not os.path.exists(contact_file):
            contact_data = {
                "contact": {
                    "phone": "(775) 847-0380",
                    "address": "165 F Street, Virginia City, Nevada",
                    "website": "virginiatruckee.com",
                    "hours": "Daily during operating season",
                    "email": "Contact through website or phone"
                }
            }

            with open(contact_file, 'w', encoding='utf-8') as f:
                json.dump(contact_data, f, indent=2)

    def index_local_knowledge(self):
        """Index local knowledge in ChromaDB for semantic search"""
        documents = []
        metadatas = []
        ids = []

        for filename, data in self.knowledge_base.items():
            if 'faq' in data:
                for i, item in enumerate(data['faq']):
                    doc_text = f"Q: {item['question']} A: {item['answer']}"
                    documents.append(doc_text)
                    metadatas.append({
                        'source': filename,
                        'type': 'faq',
                        'question': item['question'],
                        'keywords': ','.join(item.get('keywords', []))
                    })
                    ids.append(f"{filename}_faq_{i}")

            elif 'contact' in data:
                contact_text = f"Contact information: {json.dumps(data['contact'], indent=2)}"
                documents.append(contact_text)
                metadatas.append({
                    'source': filename,
                    'type': 'contact'
                })
                ids.append(f"{filename}_contact")

        if documents:
            # Clear existing and add new
            try:
                self.collection.delete(where={})
            except:
                pass

            embeddings = self.embedding_model.encode(documents).tolist()

            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            print(f"Indexed {len(documents)} local knowledge items")

    def search_local_knowledge(self, query: str, threshold: float = 0.7) -> Optional[Dict]:
        """Search local knowledge base first"""
        try:
            # Simple keyword matching first (fastest)
            keyword_result = self.keyword_search(query)
            if keyword_result:
                return keyword_result

            # Semantic search using embeddings
            query_embedding = self.embedding_model.encode([query]).tolist()

            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=3,
                include=["documents", "metadatas", "distances"]
            )

            if results['documents'] and results['documents'][0]:
                # Check if the best match is good enough
                distance = results['distances'][0][0]
                similarity = 1 - distance

                if similarity >= threshold:
                    metadata = results['metadatas'][0][0]
                    document = results['documents'][0][0]

                    # Extract answer from FAQ format
                    if metadata.get('type') == 'faq':
                        # Document is in format "Q: question A: answer"
                        answer_match = re.search(r'A: (.+)', document)
                        if answer_match:
                            return {
                                'answer': answer_match.group(1),
                                'source': 'local_knowledge',
                                'confidence': similarity,
                                'type': metadata.get('type'),
                                'found_locally': True
                            }
                    else:
                        return {
                            'answer': document,
                            'source': 'local_knowledge',
                            'confidence': similarity,
                            'type': metadata.get('type'),
                            'found_locally': True
                        }

            return None

        except Exception as e:
            print(f"Error searching local knowledge: {e}")
            return None

    def keyword_search(self, query: str) -> Optional[Dict]:
        """Fast keyword-based search"""
        query_lower = query.lower()

        for filename, data in self.knowledge_base.items():
            if 'faq' in data:
                for item in data['faq']:
                    # Check if any keywords match
                    keywords = item.get('keywords', [])
                    if any(keyword in query_lower for keyword in keywords):
                        return {
                            'answer': item['answer'],
                            'source': 'local_knowledge_keyword',
                            'confidence': 0.9,
                            'type': 'faq',
                            'found_locally': True
                        }

                    # Check if question is similar
                    if any(word in query_lower for word in item['question'].lower().split() if len(word) > 3):
                        return {
                            'answer': item['answer'],
                            'source': 'local_knowledge_keyword',
                            'confidence': 0.8,
                            'type': 'faq',
                            'found_locally': True
                        }

        return None

class OllamaClient:
    """Client for communicating with local Ollama instance"""

    def __init__(self, ollama_host: str = "localhost", ollama_port: int = 11434, model: str = "llama2"):
        self.base_url = f"http://{ollama_host}:{ollama_port}"
        self.model = model
        self.timeout = 30  # seconds

        # Virginia & Truckee Railroad context for Ollama
        self.system_prompt = """You are a helpful assistant for the Virginia & Truckee Railroad, a historic tourist railroad in Nevada. You help visitors with information about train schedules, fares, routes, and general information about the railroad.

Key information about the Virginia & Truckee Railroad:
- Historic narrow-gauge railroad built in 1869 during the Comstock mining era
- Operates between Virginia City and Gold Hill, Nevada
- Also has a Carson City route (Sisters in History Route)
- 35-minute scenic round trip to Gold Hill
- Daily train service during operating season
- Phone: (775) 847-0380
- Location: 165 F Street, Virginia City, Nevada
- Adult tickets: $29, Children (4-12): $19, Children 3 and under: free
- May see wild mustangs during the ride
- Special event trains include Halloween, Christmas, and Day Out With Thomas

Guidelines:
- Be friendly and enthusiastic about the railroad
- Provide helpful information based on what you know
- If you don't have specific current information (like today's exact schedule), direct visitors to call (775) 847-0380 or visit virginiatruckee.com
- Always encourage visitors to visit and ride the train
- Keep responses concise but informative"""

    async def generate_response(self, user_question: str, context: str = "") -> Dict:
        """Generate response from Ollama"""
        try:
            prompt = f"{self.system_prompt}\n\nContext: {context}\n\nUser Question: {user_question}\n\nResponse:"

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 500,
                    "top_p": 0.9
                }
            }

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'answer': result.get('response', 'Sorry, I could not generate a response.'),
                            'source': 'ollama',
                            'model': self.model,
                            'found_locally': False
                        }
                    else:
                        return {
                            'answer': 'I apologize, but I\'m having trouble accessing information right now. Please call (775) 847-0380 or visit virginiatruckee.com for assistance.',
                            'source': 'error',
                            'error': f"HTTP {response.status}",
                            'found_locally': False
                        }

        except asyncio.TimeoutError:
            return {
                'answer': 'I apologize for the delay. Please call (775) 847-0380 or visit virginiatruckee.com for immediate assistance.',
                'source': 'timeout',
                'error': 'Request timeout',
                'found_locally': False
            }
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return {
                'answer': 'I apologize, but I\'m having trouble right now. Please call (775) 847-0380 or visit virginiatruckee.com for assistance.',
                'source': 'error',
                'error': str(e),
                'found_locally': False
            }

    def test_connection(self) -> bool:
        """Test if Ollama is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

class HybridVTRailroadRAG:
    """
    Hybrid RAG system that:
    1. Checks local knowledge first (free, fast)
    2. Falls back to Ollama if no local answer (free, slower)
    """

    def __init__(self,
                 ollama_host: str = None,
                 ollama_port: int = 11434,
                 ollama_model: str = "llama2",
                 knowledge_path: str = "./knowledge"):

        # Initialize local knowledge base
        self.local_kb = LocalKnowledgeBase(knowledge_path)

        # Initialize Ollama client
        ollama_host = ollama_host or os.getenv('OLLAMA_HOST', 'localhost')
        self.ollama = OllamaClient(ollama_host, ollama_port, ollama_model)

        # Test Ollama connection
        if self.ollama.test_connection():
            print(f"✓ Connected to Ollama at {ollama_host}:{ollama_port}")
        else:
            print(f"⚠ Warning: Cannot connect to Ollama at {ollama_host}:{ollama_port}")
            print("  Local knowledge will still work, but no LLM fallback available")

    async def query(self, user_question: str) -> Dict:
        """
        Process user question with hybrid approach:
        1. Search local knowledge first
        2. Fall back to Ollama if needed
        """
        start_time = datetime.now()

        # Step 1: Search local knowledge base
        print("Searching local knowledge base...")
        local_result = self.local_kb.search_local_knowledge(user_question)

        if local_result and local_result.get('confidence', 0) > 0.6:
            # Good local answer found
            print(f"✓ Found local answer (confidence: {local_result.get('confidence', 0):.2f})")
            return {
                'question': user_question,
                'answer': local_result['answer'],
                'source': local_result['source'],
                'method': 'local_knowledge',
                'confidence': local_result.get('confidence'),
                'response_time': (datetime.now() - start_time).total_seconds(),
                'cost_estimate': 0.0  # Local knowledge is free
            }

        # Step 2: Fall back to Ollama
        print("Local knowledge insufficient, querying Ollama...")

        # Provide context from local search if available
        context = ""
        if local_result:
            context = f"Related local information: {local_result['answer']}"

        ollama_result = await self.ollama.generate_response(user_question, context)

        return {
            'question': user_question,
            'answer': ollama_result['answer'],
            'source': ollama_result['source'],
            'method': 'ollama_llm',
            'model': ollama_result.get('model'),
            'local_context_used': bool(local_result),
            'response_time': (datetime.now() - start_time).total_seconds(),
            'cost_estimate': 0.0,  # Ollama is free too!
            'error': ollama_result.get('error')
        }

    def get_system_status(self) -> Dict:
        """Get status of all system components"""
        return {
            'local_knowledge': {
                'available': len(self.local_kb.knowledge_base) > 0,
                'files_loaded': len(self.local_kb.knowledge_base),
                'total_items': self.local_kb.collection.count()
            },
            'ollama': {
                'available': self.ollama.test_connection(),
                'host': self.ollama.base_url,
                'model': self.ollama.model
            },
            'cost_estimate': 0.0  # Everything is free!
        }

# Example usage and testing
async def test_hybrid_system():
    """Test the hybrid RAG system"""
    print("Testing Hybrid Virginia & Truckee Railroad RAG System")
    print("=" * 60)

    # Initialize system
    rag = HybridVTRailroadRAG()

    # Show system status
    status = rag.get_system_status()
    print(f"System Status:")
    print(f"  Local Knowledge: {status['local_knowledge']['files_loaded']} files, {status['local_knowledge']['total_items']} items")
    print(f"  Ollama LLM: {'Available' if status['ollama']['available'] else 'Unavailable'}")
    print()

    # Test queries
    test_questions = [
        "What time do trains run?",  # Should find locally
        "How much do tickets cost?",  # Should find locally
        "What's the history of the Virginia and Truckee Railroad?",  # May need Ollama
        "Where can I park my car?",  # Should find locally
    ]

    for question in test_questions:
        print(f"Question: {question}")
        print("-" * 40)

        result = await rag.query(question)

        print(f"Method: {result['method']}")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Response Time: {result['response_time']:.2f}s")
        print(f"Cost: ${result['cost_estimate']:.4f}")
        print()

if __name__ == "__main__":
    asyncio.run(test_hybrid_system())