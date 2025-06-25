#!/usr/bin/env python3
"""
RAG Query System for Virginia & Truckee Railroad
Combines local embeddings with Claude API for intelligent responses
"""

import json
import os
from typing import Dict, List
from embedding_pipeline import VTRailroadEmbeddingPipeline
import anthropic
from datetime import datetime

class VTRailroadRAG:
    def __init__(self, anthropic_api_key: str = None):
        """
        Initialize the RAG system

        Args:
            anthropic_api_key: Your Anthropic API key (or set ANTHROPIC_API_KEY env var)
        """
        # Initialize embedding pipeline
        self.embedding_pipeline = VTRailroadEmbeddingPipeline()

        # Initialize Anthropic client
        api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Either pass it as parameter or set ANTHROPIC_API_KEY environment variable"
            )

        self.anthropic_client = anthropic.Anthropic(api_key=api_key)

        # System prompt for the Virginia & Truckee Railroad assistant
        self.system_prompt = """You are a helpful assistant for the Virginia & Truckee Railroad, a historic tourist railroad in Nevada. You help visitors with information about train schedules, fares, routes, reservations, and general information about the railroad.

Key guidelines:
- Be friendly and enthusiastic about the railroad
- Provide accurate information based on the context provided
- If you don't have specific information, direct visitors to contact the railroad directly
- Mention specific details like times, prices, and locations when available
- Always encourage visitors to visit and ride the train
- If asked about current schedules or availability, remind visitors to check the website or call for the most up-to-date information

The Virginia & Truckee Railroad operates between Virginia City and Gold Hill, Nevada, and also has a Carson City route. It's a historic narrow-gauge railroad that was originally built in 1869 during the Comstock mining era."""

    def query(self, user_question: str, max_context_chunks: int = 3) -> Dict:
        """
        Process a user question using RAG

        Args:
            user_question: The user's question
            max_context_chunks: Maximum number of context chunks to include

        Returns:
            Dictionary with response and metadata
        """
        # Step 1: Get relevant context from embeddings
        print(f"Searching for relevant content...")
        context_results = self.embedding_pipeline.query_similar_content(
            user_question,
            n_results=max_context_chunks
        )

        # Step 2: Format context for Claude
        context_text = self._format_context(context_results)

        # Step 3: Create the prompt for Claude
        user_prompt = f"""Based on the following information about the Virginia & Truckee Railroad, please answer the user's question.

CONTEXT INFORMATION:
{context_text}

USER QUESTION: {user_question}

Please provide a helpful and accurate response based on the context provided. If the context doesn't contain enough information to fully answer the question, say so and suggest how the visitor can get more information."""

        # Step 4: Get response from Claude
        print(f"Generating response with Claude...")
        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",  # Use Claude Sonnet 4
                max_tokens=1000,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            assistant_response = response.content[0].text

        except Exception as e:
            print(f"Error calling Claude API: {e}")
            assistant_response = "I apologize, but I'm having trouble accessing information right now. Please visit virginiatruckee.com or call the railroad directly for assistance."

        # Step 5: Return formatted response
        return {
            "question": user_question,
            "answer": assistant_response,
            "context_used": context_results["results"],
            "context_sources": [result["url"] for result in context_results["results"]],
            "timestamp": datetime.now().isoformat()
        }

    def _format_context(self, context_results: Dict) -> str:
        """Format the context results for Claude"""
        if not context_results["results"]:
            return "No relevant context found."

        formatted_context = ""
        for i, result in enumerate(context_results["results"], 1):
            formatted_context += f"SOURCE {i} (from {result['title']}):\n"
            formatted_context += f"{result['content'][:800]}...\n\n"  # Limit context length

        return formatted_context

    def chat_interface(self):
        """Simple command-line chat interface for testing"""
        print("Virginia & Truckee Railroad Assistant")
        print("=" * 50)
        print("Ask me anything about the Virginia & Truckee Railroad!")
        print("Type 'quit' to exit")
        print()

        while True:
            try:
                question = input("You: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("Thank you for using the V&T Railroad Assistant!")
                    break

                if not question:
                    continue

                print("Assistant: [Thinking...]")
                response = self.query(question)
                print(f"Assistant: {response['answer']}")
                print()

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def test_system(self):
        """Run a series of test queries"""
        test_questions = [
            "What time do trains run?",
            "How much do tickets cost?",
            "Where is the depot located?",
            "What is the history of the Virginia & Truckee Railroad?",
            "Can I see wild horses on the train ride?",
            "What special events do you have?",
            "How long is the train ride?",
            "Where can I park?"
        ]

        print("Testing RAG System with Sample Questions")
        print("=" * 60)

        for i, question in enumerate(test_questions, 1):
            print(f"\nTEST {i}: {question}")
            print("-" * 40)

            try:
                response = self.query(question)
                print(f"Answer: {response['answer'][:200]}...")
                print(f"Sources used: {len(response['context_used'])}")

            except Exception as e:
                print(f"Error: {e}")

        print("\nTesting complete!")

def main():
    """Main function"""
    print("Virginia & Truckee Railroad RAG System")
    print("=" * 50)

    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it with: export ANTHROPIC_API_KEY='your-api-key-here'")
        return

    try:
        # Initialize RAG system
        rag = VTRailroadRAG()

        # Check if database exists
        stats = rag.embedding_pipeline.get_database_stats()
        if stats["total_chunks"] == 0:
            print("ERROR: No content found in database.")
            print("Run 'python embedding_pipeline.py' first to create the embeddings.")
            return

        print(f"Database loaded: {stats['total_chunks']} chunks available")
        print()

        # Run test queries
        choice = input("Run (t)est queries or start (i)nteractive chat? [t/i]: ").strip().lower()

        if choice == 't':
            rag.test_system()
        else:
            rag.chat_interface()

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()