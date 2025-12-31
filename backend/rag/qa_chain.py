"""
QA Chain Module
Implements RAG-based question answering using Groq LLM
"""
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Tuple, Dict
import os
from dotenv import load_dotenv

load_dotenv()


class RAGQAChain:
    """RAG-based Question Answering Chain using Groq"""
    
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        """
        Initialize RAG QA Chain.
        
        Args:
            model_name: Groq model name (default: llama-3.1-8b-instant)
                       Other options: llama-3.3-70b-versatile, mixtral-8x7b-32768, gemma-7b-it
        """
        self.model_name = model_name
        self.llm = None
        self._initialize_llm()
        self._create_prompt_template()
    
    def _initialize_llm(self):
        """Initialize Groq LLM"""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Try different parameter names for ChatGroq compatibility
        try:
            # Try with model_name parameter first
            self.llm = ChatGroq(
                groq_api_key=api_key,
                model_name=self.model_name,
                temperature=0.1
            )
            print(f"Initialized Groq LLM: {self.model_name}")
        except (TypeError, ValueError) as e:
            # Fallback: try with model parameter instead of model_name
            try:
                self.llm = ChatGroq(
                    groq_api_key=api_key,
                    model=self.model_name,
                    temperature=0.1
                )
                print(f"Initialized Groq LLM: {self.model_name}")
            except Exception as e2:
                # If model fails, try alternative fallback models
                fallback_models = ["mixtral-8x7b-32768", "gemma-7b-it"]
                last_error = e2
                for fallback_model in fallback_models:
                    print(f"Warning: Failed to initialize {self.model_name}, trying fallback: {fallback_model}")
                    try:
                        self.llm = ChatGroq(
                            groq_api_key=api_key,
                            model=fallback_model,
                            temperature=0.1
                        )
                        self.model_name = fallback_model
                        print(f"Initialized Groq LLM with fallback model: {fallback_model}")
                        return
                    except Exception as e3:
                        last_error = e3
                        continue
                
                # If all models fail, raise error
                raise Exception(f"Error initializing Groq LLM. Tried {self.model_name} and fallbacks {fallback_models}. Last error: {str(last_error)}")
        except Exception as e:
            raise Exception(f"Error initializing Groq LLM: {str(e)}")
    
    def _create_prompt_template(self):
        """Create prompt template for RAG"""
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions and performs tasks based on the provided context from uploaded documents.

Your capabilities:
- Answer specific questions using information from the context take references from the context.
- Summarize documents or sections when asked
- Extract key information and insights
- Provide analysis based on the provided content

Rules:
1. Use ONLY the information provided in the context below. Do not use external knowledge.
2. For summarization requests: Provide a comprehensive summary of the relevant content from the context.
3. For specific questions: Answer directly using the context. If the specific answer is not in the context, say "The specific information is not available in the uploaded documents."
4. Synthesize information when appropriate (e.g., for summaries, comparisons, or analyses).
5. Be precise and cite page numbers when available.
6. If the context is completely empty or irrelevant to the question, say "Answer not found in uploaded documents"

Context from uploaded documents:
{context}"""),
            ("human", "{question}")
        ])
    
    def answer_question(self, question: str, retrieved_chunks: List[Tuple[Dict, float]]) -> Dict:
        """
        Answer question using retrieved context.
        
        Args:
            question: User's question
            retrieved_chunks: List of (metadata_dict, distance) tuples from vector search
            
        Returns:
            Dictionary with 'answer' and 'sources' keys
        """
        if not retrieved_chunks:
            return {
                "answer": "Answer not found in uploaded documents",
                "sources": []
            }
        
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        
        # Sort chunks by page number for better organization (especially for summarization)
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x[0].get('page', 0))
        
        for chunk_meta, distance in sorted_chunks:
            text = chunk_meta.get('text', '').strip()
            if not text:
                continue
                
            page = chunk_meta.get('page', 'Unknown')
            filename = chunk_meta.get('filename', 'Unknown')
            
            context_parts.append(f"[Page {page}]: {text}")
            sources.append(f"Page {page} of {filename}")
        
        context = "\n\n".join(context_parts)
        
        # If no valid context, return early
        if not context.strip():
            return {
                "answer": "Answer not found in uploaded documents",
                "sources": []
            }
        
        # Format prompt
        messages = self.prompt_template.format_messages(
            context=context,
            question=question
        )
        
        try:
            # Get response from LLM
            response = self.llm.invoke(messages)
            answer = response.content.strip()
            
            # Verify answer is not generic (but allow summarization responses)
            answer_lower = answer.lower()
            generic_responses = ["i don't know", "i cannot answer"]
            # Don't override if it's a valid response or if it contains substantial content
            if not answer or (answer_lower in generic_responses and len(answer) < 50):
                # Only override if it's truly generic and short
                if "summar" not in question.lower() and "overview" not in question.lower():
                    answer = "Answer not found in uploaded documents"
            
            return {
                "answer": answer,
                "sources": list(set(sources))  # Remove duplicates
            }
        
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": []
            }

