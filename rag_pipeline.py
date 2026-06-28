"""
RAG (Retrieval-Augmented Generation) Pipeline
Combines retrieval from vector store with LLM generation
Enhanced with async support, context compression and hallucination prevention
"""

from typing import List, Dict, Optional, Tuple
from vector_store import VectorStore
import os
import re
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class RAGPipeline:
    """Complete RAG pipeline for question answering"""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_model: str = "gemini-1.5-flash",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        max_context_length: int = 4000,
        enable_compression: bool = True,
        require_citations: bool = True
    ):
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_context_length = max_context_length
        self.enable_compression = enable_compression
        self.require_citations = require_citations

        # Auto-upgrade deprecated models if necessary
        if self.llm_model == "gemini-1.5-flash-latest":
            self.llm_model = "gemini-1.5-flash"

        self.use_gemini = "gemini" in self.llm_model.lower()
        
        if self.use_gemini:
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai is required for Gemini.")
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found.")
            
            genai.configure(api_key=google_api_key)
            self.gemini_model = genai.GenerativeModel(self.llm_model)
        else:
            raise NotImplementedError("Currently only Gemini is configured for the async pipeline.")

    async def retrieve_async(self, query: str, top_k: int = 3) -> List[Dict]:
        """Async wrapper for Pinecone search"""
        return await asyncio.to_thread(self.vector_store.search, query, top_k)

    def _is_greeting(self, text: str) -> bool:
        normalized = re.sub(r"[^a-zA-Z\s]", " ", text.lower()).strip()
        normalized = re.sub(r"\s+", " ", normalized)
        if not normalized:
            return False

        greeting_prefixes = ("hi", "hello", "hey", "yo", "good morning", "good afternoon", "good evening")
        if normalized in greeting_prefixes:
            return True
        return len(normalized.split()) <= 8 and normalized.startswith(greeting_prefixes)

    def truncate_contexts(self, contexts: List[str]) -> List[str]:
        # Simple truncation for now to keep things fast
        combined = "\n\n".join(contexts)
        if len(combined) <= self.max_context_length:
            return contexts

        truncated = []
        current_length = 0
        for ctx in contexts:
            if current_length + len(ctx) + 2 <= self.max_context_length:
                truncated.append(ctx)
                current_length += len(ctx) + 2
            else:
                break
        return truncated if truncated else [contexts[0][:self.max_context_length]]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_async(
        self,
        query: str,
        contexts: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        processed_contexts = self.truncate_contexts(contexts)

        if system_prompt is None:
            system_prompt = """You are an intelligent, helpful research assistant powered by a Retrieval-Augmented Generation (RAG) system. Your primary knowledge comes from the retrieved document contexts provided below.

## Core Principles
1. **Ground your answers in the provided contexts.** These are your primary source of truth. When a context supports your claim, cite it naturally using [Context N] (e.g., "The Normans originated in France [Context 1].").
2. **Be honest about uncertainty.** If the contexts don't contain enough information to fully answer the question, say so clearly — then optionally supplement with general knowledge, clearly marked as such.
3. **Never fabricate facts or citations.** Do not invent information and attribute it to a context. If you're unsure, say "I'm not certain based on the available documents."

## Answering Guidelines
- **Be concise and direct.** Lead with the answer, then provide supporting detail if needed.
- **Adapt your format to the question.** Simple factual questions deserve a brief answer. Complex or multi-part questions benefit from structured responses with bullet points or numbered lists.
- **For multi-part questions:** Address each part, noting which parts are covered by the contexts and which are not.
- **For ambiguous questions:** Briefly state your interpretation before answering.
- **For unanswerable questions:** State that the provided documents don't contain the answer. You may offer general knowledge if relevant, prefixed with "Based on general knowledge:" to distinguish it from document-grounded information.

## Citation Rules
- Cite the specific context that supports each claim: [Context 1], [Context 2], etc.
- You may combine information from multiple contexts in a single statement, citing all relevant ones.
- Do NOT cite a context unless it genuinely supports the claim.

## What NOT to Do
- Do not repeat the question back to the user.
- Do not use filler phrases like "Great question!" or "Based on the provided contexts, I can tell you that..."
- Do not generate information that contradicts the provided contexts.
- Do not hallucinate context numbers that don't exist."""

        combined_context = "\n\n".join([f"[Context {i+1}]:\n{ctx}" for i, ctx in enumerate(processed_contexts)])
        
        full_prompt = f"{system_prompt}\n\n===== RETRIEVED CONTEXTS =====\n{combined_context}\n\n===== USER QUESTION =====\n{query}"

        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        
        # Call Gemini Async
        response = await self.gemini_model.generate_content_async(
            full_prompt,
            generation_config=generation_config
        )
        
        answer = response.text.strip()
        
        if self.require_citations and "[Context" not in answer and "cannot answer" not in answer.lower():
            answer = f"{answer}\n\n[Note: This answer may not be fully grounded in the provided contexts]"
            
        return answer

    async def query_async(self, question: str, top_k: int = 3) -> Dict:
        """
        Complete async RAG query: retrieve + generate
        """
        if self._is_greeting(question):
            return {
                "answer": "Hi there. I can help answer questions from your documents.",
                "metadata": {"retrieved_docs": []}
            }

        # Retrieve documents asynchronously
        retrieved_docs = await self.retrieve_async(question, top_k=top_k)
        contexts = [doc['text'] for doc in retrieved_docs]

        # Generate answer asynchronously
        answer = await self.generate_async(question, contexts)

        return {
            "answer": answer,
            "metadata": {
                "retrieved_docs": retrieved_docs,
                "num_contexts": len(contexts),
                "model": self.llm_model
            }
        }
