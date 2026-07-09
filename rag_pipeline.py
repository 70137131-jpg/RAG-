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
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from google import genai
    from google.genai import types
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
        require_citations: bool = True,
        llm_provider: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        openrouter_site_url: str = "",
        openrouter_app_name: str = "RAG Pipeline",
    ):
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_context_length = max_context_length
        self.enable_compression = enable_compression
        self.require_citations = require_citations
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.openrouter_base_url = openrouter_base_url
        self.openrouter_site_url = openrouter_site_url
        self.openrouter_app_name = openrouter_app_name

        # Auto-upgrade deprecated models if necessary
        if self.llm_model == "gemini-1.5-flash-latest":
            self.llm_model = "gemini-1.5-flash"

        provider = (llm_provider or os.getenv("LLM_PROVIDER", "")).lower().strip()
        if not provider:
            provider = "gemini" if "gemini" in self.llm_model.lower() else "openrouter"

        self.use_gemini = provider == "gemini"
        self.use_openrouter = provider == "openrouter"
        
        if self.use_gemini:
            if not GEMINI_AVAILABLE:
                raise ImportError("google-genai is required for Gemini.")
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found.")

            self.gemini_client = genai.Client(api_key=google_api_key)
        elif self.use_openrouter:
            if not self.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY not found.")
        else:
            raise NotImplementedError(f"Unsupported LLM provider: {provider}")

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

    def _is_document_gap_answer(self, answer: str) -> bool:
        normalized = re.sub(r"\s+", " ", answer.lower()).strip()
        gap_phrases = (
            "cannot answer",
            "can't answer",
            "do not contain",
            "don't contain",
            "does not contain",
            "doesn't contain",
            "not enough information",
            "insufficient information",
            "not provided in the",
            "not mentioned in the",
            "not covered in the",
        )
        return any(phrase in normalized for phrase in gap_phrases)

    def _generate_openrouter(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if self.openrouter_site_url:
            headers["HTTP-Referer"] = self.openrouter_site_url
        if self.openrouter_app_name:
            headers["X-OpenRouter-Title"] = self.openrouter_app_name

        response = requests.post(
            self.openrouter_base_url,
            headers=headers,
            json={
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

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

        if self.use_gemini:
            generation_config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )

            response = await self.gemini_client.aio.models.generate_content(
                model=self.llm_model,
                contents=full_prompt,
                config=generation_config,
            )
            answer = response.text.strip()
        else:
            answer = await asyncio.to_thread(self._generate_openrouter, full_prompt)
        
        if (
            self.require_citations
            and "[Context" not in answer
            and not self._is_document_gap_answer(answer)
        ):
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
