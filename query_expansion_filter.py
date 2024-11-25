"""
title: Query Expansion Filter
author: Claude
date: 2024-11-24
version: 1.0
description: A filter pipeline that expands queries using llama3.2 for better RAG retrieval.
"""

from typing import List, Optional
from pydantic import BaseModel
import json
import aiohttp
from utils.pipelines.main import get_last_user_message, get_last_assistant_message

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        ollama_base_url: str = "http://ollama:11434"
        expansion_model: str = "llama3.2"

    def __init__(self):
        self.type = "filter"
        self.name = "Query Expansion Filter"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
            }
        )

    async def expand_query(self, query: str) -> str:
        """
        Uses llama3.2 to expand the query for better RAG retrieval.
        """
        print(f"📥 Original query: {query}")
        
        system_prompt = """You are an expert at expanding search queries to improve retrieval.
For the given query, add relevant concepts, synonyms and related terms in natural language.
Avoid using boolean operators (AND, OR) and parentheses.
Focus on adding relevant terms and phrases that help capture the full meaning.
Respond ONLY with the expanded query in natural language."""

        payload = {
            "model": self.valves.expansion_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"}
            ]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.valves.ollama_base_url}/api/chat", json=payload) as response:
                if response.status == 200:
                    expanded = ""
                    async for line in response.content:
                        data = json.loads(line)
                        expanded += data.get("message", {}).get("content", "")
                    expanded = expanded.strip()
                    print(f"🔄 Expanded query: {expanded}")
                    return expanded
                return query

    def is_system_message(self, message: str) -> bool:
        """Check if this is a system/internal message that should be ignored."""
        if not message:
            return True
        return (
            message.startswith("###") or
            message.startswith("{") or
            "```" in message or
            "Task:" in message
        )

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Expands user queries using llama3.2 before they go to the main LLM.
        """
        print(f"inlet:{__name__}")

        messages = body.get("messages", [])
        user_message = get_last_user_message(messages)
        
        if user_message and not self.is_system_message(user_message):
            expanded_query = await self.expand_query(user_message)
            
            # Update the last user message with the expanded query
            for message in reversed(messages):
                if message["role"] == "user":
                    message["content"] = expanded_query
                    break

            body = {**body, "messages": messages}
        
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Pass-through for the main LLM's responses.
        """
        print(f"outlet:{__name__}")
        return body
