"""
title: Query Expansion Filter Pipeline
author: Claude
date: 2024-11-24
version: 1.0
description: A filter pipeline that expands queries using llama3.2 for better RAG retrieval.
"""

from typing import List, Optional
from pydantic import BaseModel
import json
import aiohttp
from utils.pipelines.main import get_last_user_message

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

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    async def expand_query(self, query: str) -> str:
        """
        Uses llama3.2 to expand the query for better RAG retrieval.
        """
        print(f"QUERY-EXPANSION | Original Query: {query}")
        
        system_prompt = """You are an expert in query optimization. Based on the complexity of the query:
        
1. If the query is broad or vague, expand it with synonyms and related terms.
2. If the query is complex or multifaceted, decompose it into 2-4 concise sub-queries.
3. Respond ONLY with the expanded or decomposed query. Do not include explanations or additional commentary.
        """

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
                    print(f"QUERY-EXPANSION | Expanded Query: {expanded.strip()}")
                    return expanded.strip()
                return query

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Expands user queries using llama3.2 before they go to the main LLM.
        """
        print(f"inlet:{__name__}")

        required_keys = ["messages"]
        missing_keys = [key for key in required_keys if key not in body]
        
        if missing_keys:
            print(f"Error: Missing keys in request body: {', '.join(missing_keys)}")
            return body

        if isinstance(body, str):
            body = json.loads(body)
        
        user_message = get_last_user_message(body.get("messages", []))
        
        # Only process actual user queries, not system messages or JSON
        if user_message and not (user_message.startswith("###") or user_message.startswith("{")):
            expanded_query = await self.expand_query(user_message)
            
            # Update the last user message with the expanded query
            for message in reversed(body.get("messages", [])):
                if message["role"] == "user":
                    message["content"] = expanded_query
                    break

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Pass-through for the main LLM's responses.
        """
        print(f"outlet:{__name__}")
        return body
