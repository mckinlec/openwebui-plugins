"""
title: Simple RAG Query Expansion Filter
description: A streamlined query expansion pipeline using llama3.2 for RAG optimization
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
        self.name = "Simple RAG Query Expansion"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
            }
        )

    async def expand_query(self, query: str) -> str:
        """
        Uses llama3.2 to expand the query for better RAG retrieval.
        """
        print(f"Original query: {query}")
        
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

        print("Sending payload to Ollama:")
        print(json.dumps(payload, indent=2))

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.valves.ollama_base_url}/api/chat", json=payload) as response:
                print(f"Received response from Ollama with status: {response.status}")
                if response.status == 200:
                    expanded = ""
                    async for line in response.content:
                        data = json.loads(line)
                        expanded += data.get("message", {}).get("content", "")
                    print(f"Optimized query: {expanded.strip()}")
                    return expanded.strip()
                return query

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Processes user input by expanding the query using llama3.2.
        """
        print(f"inlet:{__name__}")

        if isinstance(body, str):
            print("Converting body from string to dict")
            body = json.loads(body)
        
        user_message = get_last_user_message(body.get("messages", []))
        print(f"Extracted user message: {user_message}")

        if user_message:
            expanded_query = await self.expand_query(user_message)
            print(f"Expanded query: {expanded_query}")

            # Update the last user message with the expanded query
            for message in reversed(body.get("messages", [])):
                if message["role"] == "user":
                    message["content"] = expanded_query
                    print("Updated message content with expanded query")
                    break

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Pass-through for responses.
        """
        print("Outlet function called")
        return body
