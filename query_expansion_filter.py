"""
title: Query Expansion Pipeline
author: Erica
date: 2024-11-24
version: 1.0
license: MIT
description: A pipeline for expanding user queries using llama3.2 via the Ollama API before passing them to ihsgpt.
requirements: pydantic, aiohttp
"""

from typing import List, Optional
from pydantic import BaseModel
import json
import aiohttp
from utils.pipelines.main import get_last_user_message

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["ihsgpt"]  
        priority: int = 0
        ollama_base_url: str = "http://localhost:11434"  # Base URL for Ollama API
        expansion_model: str = "llama3.2"  # Model to use for query expansion

    def __init__(self):
        self.type = "filter"
        self.name = "Query Expansion Filter"
        self.valves = self.Valves(
            **{
                "pipelines": ["ihsgpt"],  # Target only ihsgpt
            }
        )

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def expand_query(self, user_query: str, ollama_base_url: str, model: str) -> str:
        """
        Use the Ollama API to expand the user's query.
        """
        url = f"{ollama_base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in query optimization. Expand the following query to improve its utility for knowledge retrieval:"
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    expanded_query = []
                    async for line in response.content:
                        data = json.loads(line)
                        expanded_query.append(data.get("message", {}).get("content", ""))
                    return "".join(expanded_query).strip()
                else:
                    print(f"Failed to expand query, status code: {response.status}")
                    return user_query  # Fallback to original query if expansion fails

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Modify the user's query by expanding it before passing it to ihsgpt.
        """
        print(f"inlet:{__name__}")

        # Ensure the body is a dictionary
        if isinstance(body, str):
            body = json.loads(body)
        
        model = body.get("model", "")
        if model != "ihsgpt":
            return body  # Skip processing if the model isn't ihsgpt

        # Get the user's query
        user_message = get_last_user_message(body["messages"])
        if user_message:
            original_query = user_message["content"]
            print(f"Original query: {original_query}")

            # Expand the query using llama3.2
            expanded_query = await self.expand_query(
                user_query=original_query,
                ollama_base_url=self.valves.ollama_base_url,
                model=self.valves.expansion_model
            )
            print(f"Expanded query: {expanded_query}")

            # Update the user's query in the message history
            for message in body["messages"]:
                if message["role"] == "user":
                    message["content"] = expanded_query
                    break

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Pass the response from ihsgpt unchanged.
        """
        print(f"outlet:{__name__}")
        return body
