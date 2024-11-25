"""
title: Ollama Query Expansion Pipeline
author: Your Name
date: 2024-11-23
version: 1.1
license: MIT
description: A pipeline for expanding queries dynamically using an Ollama model.
requirements: pydantic, aiohttp
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
        expansion_model: str = "phi3:14b" 

    def __init__(self):
        self.type = "filter"
        self.name = "Ollama Query Expansion Filter"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
            }
        )

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def expand_query_with_ollama(self, query: str, ollama_base_url: str, expansion_model: str) -> str:
        """
        Calls the Ollama model to expand the query.
        """
        url = f"{ollama_base_url}/api/chat"
        payload = {
            "model": expansion_model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Expand this query for a knowledge base search, including relevant concepts and broader topics: {query}" 
                }
            ]
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()  # Raise an exception for bad status codes

                    content = []
                    async for line in response.content:
                        data = json.loads(line)
                        content.append(data.get("message", {}).get("content", ""))
                    return "".join(content)
        except aiohttp.ClientError as e:
            print(f"Error expanding query with Ollama: {e}")
            return query  # Fallback to original query

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Processes user input by expanding the query using the Ollama model.
        """
        print(f"inlet:{__name__}")

        # Ensure the body is a dictionary
        if isinstance(body, str):
            body = json.loads(body)
        
        # Extract user query
        user_message = get_last_user_message(body.get("messages", []))

        # Expand query if a valid user message exists
        if user_message:
            expanded_query = await self.expand_query_with_ollama(
                query=user_message,
                ollama_base_url=self.valves.ollama_base_url,
                expansion_model=self.valves.expansion_model
            )
            print(f"Expanded query: {expanded_query}")

            # Update the last user message with the expanded query
            for message in reversed(body.get("messages", [])):
                if message["role"] == "user":
                    message["content"] = expanded_query
                    break

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Optionally processes assistant messages or other output if needed.
        """
        return body
