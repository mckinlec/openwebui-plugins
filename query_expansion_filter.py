"""
title: Ollama Query Expansion Pipeline
author: Your Name
date: 2024-11-24
version: 1.4
license: MIT
description: A pipeline for dynamically expanding user queries using the Ollama API.
requirements: pydantic, aiohttp
"""

from typing import List, Optional
from pydantic import BaseModel
import json
import aiohttp
from utils.pipelines.main import get_last_user_message

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["ihsgpt"]  # Only expand queries for the 'ihsgpt' model
        priority: int = 0
        expansion_model: str = "llama3.2"
        ollama_base_url: str = "http://ollama:11434"

    def __init__(self):
        self.type = "filter"
        self.name = "Ollama Query Expansion Filter"
        self.valves = self.Valves(
            **{
                "pipelines": ["ihsgpt"],  # Target specific pipelines
            }
        )

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    async def expand_query_with_ollama(self, query: str, expansion_model: str, ollama_base_url: str) -> str:
        """
        Calls the Ollama API to expand the user query with synonyms and related terms.
        """
        url = f"{ollama_base_url}/api/chat"
        payload = {
            "model": expansion_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in query optimization. Expand the following query with synonyms and related terms."
                },
                {
                    "role": "user",
                    "content": query
                },
            ]
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        content = []
                        async for line in response.content:
                            data = json.loads(line)
                            content.append(data.get("message", {}).get("content", ""))
                        return "".join(content).strip()
                    else:
                        print(f"Failed to expand query, status code: {response.status}")
                        return query  # Return original query if expansion fails
            except Exception as e:
                print(f"Error while calling Ollama API: {e}")
                return query  # Fallback to original query in case of error

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Processes user input by expanding the query before it is sent to the LLM.
        """
        print(f"inlet:{__name__}")

        # Ensure the body is a dictionary
        if isinstance(body, str):
            body = json.loads(body)

        messages = body.get("messages", [])
        user_message = get_last_user_message(messages)

        if not user_message:
            print("No user message found for query expansion.")
            return body

        # Expand the user's query
        print(f"Original query: {user_message}")
        expanded_query = await self.expand_query_with_ollama(
            query=user_message,
            expansion_model=self.valves.expansion_model,
            ollama_base_url=self.valves.ollama_base_url
        )
        print(f"Expanded query: {expanded_query}")

        # Update the user's message with the expanded query
        for message in reversed(messages):
            if message["role"] == "user":
                message["content"] = expanded_query
                break

        body["messages"] = messages
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Processes output from the LLM if needed. Currently passes data through without modification.
        """
        print(f"outlet:{__name__}")
        return body
