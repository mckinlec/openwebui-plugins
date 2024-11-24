"""
title: Ollama Advanced Query Pipeline
author: Christopher McKinley
date: 2024-11-23
version: 1.0
license: MIT
description: A pipeline for dynamically expanding or decomposing queries using an Ollama model.
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
        ollama_base_url: str = "http://ollama:11434"  # Default URL for Ollama
        expansion_model: str = "llama3.2"  # Replace with your model's name

    def __init__(self):
        self.type = "filter"
        self.name = "Ollama Advanced Query Filter"
        self.valves = self.Valves(
            **{
                "pipelines": ["ihsgpt"],  # Target pipelines
            }
        )

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def process_query_with_ollama(self, query: str, ollama_base_url: str, expansion_model: str) -> str:
        """
        Calls the Ollama model to either expand or decompose the query.
        """
        url = f"{ollama_base_url}/api/chat"
        system_message = """You are an advanced AI for query optimization.
Your task is to determine the best approach for this query:
1. Expand the query with synonyms and related terms if it is vague or broad.
2. Decompose the query into 2-4 specific sub-questions if it is complex or multifaceted.
Choose the best approach based on the query and respond only with the result. Do not include any explanation or metadata."""

        payload = {
            "model": expansion_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Query: {query}"}
            ]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    content = []
                    async for line in response.content:
                        data = json.loads(line)
                        content.append(data.get("message", {}).get("content", ""))
                    return "".join(content).strip()
                else:
                    print(f"Failed to process query with Ollama, status code: {response.status}")
                    return query  # Fallback to the original query

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Processes user input by optimizing the query using the Ollama model.
        """
        print(f"inlet:{__name__}")

        # Ensure the body is a dictionary
        if isinstance(body, str):
            body = json.loads(body)
        
        # Extract user query
        user_message = get_last_user_message(body.get("messages", []))

        # Process query if a valid user message exists
        if user_message:
            print(f"Original query: {user_message}")
            optimized_query = await self.process_query_with_ollama(
                query=user_message,
                ollama_base_url=self.valves.ollama_base_url,
                expansion_model=self.valves.expansion_model
            )
            print(f"Optimized query: {optimized_query}")

            # Update the last user message with the optimized query
            for message in reversed(body.get("messages", [])):
                if message["role"] == "user":
                    message["content"] = optimized_query
                    break

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Optionally processes assistant messages or other output if needed.
        """
        return body
