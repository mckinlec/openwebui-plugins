"""
title: Ollama Query Expansion Pipeline
author: Your Name
date: 2024-11-24
version: 1.1
license: MIT
description: A pipeline for expanding user queries dynamically using the Ollama API.
requirements: pydantic, aiohttp, requests
"""

from typing import List, Optional
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import requests
import os

from utils.pipelines.main import get_last_user_message, get_last_assistant_message


class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to
        pipelines: List[str] = ["ihsgpt"]

        # Assign a priority level to the filter pipeline.
        priority: int = 0

        # Ollama-specific configurations
        OLLAMA_API_BASE_URL: str = "http://ollama:11434"
        EXPANSION_MODEL: str = "llama3.2"

    def __init__(self):
        # Filter pipeline configuration
        self.type = "filter"
        self.name = "Ollama Query Expansion Filter"
        self.valves = self.Valves(
            **{
                "pipelines": ["ihsgpt"],  # Target ihsgpt specifically
            }
        )

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    def expand_query(self, query: str) -> str:
        """
        Calls the Ollama API to expand the user query with synonyms and related terms.
        """
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.valves.EXPANSION_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in query optimization. Expand the following query with synonyms and related terms."
                },
                {
                    "role": "user",
                    "content": query
                },
            ],
        }

        try:
            response = requests.post(
                url=f"{self.valves.OLLAMA_API_BASE_URL}/api/chat",
                json=payload,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            expanded_query = "".join(
                [msg.get("message", {}).get("content", "") for msg in data]
            )
            return expanded_query.strip() if expanded_query else query
        except Exception as e:
            print(f"Failed to expand query: {e}")
            return query  # Fallback to the original query

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Process user input by expanding the query using the Ollama API.
        """
        print(f"inlet:{__name__}")

        messages = body.get("messages", [])
        user_message = get_last_user_message(messages)

        if not user_message:
            print("No user message to process.")
            return body

        original_query = user_message.get("content", "")
        print(f"Original query: {original_query}")

        expanded_query = self.expand_query(original_query)
        print(f"Expanded query: {expanded_query}")

        # Update the last user message with the expanded query
        for message in reversed(messages):
            if message["role"] == "user":
                message["content"] = expanded_query
                break

        body["messages"] = messages
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Optionally process assistant messages or other output if needed.
        """
        print(f"outlet:{__name__}")
        return body
