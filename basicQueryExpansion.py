"""
title: Simplified Query Expansion Pipeline
author: Your Name
date: 2024-11-24
version: 1.0
license: MIT
description: A streamlined pipeline for expanding or decomposing queries using an Ollama model.
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
        self.name = "Simplified Query Expansion Filter"
        self.valves = self.Valves(
            **{
                "pipelines": ["ihsgpt"],
            }
        )

    async def process_query_with_ollama(self, query: str, history: Optional[List[dict]], ollama_base_url: str, expansion_model: str) -> str:
        """
        Calls the Ollama model to process the query.
        """
        url = f"{ollama_base_url}/api/chat"
        system_message = """You are an expert in query optimization. Based on the complexity of the query:
        
        1. If the query is broad or vague, expand it with synonyms and related terms.
        2. If the query is complex or multifaceted, decompose it into 2-4 concise sub-queries.
        3. Respond ONLY with the expanded or decomposed query. Do not include explanations or additional commentary.
        """
        messages = [{"role": "system", "content": system_message}]
        
        # Add query to the payload
        messages.append({"role": "user", "content": f"Query: {query}"})
        
        # Add chat history if available
        if history:
            history_content = "\n".join([f"{entry['role']}: {entry['content']}" for entry in history])
            messages.append({"role": "user", "content": f"Chat history:\n{history_content}"})
            print(f"Chat history included:\n{history_content}")
        else:
            print("No chat history included.")

        payload = {
            "model": expansion_model,
            "messages": messages
        }

        print(f"Sending payload to Ollama:\n{json.dumps(payload, indent=2)}")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    content = []
                    async for line in response.content:
                        data = json.loads(line)
                        content.append(data.get("message", {}).get("content", ""))
                    result = "".join(content).strip()
                    print(f"Received response from Ollama:\n{result}")
                    return result
                else:
                    print(f"Failed to process query. HTTP status: {response.status}")
                    return query  # Fallback to the original query

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Processes user input by optimizing the query using the Ollama model.
        """
        print("Inlet function called")

        # Ensure the body is a dictionary
        if isinstance(body, str):
            body = json.loads(body)
        
        # Extract user query and history
        user_message = get_last_user_message(body.get("messages", []))
        chat_history = body.get("messages", [])[:-1] if "messages" in body else None
        
        if user_message:
            print(f"Original query: {user_message}")
            optimized_query = await self.process_query_with_ollama(
                query=user_message,
                history=chat_history,
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
        print("Outlet function called")
        return body
