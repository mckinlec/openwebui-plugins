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

    async def expand_query(self, query: str) -> str:
        """
        Uses llama3.2 to expand the query for better RAG retrieval.
        """
        print("=== QUERY EXPANSION PIPELINE ===")
        print(f"📥 ORIGINAL QUERY: {query}")
        
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
                    print(f"🔄 EXPANDED QUERY: {expanded.strip()}")
                    return expanded.strip()
                return query

    def is_system_message(self, message: str) -> bool:
        """
        Check if this is a system/internal message that should be ignored.
        """
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
        print("⚡ INLET FUNCTION CALLED")
        
        if isinstance(body, str):
            body = json.loads(body)

        user_message = get_last_user_message(body.get("messages", []))
        
        # Skip processing if:
        # 1. No user message
        # 2. It's a system message
        # 3. It looks like an already expanded query
        if (user_message and 
            not self.is_system_message(user_message) and
            not any(x in user_message for x in ["OR", "AND", "(", ")", "`"])):
            
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
        print("🔚 OUTLET FUNCTION CALLED")
        return body
