import json
import aiohttp
import logging
from typing import List, Optional, Dict
from pydantic import BaseModel
from utils.pipelines.main import get_last_user_message

logging.basicConfig(level=logging.DEBUG)  # Enable detailed logs
logger = logging.getLogger(__name__)

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        ollama_base_url: str = "http://ollama:11434"
        expansion_model: str = "llama3.2"

    def __init__(self):
        self.type = "filter"
        self.name = "Enhanced Query Expansion Pipeline"
        self.valves = self.Valves(
            **{
                "pipelines": ["ihsgpt"],  # Target specific pipelines
            }
        )

    async def query_transform(self, query: str, transform_type: str, context: Dict[str, any]) -> str:
        """
        Transforms query with dynamic context-aware prompting.
        """
        logger.debug(f"Starting {transform_type} transformation for query: {query} with context: {context}")

        # Base system message for LLM
        system_base = """You are an AI assistant specializing in query enhancement for information retrieval. Your task is to provide only the response. Avoid extraneous commentary, formatting, or additional metadata."""

        if context.get("is_ihs_related"):
            system_base += "\n\nRelevant Context:\n"
            system_base += f"- This query relates to: {', '.join(context.get('relevant_aspects', []))}\n"
            system_base += f"- Additional context: {context.get('suggested_context', '')}"

        # Prompts for transformations
        prompts = {
            "rewrite": {
                "system": system_base + "\n\nRewrite the query to make it more specific and detailed while preserving the original intent. Respond only with the enhanced query.",
                "user": f"Query: {query}"
            },
            "decompose": {
                "system": system_base + "\n\nBreak the query into 2-4 specific, answerable sub-questions that provide comprehensive coverage of the topic. Respond only with the list of sub-questions.",
                "user": f"Query: {query}"
            }
        }

        url = f"{self.valves.ollama_base_url}/api/chat"
        payload = {
            "model": self.valves.expansion_model,
            "messages": [
                {"role": "system", "content": prompts[transform_type]["system"]},
                {"role": "user", "content": prompts[transform_type]["user"]}
            ]
        }

        logger.debug(f"{transform_type} transformation payload:\n{json.dumps(payload, indent=2)}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    logger.debug(f"{transform_type} transformation HTTP status: {response.status}")
                    if response.status == 200:
                        raw_response = await response.text()
                        logger.debug(f"Raw LLM response for {transform_type} transformation:\n{raw_response}")

                        # Clean and return the response
                        cleaned_response = raw_response.strip("```").strip()
                        logger.info(f"Transformation result [{transform_type}]:\n{cleaned_response}")
                        return cleaned_response
        except Exception as e:
            logger.error(f"Error during {transform_type} transformation: {str(e)}")
            return query

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Processes user input by expanding the query using the Ollama model.
        """
        logger.info(f"Received inlet request with body: {json.dumps(body, indent=2)}")

        # Ensure the body is a dictionary
        if isinstance(body, str):
            body = json.loads(body)

        # Extract user query
        user_message = get_last_user_message(body.get("messages", []))

        if user_message:
            # Log the original user query
            logger.info(f"Original user query: {user_message}")

            # Perform query transformation
            expanded_query = await self.query_transform(
                query=user_message,
                transform_type="rewrite",
                context={"is_ihs_related": True, "relevant_aspects": ["Example"], "suggested_context": "Example context"}
            )

            # Log the expanded query
            logger.info(f"Expanded query result:\n{expanded_query}")

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
