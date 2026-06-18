import json
from typing import Any, Dict

from src.infra.llm.local import OllamaLLM


class QueryAnalyzer:
    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyzes the user query to extract metadata filters.
        Returns a dictionary suitable for ChromaDB's `where` clause.
        """
        system_prompt = (
            "You are a query analysis expert. Your task is to extract metadata filters from a user query.\n"
            "We have the following metadata fields available for documents:\n"
            "- 'filename' (string): The name of the source file (e.g., 'report.pdf', 'data.txt')\n"
            "- 'section' (string): The section or heading context.\n\n"
            "Output the filters in JSON format. The keys should match the metadata fields.\n"
            "If no specific filters are found, return an empty JSON object: {}.\n\n"
            "Examples:\n"
            "User: 'What does the safety section in report.pdf say?'\n"
            'Output: {"filename": "report.pdf", "section": "Safety"}\n\n'
            "User: 'Summarize the introduction'\n"
            'Output: {"section": "Introduction"}\n\n'
            "User: 'Explain quantum physics'\n"
            "Output: {}\n\n"
            "Important:\n"
            "- ChromaDB metadata filtering only supports EXACT matches for strings.\n"
            "- Do not use '$contains'. Only use direct string values which imply equality.\n"
            "- ONLY output the JSON."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        try:
            response = self.llm.generate(messages, temperature=0.1)
            # Basic cleanup if the LLM adds markdown blocks
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]

            filters = json.loads(response.strip())

            # ChromaDB requires explicit $and for multiple conditions
            if len(filters) > 1:
                return {"$and": [{k: v} for k, v in filters.items()]}

            return filters
        except Exception as e:
            print(f"Error parsing query filters: {e}")
            return {}
