from typing import Dict, List, Optional

import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama


class OllamaLLM:
    def __init__(
        self, model: str = "llama3.2", base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url

    def is_available(self) -> bool:
        # LangChain doesn't have a native "ping" method, so we keep this lightweight check
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except requests.RequestException:
            return []

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        # 1. Initialize LangChain's ChatOllama client
        chat_llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=temperature,
            num_predict=max_tokens,
        )

        # 2. LangChain natively accepts standard dictionary message lists!
        response = chat_llm.invoke(messages)

        # 3. Return the string content from the AIMessage object
        return str(response.content)

    def generate_with_context(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]] | None = None,
        temperature: float = 0.7,
    ) -> str:

        chat_llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=temperature,
        )

        system_template = (
            "You are an expert research assistant dedicated to providing accurate, verified information.\n\n"
            "STRICT GUIDELINES:\n"
            "1. **Context Priority**: Answer strictly based on the 'Context' provided below if possible.\n"
            "2. **Verification & Labeling**:\n"
            "   - Never present generated/inferred content as fact.\n"
            "   - If information is not in the context and you use general knowledge, you MUST label it.\n"
            "   - Start sentences with **[Inference]**, **[Speculation]**, or **[Unverified]** if the content is not directly supported by the context.\n"
            "   - If you cannot verify something, say: 'I cannot verify this' or 'My knowledge base does not contain that'.\n"
            "3. **Tone & Style**:\n"
            "   - Maintain a professional, objective, academic tone.\n"
            "   - Do not paraphrase or reinterpret user input unless requested.\n"
            "   - **CRITICAL**: All mathematical formulas, equations, and symbols MUST be written in LaTeX format (e.g., $E=mc^2$).\n"
            "4. **Forbidden Absolutes**: Unless sourced from context, label claims using words like 'Prevent', 'Guarantee', 'Fixes', 'Eliminates', 'Ensures'.\n"
            "5. **Correction Protocol**: If you realize a previous mistake, say: 'Correction: I previously made an unverified claim...'.\n\n"
            "Context:\n{context}"
        )

        # Build the prompt using LangChain's template engine
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{query}"),
        ])

        # Create an LCEL (LangChain Expression Language) pipeline
        chain = prompt | chat_llm | StrOutputParser()

        # Execute the chain
        return chain.invoke({
            "context": context,
            "chat_history": conversation_history or [],
            "query": query,
        })
