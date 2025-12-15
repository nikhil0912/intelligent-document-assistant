"""Q&A Agent using LangChain and GenAI for intelligent document processing."""

import logging
from typing import List, Dict, Any, Optional
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from config import Config, get_config

logger = logging.getLogger(__name__)


class QAAgent:
    """Intelligent Q&A agent powered by LangChain and LLMs."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the Q&A agent.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or get_config()
        self.llm = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize the LLM based on configuration."""
        if self.config.LLM_PROVIDER == "openai":
            self.llm = ChatOpenAI(
                model_name=self.config.LLM_MODEL,
                temperature=self.config.LLM_TEMPERATURE,
                max_tokens=self.config.LLM_MAX_TOKENS,
                openai_api_key=self.config.OPENAI_API_KEY
            )
            logger.info(f"Initialized OpenAI LLM: {self.config.LLM_MODEL}")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.LLM_PROVIDER}")

    def setup_retrieval_qa(self, vector_store: FAISS) -> None:
        """Setup retrieval-based QA chain.
        
        Args:
            vector_store: FAISS vector store with embedded documents
        """
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question.
        If you don't know the answer, just say so.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_kwargs={"k": self.config.TOP_K_DOCS}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            memory=self.memory,
            return_source_documents=True
        )
        logger.info("Retrieval QA chain initialized")

    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question to the agent.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary with answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call setup_retrieval_qa first.")
        
        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                "sources": result.get("source_documents", [])
            }
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get conversation history.
        
        Returns:
            List of chat messages
        """
        return self.memory.load_memory_variables({})["chat_history"]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.memory.clear()
        logger.info("Conversation history cleared")


if __name__ == "__main__":
    # Example usage
    agent = QAAgent()
    print(f"Q&A Agent initialized with {agent.config.LLM_MODEL}")
    print("Ready to process documents and answer questions!")
