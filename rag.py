# from langchain.tools.base import BaseTool  # Superclass to build our agent
# from langchain_community.llms import HuggingFaceHub  # Use LLM models stored in HF, deprecated
from langchain_core.runnables import Runnable
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)  # Structure Prompts
from langchain.docstore.document import (
    Document,
)  # Format documents retrieved from the datasets
from langchain_community.retrievers import (
    BM25Retriever,
)  # Retrieve documents from dataset
from huggingface_hub import InferenceClient

from datasets import load_dataset  # Load Guests Dataset
import logging

from typing import Optional, Union, Any, List  # Typed programming
import os, getpass  # Get HF API token

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler("rag.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

for handler in logger.handlers:
    handler.flush()


# TODO: Migrate _get_var to utils script
# Get HF API token for model reference
def _get_var(var) -> None:
    if os.getenv(var):
        logger.info(f"{var} successfully processed")
    else:
        os.environ[var] = getpass.getpass(prompt=f"Type the value of {var}: ")


var = "HUGGINGFACEHUB_API_TOKEN"
_get_var(var)
hf_token = os.getenv(var)


class RAGTool(Runnable):
    name = "rag_tool"
    description = "Retrieves detailed information about gala guests based on their name or relation."
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest you want information about.",
        }
    }
    output_type = "string"

    def __init__(self) -> None:
        # self.index = None
        self.query: Optional[str] = None
        self.query_translated: Optional[str] = None
        self.docs: list[Document] = None

        # self.chat_template: Any = ChatPromptTemplate([
        #    SystemMessagePromptTemplate(prompt=["You are a proffesional butler, specialized to organized the most successful Galas"]),
        #    HumanMessagePromptTemplate(prompt=["Find information about: {query}"])
        # ])

        # Deprecated and waiting for LangChain maintance
        # self.llm: Any = HuggingFaceEndpoint(
        #    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        #    task="text-generation",
        #    huggingfacehub_api_token=hf_token,
        #    temperature=0.1,
        #    max_new_tokens=100
        # )
        self.llm = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.1",  # "Qwen/Qwen2.5-Coder-32B-Instruct",
            token=hf_token,
        )

    def _translate_query(self) -> None:
        # Step-Back
        # TODO: solve inference credits limit
        # prompt = f"Write step-back questions to optimize clarity: {self.query}"  # Temporarily omited due to credit limits
        # ai_response = self.llm.text_generation(
        #    prompt=prompt,
        #    max_new_tokens=100,
        #    temperature=0.1
        # ) # TODO: Estructurar prompt antes de pasarlo

        # self.query_translated = ai_response
        self.query_translated = self.query  # TODO: delete this line

    def _load_dataset(self) -> None:
        # Import guest dataset
        logger.info(f"Processing user query...")
        url = "agents-course/unit3-invitees"  # TODO: Create a Basemodel subclass to save all locations
        guest_dataset = load_dataset(path=url, split="train")

        self.docs = [
            Document(
                page_content="\n".join(
                    [
                        f'Name: {guest_info["name"]}',
                        f'Relation: {guest_info["relation"]}',
                        f'Description: {guest_info["description"]}',
                        f'Email: {guest_info["email"]}',
                    ]
                ),
                metadata={"name": guest_info["name"]},
            )
            for guest_info in guest_dataset
        ]

    def _embed_text(self) -> None:
        pass

    def _retrieval(self) -> str:
        # Verify is the docs are already loaded
        try:
            _ = self.docs
        except AttributeError:
            self._load_dataset()

        # Create retriever
        self.retriever = BM25Retriever.from_documents(documents=self.docs)

        # Similarity search
        result = self.retriever.invoke(self.query_translated)
        result = result[:3]  # Get the 3 most relevant docs
        retrieved_docs = "\n\n".join([doc.page_content for doc in result])

        if retrieved_docs:
            # self.retrieved_docs = retrieved_docs
            return retrieved_docs
        else:
            return "No information found about this Guest"

    def _augment_prompt(self) -> None:
        pass

    def invoke(self, query: str) -> str:
        self.query = query

        logger.info("Translating Query...")
        self._translate_query()
        logger.info("Loading Dataset...")
        self._load_dataset()
        logger.info("Retrieving Docs...")
        retrieved_docs = self._retrieval()
        logger.info("Retrieval successfully completed!")
        print(
            "\n\nSummary\n"
            + "=" * 20
            + f"\nUser query: {self.query}\n"
            + "=" * 20
            + f"\nRetrieved Docs:\n\n{retrieved_docs}"
        )

        return retrieved_docs


if __name__ == "__main__":
    logger.info("Starting RAG tool...")
    retriever = RAGTool()
    retriever.invoke("Who is Ada?")
