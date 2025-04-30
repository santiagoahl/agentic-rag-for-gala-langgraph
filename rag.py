#from langchain.tools.base import BaseTool  # Superclass to build our agent
from langchain_community.llms import HuggingFaceHub  # Use LLM models stored in HF
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate  # Structure Prompts
from langchain.docstore.document import Document  # Format documents retrieved from the datasets
from langchain.retrievers import BM25Retriever  # Retrieve documents from dataset
from datasets import load_dataset  # Load Guests Dataset

from typing import Optional, Union, Any  # Typed programming
import os, getpass  # Get HF API token

# TODO: Migrate _get_var to utils script

# Get HF API token for model reference
def _get_var(var) -> None:
    if os.getenv(var):
        print("Variable already exists.")
    else:
        os.environ[var] = getpass.getpass(prompt=f"Type the value of {var}: ")

var = "HUGGINGFACEHUB_API_TOKEN"
_get_var(var)
hf_token = os.getenv(var)

class RAGTool():
    def __init__(self) -> None:
        # self.index = None
        self.query: Optional[str] = None
        self.query_translated: Optional[str] = None
        self.docs: list[Document] = None

        self.chat_template: Any = ChatPromptTemplate([
            SystemMessagePromptTemplate(prompt="You are a proffesional butler, specialized to organized the most successful Galas"),
            HumanMessagePromptTemplate(prompt="Find information about: {query}")
        ])
        
        self.llm: Any = HuggingFaceHub(
            repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            huggingfacehub_api_token=hf_token,
            parameters={
                "temperature": 0.1,
                "max_length": 100
            }
        )

    def _translate_query(self) -> None:
        # Step-Back
        prompt = f"Write step-back questions to optimize clarity: {self.query}"
        ai_response = self.llm.invoke(prompt=prompt) # TODO: Estructurar modelo antes de pasarlo
        self.query_translated = ai_response
        
    def _load_dataset(self) -> None:
        # Import guest dataset
        url = "agents-course/unit3-invitees"  # TODO: Create a Basemodel subclass to save all locations
        guest_dataset = load_dataset(path=url, name="Guest Dataset", split="train")
        
        self.docs = [
            Document(
                page_content="\n".join([
                        f'Name: {guest_info["name"]}',
                        f'Relation: {guest_info["relation"]}',
                        f'Description: {guest_info["description"]}',
                        f'Email: {guest_info["email"]}',
                    ]),
                metadata={"name": guest_info["name"]}
            )
            for guest_info in guest_dataset
        ] 

    def _embed_text(self) -> None:
        pass
    
    def _retrieval(self) -> None:
        # Verify is the docs are already loaded
        try: 
            _ = self.docs 
        except AttributeError:
            self._load_dataset()
            
        # Create retriever
        self.retriever = BM25Retriever.from_documents(documents=self.docs)
        
        # Similarity search
        result = self.retriever.get_relevant_documents(self.query_translated)
        result = result[:3]  # Get the 3 most relevant docs
        retrieved_docs = "\n\n".join([
                    doc.page_content
                    for doc in result
                ])
        
        if retrieved_docs:
            #self.retrieved_docs = retrieved_docs
            return retrieved_docs
        else:
            return "No information found about this Guest"       
    
    def _augment_prompt(self) -> None:   
        pass
    
    def __call__(self, query) -> None:
        self._translate_query()
        self._load_dataset()
        retrieved_docs = self._retrieval()
        return retrieved_docs

if __name__=="__main__":
    retriever = RAGTool()
    retriever("Who is Ada?")