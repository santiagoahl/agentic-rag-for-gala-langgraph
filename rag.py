from langchain_core import Tool
from typing import Optional, Union
from langchain_community import HuggingFaceHub
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

import os, getpass

# TODO: Migrate to utils script
def _get_var(var) -> None:
    if os.getenv(var):
        print("Variable already exists.")
    else:
        os.environ[var] = getpass.getpass(prompt=f"Type the value of {var}: ")

var = "HUGGINGFACEHUB_API_TOKEN"
_get_var(var)
hf_token = os.getenv(var)

class RAGTool(Tool):
    def __init__(self, query: Union[str, dict]) -> None:
        self.index = None
        self.query = query
        self.query_translated: Optional[str] = None
        self.chat_template = ChatPromptTemplate([
            SystemMessagePromptTemplate(prompt="You are a proffesional butler, specialized to organized the most successful Galas"),
            HumanMessagePromptTemplate(prompt="Find information about: {query}")
        ])
        
        self.llm = HuggingFaceHub(
            repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            huggingfacehub_api_token=hf_token,
            parameters={
                "temperature": 0.1,
                "max_length": 100
            }
        )

    def _query_translation(self):
        # Step-Back
        prompt = f"Write step-back questions to avoid clarity: {self.query}"
        ai_response = self.llm.invoke(prompt=prompt, chat_template=self.chat_template) # TODO: Estructurar modelo antes de pasarlo
        self.query_translated = ai_response

    def _embed_text(self):
        pass
    
    def _retrieval(self):
        pass
    
    def _augment_prompt(self):   
        pass  
    
    def __call__(self):
        #structured_query = 
        #retrieved_docs = 
        #augmented_prompt = 
        #return augmented_prompt
        pass

def main() -> None:
    pass

if __name__=="__main__":
    main()