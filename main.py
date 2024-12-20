from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from github import Github
from bs4 import BeautifulSoup
import requests

# Updated imports for new LangChain structure
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize GitHub client
github_token = os.getenv("GITHUB_TOKEN")
github_client = Github(github_token)

# Initialize OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    answer: str

def fetch_github_content():
    """Fetch only README content from GitHub repositories"""
    try:
        username = os.getenv("GITHUB_USERNAME")
        github_token = os.getenv("GITHUB_TOKEN")
        
        print(f"Debug: Fetching READMEs for username: {username}")
        
        if not username or not github_token:
            raise ValueError("GitHub credentials not properly configured")
            
        github_client = Github(github_token)
        user = github_client.get_user(username)
        
        readme_contents = []
        print(f"Debug: Starting to fetch READMEs")
        
        for repo in user.get_repos():
            try:
                # Skip forked repositories
                if not repo.fork:
                    try:
                        readme = repo.get_readme()
                        readme_content = readme.decoded_content.decode()
                        soup = BeautifulSoup(readme_content, 'html.parser')
                        cleaned_content = f"""
                        Repository: {repo.name}
                        README Content:
                        {soup.get_text()}
                        ---
                        """
                        readme_contents.append(cleaned_content)
                        print(f"Debug: Successfully fetched README for {repo.name}")
                    except Exception as e:
                        print(f"Debug: No README found for {repo.name}")
                        # Still include basic repo info even if no README
                        basic_info = f"""
                        Repository: {repo.name}
                        Description: {repo.description}
                        Language: {repo.language}
                        ---
                        """
                        readme_contents.append(basic_info)
            except Exception as e:
                print(f"Debug: Error processing repository {repo.name}: {str(e)}")
                continue
        
        joined_content = "\n".join(readme_contents)
        print(f"Debug: Total README content length: {len(joined_content)} characters")
        return joined_content
        
    except Exception as e:
        print(f"Error in fetch_github_content: {str(e)}")
        raise

def create_vectorstore():
    """Create a vector store from GitHub content"""
    # Fetch content
    raw_content = fetch_github_content()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_content)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore

# Initialize conversation chain
vectorstore = create_vectorstore()
llm = ChatOpenAI(
    model_name="gpt-4-turbo-preview",
    temperature=0.7,
    openai_api_key=openai_api_key
)

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    ("human", "Please provide an answer based on the context of my GitHub repositories and previous conversation. If the information isn't available in my repositories, please indicate that.")
])

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="history"
)

# Create the conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Convert chat history to the format expected by the chain
        if request.chat_history:
            for msg in request.chat_history:
                if 'question' in msg:
                    memory.chat_memory.add_message(HumanMessage(content=msg['question']))
                if 'answer' in msg:
                    memory.chat_memory.add_message(AIMessage(content=msg['answer']))

        # Get relevant documents from vector store
        docs = vectorstore.similarity_search(request.question)
        context = "\n".join(doc.page_content for doc in docs)
        
        # Get response from the chain
        response = conversation.predict(
            input=f"Context: {context}\nQuestion: {request.question}"
        )

        return ChatResponse(answer=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)