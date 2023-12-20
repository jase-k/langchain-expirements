from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough

# Used for VectorStore
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser


load_dotenv()  # take environment variables from .env.
import os, requests

# Setting up VectorStore -> This vectorizes one file of graphql queries. 
# We need to handle this better from a schema perspective. Once a user is created we should save their schema file and vectorize it.
# Then based on the user's authentication we should be able to pull the correct vector files to use for the retriever. 
def get_docs():
    script_dir = os.path.dirname(__file__)  # get the directory of the current script file
    rel_path = "../training_data/graphql/create_task.graphql"
    abs_file_path = os.path.join(script_dir, rel_path)

    loader = TextLoader(abs_file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=0, separator="\n\n")
    docs = text_splitter.split_documents(documents)
    return docs

# OpenAi's method for embedding text
embeddings = OpenAIEmbeddings()

# Setting up Qdrant as our vectorstore. Location would need to change based on user's authentication.
qdrant = Qdrant.from_documents(
    docs=get_docs(),
    embeddings=embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="my_documents",
    force_recreate=True,
    verbose=True,
)

# Setting Qdrant as retriever
retriever = qdrant.as_retriever()

## Initializing OpenAI and ChatOpenAI models. -> These should be replaced by trained open sourced models. 
llm = OpenAI(openai_organization=os.environ['OPENAI_ORG'], openai_api_key=os.environ['OPENAI_API_KEY'], verbose=True)
chat_model = ChatOpenAI(openai_organization=os.environ['OPENAI_ORG'], openai_api_key=os.environ['OPENAI_API_KEY'], verbose=True)


gql_prompt = PromptTemplate.from_template("""Write a graphql query based on the following template: {graphql_doc}. 
Use the following user input to fill in the template: {user_input}.
""")

# Setting up the first chain to retrieve the best action based on the user's input. (note as of now, only one change is supported)
# It would be nice to add a 'need more context' prompt here. 
make_graphql_chain = (
    {"user_input": RunnablePassthrough(), "graphql_doc": retriever}
    | gql_prompt 
    | llm
    | StrOutputParser()
)

# TODO: Move to a different file. These are a little out of place. 
def authenticate(login, password, tenantid):
    url = "https://api.devii.io/auth"
    payload = {"login": login, "password": password, "tenantid": tenantid}
    response = requests.post(url, json=payload)
    print(response)
    return response.json().get('access_token')

def call_Devii(graphql: str):
    print(f"payload {graphql}")
    access_token = authenticate(os.environ['DEVII_LOGIN'], os.environ['DEVII_PASSWORD'], os.environ['DEVII_TENANTID'])
    payload = {"query": graphql}
    response = requests.post('https://api.devii.io/query', data=payload, headers={'Authorization': f'Bearer {access_token}'})
    print(response.json())
    return response.json()


# Prompt used to recap the transaction back to the user. 
response_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. You help the user manipulate their data based on the prompts. Your job is to respond to the user to tell them whether or not the data manipulation was successful.

Here is the user's response: {user_input}
Here is the graphql query: {graphql_doc}
Repsonse from the database: {devii_response}
Please report on whether or not the data manipulation was successful by summarizing what was asked and what was done.
""")

# Need to use PromptTemplate here because it's easier to use with the playground out of the box. Could handle differently in the future.
chain = (
    {"user_input": PromptTemplate.from_template("""{user_input}""")}
    | RunnablePassthrough.assign(
        dummy=lambda x: print(x),
        graphql_doc=lambda x: make_graphql_chain.invoke(x['user_input'].to_string())
        )
    | RunnablePassthrough.assign(
        devii_response=lambda x: call_Devii(x['graphql_doc'])
    )
    | response_prompt
    | chat_model
)



######## Uncomment to run main.py for debugging purposes.

# USER_QUERY = "add clean house to user 1 todo list"
# response = chain.invoke({"user_input" : USER_QUERY})
# print(response)
