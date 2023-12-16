from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage, FunctionMessage
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Used for VectorStore
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant

# Used for custom tools 
from langchain.tools import tool


load_dotenv()  # take environment variables from .env.
import os, requests




## Example of using Qdrant
script_dir = os.path.dirname(__file__)  # get the directory of the current script
rel_path = "../training_data/graphql/create_task.graphql"
abs_file_path = os.path.join(script_dir, rel_path)

loader = TextLoader(abs_file_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=0, separator="\n\n")
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="my_documents",
    force_recreate=True,
)
# Setting Qdrant as retriever
retriever = qdrant.as_retriever()

# query = "remove go shopping from my todo list"
# found_docs = qdrant.similarity_search_with_score(USER_QUERY)

# for page_content, score in found_docs:
#     print(f"Found: {score} : {page_content.page_content}")

# db_call = found_docs[0][0].page_content

from datetime import date
# print(date.today())

def get_graphql_template(path: str):
    script_dir = os.path.dirname(__file__)  # get the directory of the current script
    rel_path = path
    abs_file_path = os.path.join(script_dir, rel_path)

    with open(abs_file_path, 'r') as file:
        query = file.read()

    return query

# docs_for_db_call = get_graphql_template("../training_data/graphql/create_task.graphql")

## Example of using OpenAI LLM
llm = OpenAI(openai_organization=os.environ['OPENAI_ORG'], openai_api_key=os.environ['OPENAI_API_KEY'])
chat_model = ChatOpenAI(openai_organization=os.environ['OPENAI_ORG'], openai_api_key=os.environ['OPENAI_API_KEY'])


gql_prompt = PromptTemplate.from_template("""Write a graphql query based on the following template: {graphql_doc}. 
Use the following user input to fill in the template: {user_input}.
""")

# text = gql_prompt.format(graphql_doc=, user_input=USER_QUERY)

# response = llm.invoke(text)
# print(response)



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



make_graphql_chain = (
    {"graphql_doc": retriever, "user_input": RunnablePassthrough()}
    | gql_prompt #Needs graphql_doc and user_input
    | llm
)

# devii_res = make_graphql_chain.invoke(USER_QUERY)
# print(devii_res)

system_template = "You are a helpful assistant. You help the user manipulate their data based on the prompts. Your job is to respond to the user to tell them whether or not the data manipulation was successful."

human_template = """Here is the user's response: {user_input}
Repsonse from the database: {devii_response}
Please report on whether or not the data manipulation was successful by summarizing what was asked and what was done.
"""

tool_template = "Repsonse from the database: {devii_response}"

response_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_template),
    HumanMessage(content=human_template),
    # FunctionMessage(name="call_db", content=tool_template),
    # HumanMessage(content="Please report on whether or not the data manipulation was successful by summarizing what was asked and what was done."),
])


chain = (
    {"user_input": RunnablePassthrough()}
    | RunnablePassthrough.assign(
        devii_response=lambda x: call_Devii(make_graphql_chain.invoke( x['user_input']))
    )
    | response_prompt
    | chat_model
)



USER_QUERY = "add clean house to user 1 todo list"

response = chain.invoke(USER_QUERY)
print(response)
