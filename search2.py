import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import AstraDB
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import json 
import requests

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_ENDPOINT")
ASTRA_DB_COLLECTION = os.environ.get("ASTRA_DB_COLLECTION")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CLICKUP_KEY = os.environ.get("CLICKUP_KEY")

embedding = OpenAIEmbeddings()

vstore = AstraDB(
    embedding=embedding,
    collection_name=ASTRA_DB_COLLECTION,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
)

# get clickup info
# The task_id will be obtained from the webhook event.
task_id = "868740y0h"
base_url = "https://api.clickup.com/api/v2/task/"
full_url = f"{base_url}{task_id}"

headers = {
    "Authorization": f"{CLICKUP_KEY}",
    "Content-Type": "application/json"
}

response = requests.get(full_url, headers=headers)

task_name=''
task_description=''



if response.status_code == 200:
    data = response.json()
    task_name = data['name']
    task_description = data['description']
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")



retriever = vstore.as_retriever()


search_template = """
You are a search bot that performs similarity searches on tasks based on various criteria such as client email, people names, task description, task name, and school names. 
Your goal is to find tasks that closely match these criteria and provide concise and relevant answers to user queries.

Utilize the provided context as the foundation for your search. Take 
into consideration client emails, names of individuals involved, task descriptions, and mentions of school names to identify and return the most relevant similar tasks.

Your responses should be clear and on point, focusing on tasks that align closely with the specified criteria.

And add "THIS UPDATE ORIGINATES FROM SUPPORT-AI CHATBOT" to the beggining and make it bold. 

CONTEXT:
{context}

TASK_INPUT: {task_input}

YOUR ANSWER:"""

search_prompt = ChatPromptTemplate.from_template(search_template)
llm = ChatOpenAI()

chain = (
    {"context": retriever, "task_input": RunnablePassthrough()}
    | search_prompt
    | llm
    | StrOutputParser()
)



query = f"task name: {task_name}, task description: {task_description}"


results = chain.invoke(query)
print(results)

updateTaskCommentURL = f"https://api.clickup.com/api/v2/task/{task_id}/comment"

commnet_data = {
  "comment_text": results,
  "assignee": 0,
  "notify_all": False
}


# Serialize the dictionary to JSON format
comment_json = json.dumps(commnet_data)
print(comment_json)
requests.post(updateTaskCommentURL, headers=headers, data=comment_json)

# Check if the request was successful
if response.status_code == 200:
    print("Comment added successfully.")
else:
    print(f"Failed to add comment. Status code: {response.status_code}")