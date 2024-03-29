import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import AstraDB
from dotenv import load_dotenv
import json 
import requests
# import boto3

# # Initialize a DynamoDB client
# dynamodb = boto3.resource('dynamodb')

load_dotenv()
def lambda_handler(event, context):

    # The task_id will be obtained from the webhook event.
    task_id = "8687a4g2v"

    # body = json.loads(event['body'])
    # task_id = body['task_id']
    
    # table = dynamodb.Table('supportai')
    
    # response = table.get_item(
    #     Key={
    #         'task_id': task_id
    #     }
    # )
    
    # # Check if item exists
    # if 'Item' in response:
    #     # Item exists, stop the function
    #     print('task_Id already exists.')
    #     return {
    #         'statusCode': 400,
    #         'body': json.dumps('task_id already exists')
    #     }
    # else:
    #     print("Task_id added successful")
    #     # Item does not exist, insert the new item
    #     table.put_item(
    #         Item={
    #             'task_id': task_id
    #             # Add other attributes here if necessary
    #         }
    #     )

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





    query = f"task name: {task_name}, task description: {task_description}"
    # Set a similarity threshold (adjust as needed)
    # similarity_threshold = 0.7

    # Perform the similarity search with the threshold
    results = vstore.similarity_search(query, k=4)
    print(results)

    concatenated_contents = ""
    separator = "\n" + "-"*60 + "\n"  # 50 dashes as a separator

    for index, doc in enumerate(results):
        page_content = doc.page_content.replace('\xa0', ' ')
        concatenated_contents += f"Entry {index + 1}:\n{page_content}{separator}"

    concatenated_contents = concatenated_contents.strip()


    updateTaskCommentURL = f"https://api.clickup.com/api/v2/task/{task_id}/comment"

    commnet_data = {
    "comment_text": concatenated_contents,
    "assignee": 0,
    "notify_all": False
    }

    json_data = json.dumps(commnet_data)
    print(json_data)

    postResponse = requests.post(updateTaskCommentURL, headers=headers, data=json_data)

    # Check if the request was successful
    if postResponse.status_code == 200:
        print("Comment added successfully.")
    else:
        print(f"Failed to add comment. Status code: {postResponse.text}")


lambda_handler("", "")