import os
from dotenv import load_dotenv
import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import AstraDB
from langchain_community.document_loaders.csv_loader import CSVLoader
from datetime import datetime, timedelta
import pandas as pd

load_dotenv()

def handler(event, context):
    

    CLICKUP_KEY = os.environ.get("CLICKUP_KEY")
    ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_ENDPOINT")
    ASTRA_DB_COLLECTION = os.environ.get("ASTRA_DB_COLLECTION")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    # QA and PROD clickup Env Support Ids
    env_list_ids = [134110690, 181779213]

    headers = {
    'Authorization': CLICKUP_KEY,
    'Content-Type': 'application/json'
    }

    # Calculate last 7 days in milliseconds
    today = datetime.now()
    two_days_ago = today - timedelta(days=2)
    two_days_ago_time_ms = int(two_days_ago.timestamp() * 1000)
   

    def get_tasks(list_id):
        tasks = []
        pageNumber=0
        nextPage = True

        while nextPage:
            url = f"https://api.clickup.com/api/v2/list/{list_id}/task?page={pageNumber}&date_created_gt={two_days_ago_time_ms}&include_closed=true"
            
            response = requests.get(url, headers=headers)
            lastPage = response.json()['last_page']
        

            if response.status_code == 200:
                data = response.json()
                tasks.extend(data['tasks'])

                # Handle pagination
                if lastPage == False:
                    nextPage = True 
                    pageNumber = pageNumber+1
                else: 
                    nextPage = False
            else:
                print(f"Failed to fetch tasks: {response.status_code}")
                break
        return tasks
    

    def get_comments(task_id):
        comments = []
        
        url = f"https://api.clickup.com/api/v2/task/{task_id}/comment"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            comments.extend(data['comments'])
        else:
            print(f"Failed to fetch comments for task {task_id}: {response.status_code}")
        return comments
    
    tasks = []
    for list_id in env_list_ids: 
        tasks.extend(get_tasks(list_id))

    for task in tasks:
        task['comments'] = get_comments(task['id'])

    # Convert response to CSV
    csv_file_path = '/tmp/temp_tasks_details.csv'

    if os.path.exists(csv_file_path):
        # Delete the file
        os.remove(csv_file_path)
        print(f"{csv_file_path} has been deleted.")
    else:
        print(f"{csv_file_path} does not exist.")

    
    def flatten_task(task):
        # Basic task details
        task_details = {
            "Task ID": task.get('id'),
            "Custom ID": task.get('custom_id'),
            "Name": task.get('name'),
            "Status": task['status']['status'] if 'status' in task else '',
            "Priority": task.get('priority'),
            "Date Created": task.get('date_created'),
            "Date Updated": task.get('date_updated'),
            "Creator": task['creator']['username'] if 'creator' in task else '',
            "Assignees": ", ".join([assignee['username'] for assignee in task.get('assignees', [])]),
            "Watchers": ", ".join([watcher['username'] for watcher in task.get('watchers', [])]),
            "Description": task.get('description'),
        }

        # Flatten custom fields
        for field in task.get('custom_fields', []):
            task_details[field['name']] = field.get('value')

        # Flatten comments
        comments = " | ".join([comment['comment_text'] for comment in task.get('comments', [])])
        task_details["Comments"] = comments

        return task_details
    
    flattened_tasks = [flatten_task(task) for task in tasks]
    df = pd.DataFrame(flattened_tasks)
    df.to_csv(csv_file_path, index=False, encoding='utf-8')
    print(f"{csv_file_path} was successfully created.")


    # Load, Split, and save data to DB
    loader = CSVLoader(file_path=csv_file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(data)

    embedding = OpenAIEmbeddings()

    vstore = AstraDB(
        embedding=embedding,
        collection_name=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
    )

    # vstore.delete_collection()
    supportTickets = vstore.add_documents(splits)


