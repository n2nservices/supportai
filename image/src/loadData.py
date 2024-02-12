import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import AstraDB
from langchain_community.document_loaders.csv_loader import CSVLoader
from datetime import datetime, timedelta
import pandas as pd
import aiohttp
import asyncio

load_dotenv()

def handler(event, context):
    

    CLICKUP_KEY = os.environ.get("CLICKUP_KEY")
    ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_ENDPOINT")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    # PROD environment
    ASTRA_DB_PROD_KEYSPACE = os.environ.get("ASTRA_DB_PROD_KEYSPACE")
    ASTRA_DB_PROD_CLICKUP_COLLECTION = os.environ.get("ASTRA_DB_PROD_CLICKUP_COLLECTION")

    # DEV Environenet
    # ASTRA_DB_DEV_KEYSPACE = os.environ.get("ASTRA_DB_DEV_KEYSPACE")
    # ASTRA_DB_DEV_CLICKUP_COLLECTION = os.environ.get("ASTRA_DB_DEV_CLICKUP_COLLECTION")

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
   

    async def get_tasks(session, list_id):
        tasks = []
        pageNumber = 0
        nextPage = True

        while nextPage:
            url = f"https://api.clickup.com/api/v2/list/{list_id}/task?page={pageNumber}&date_created_gt={two_days_ago_time_ms}&include_closed=true"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    tasks.extend(data['tasks'])
                    nextPage = not data['last_page']
                    pageNumber += 1
                else:
                    print(f"Failed to fetch tasks: {response.status}")
                    break
        return tasks
    

    async def get_comments(session, task_id):
        comments = []
        url = f"https://api.clickup.com/api/v2/task/{task_id}/comment"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                comments.extend(data['comments'])
            else:
                print(f"Failed to fetch comments for task {task_id}: {response.status}")
        return comments
    
    async def fetch_all_tasks_and_comments():
        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = []
            for list_id in env_list_ids:
                tasks.extend(await get_tasks(session, list_id))

            comments_tasks = []
            for task in tasks:
                comments_task = asyncio.create_task(get_comments(session, task['id']))
                comments_tasks.append(comments_task)

            comments = await asyncio.gather(*comments_tasks)
            for task, task_comments in zip(tasks, comments):
                task['comments'] = task_comments

            return tasks

    # Running the main async function
    if __name__ == "__main__":
        tasks = asyncio.run(fetch_all_tasks_and_comments())

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
            "Description": task.get('description'),
        }

        # Flatten comments
        comments = " | ".join([comment['comment_text'] for comment in task.get('comments', [])])
        task_details["Comments"] = comments

        return task_details
    
    flattened_tasks = [flatten_task(task) for task in tasks]
    df = pd.DataFrame(flattened_tasks)
    df.to_csv(csv_file_path, index=False, encoding='utf-8')
    print(f"{csv_file_path} was successfully created.")


    # Load, Split, and save data to DB
    loader = CSVLoader(file_path=csv_file_path, encoding='utf-8')
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(data)

    embedding = OpenAIEmbeddings()

    vstore = AstraDB(
        embedding=embedding,
        namespace=ASTRA_DB_PROD_KEYSPACE,
        collection_name=ASTRA_DB_PROD_CLICKUP_COLLECTION,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
    )

    # vstore.delete_collection()
    supportTickets = vstore.add_documents(splits)


