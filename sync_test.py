from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pymongo import MongoClient


# MongoDB 설정
MONGO_DETAILS = "mongodb://localhost:27017"
client = MongoClient(MONGO_DETAILS)
db = client.test_db

model = ChatOllama(model="llama3.1")

@tool
def search_by_date(start_date: str = None, end_date: str = None, collection_name: str = None) -> str:
    """Search data by date range. Input format: YYYY-MM-DD, "COLLECTION NAME"

    Args:
        start_date (str, optional): start date by filter. Defaults to None.
        end_date (str, optional): end date by filter. Defaults to None.
        collection_name (str): collection name to find the data

    Returns:
        str: search result
    """
    query = {"detection_time": {"$gte": start_date, "$lte": end_date}}
    # print(collection_name)
    
    results = list(db[collection_name].find(query).limit(10))
    
    # 결과를 보기 좋게 포맷팅
    formatted_results = []
    for r in results:
        formatted_result = {
            "detection_time": r["detection_time"],
            "anomaly_type": {
                k: f"{v}%" for k, v in r["top_k_class"].items() if v > 0
            },
            "video_name": r["video_name"],
            "average_anomaly_score": f"{sum(r['anomaly_scores']) / len(r['anomaly_scores']):.4f}"
        }
        formatted_results.append(formatted_result)
    # print(formatted_results)
    return formatted_results

@tool
def add(a: int, b: int) -> int:
    """Add two integers together.

    Args:
        a (int): The first number to add
        b (int): The second number to add

    Returns:
        int: The sum of the two numbers
    """
    return a + b

tools = [search_by_date, add]

template = """
    Hello, I'm an AI assistant. I can help you with the following tasks:
    
    - **Search_by_date**: You can search for data by date range. Input format: YYYY-MM-DD, "COLLECTION NAME"
    - **Add**: You can add two numbers together.
    
    User Input:
"""

query = input(template)

messages = [HumanMessage(query)]

model_with_tools = model.bind_tools(tools)

# print(model_with_tools.invoke(query))

result = model_with_tools.invoke(query).tool_calls

print("#######################")
# print(f"Model Invoke Result:{result}")
# print("#######################")

messages.append(result)

for tool_call in result:
    selected_tool = {"add": add, "search_by_date": search_by_date}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)
    
llm_answer = messages[2].content
if llm_answer:
    print(f"LLM Answer: {llm_answer}")  # ToolMessage의 content를 출력 -> return 값과 동일
else:
    print(f"LLM Answer: No matching data found.")