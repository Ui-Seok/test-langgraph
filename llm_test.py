from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, AIMessage
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import json


# MongoDB 설정
MONGO_DETAILS = "mongodb://localhost:27017"
client = MongoClient(MONGO_DETAILS)
db = client.test_db

model = ChatOllama(model="llama3.1")

@tool
def search_by_date(start_date: str = None, end_date: str = None) -> str:
    """Search data by date range. Input format: YYYY-MM-DD

    Args:
        start_date (str, optional): start date by filter. Defaults to None.
        end_date (str, optional): end date by filter. Defaults to None.

    Returns:
        str: search result
    """
    query = {"detection_time": {"$gte": start_date, "$lte": end_date}}
    
    results = list(db["anomaly_data"].find(query).limit(10))
    
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

# 최종 결과 출력을 위한 함수 추가
def print_results(messages):
    for msg in messages:
        if isinstance(msg, ToolMessage):
            results = eval(msg.content)  # 문자열을 파이썬 객체로 변환
            print("\n검색 결과:")
            print("-" * 50)
            for idx, result in enumerate(results, 1):
                print(f"\n[이벤트 {idx}]")
                print(f"발생 시간: {result['detection_time']}")
                print("감지된 이상 행동:")
                for behavior, prob in result['anomaly_type'].items():
                    print(f"- {behavior}: {prob}")
                print(f"평균 이상 점수: {result['average_anomaly_score']}")
                print(f"비디오 파일: {result['video_name']}")
                print("-" * 30)

tools = [search_by_date, add]

template = """
    Hello, I'm an AI assistant. I can help you with the following tasks:
    
    - **Search_by_date**: You can search for data by date range. Input format: YYYY-MM-DD
    - **Add**: You can add two numbers together.
    
    User Input:
"""

query = input(template)
messages = [HumanMessage(query)]

model_with_tools = model.bind_tools(tools)

try:
    result = model_with_tools.invoke(query).tool_calls
    
    print("#######################")
    print(f"Model Invoke Result:{result}")
    print("#######################")
    
    # Tool이 호출되지 않은 경우 처리
    if not result:
        response = model.invoke(query)
        print(f"AI Response: {response.content}")
        exit()
        
    messages.append(result)
    
    for tool_call in result:
        try:
            # 존재하는 tool인지 확인
            tool_name = tool_call["name"].lower()
            if tool_name not in {"add", "search_by_date"}:
                raise KeyError(f"Tool '{tool_name}' not found")
                
            selected_tool = {"add": add, "search_by_date": search_by_date}[tool_name]
            tool_msg = selected_tool.invoke(tool_call)
            messages.append(tool_msg)
            
            # 결과 출력
            if isinstance(tool_msg, ToolMessage):
                try:
                    results = json.loads(tool_msg.content) if isinstance(tool_msg.content, str) else tool_msg.content
                    if not results:  # 빈 결과 처리
                        print("No matching data found for your query.")
                    else:
                        print_results([tool_msg])
                except json.JSONDecodeError:
                    print(f"Raw result: {tool_msg.content}")
            
        except KeyError as e:
            print(f"Error: {e}")
            print("Available tools are: 'search_by_date', 'add'")
        except Exception as e:
            print(f"An error occurred while processing the tool: {e}")

except Exception as e:
    print(f"An error occurred while processing your request: {e}")
    print("\nYou can try:")
    print("1. Search data by date: 'Find data from 2024-01-01 to 2024-12-31'")
    print("2. Add numbers: 'Add 5 and 3'")