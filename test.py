from langchain_ollama import ChatOllama
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain.schema import AgentFinish
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import json
import asyncio

# MongoDB 설정
MONGO_DETAILS = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_DETAILS)
db = client.test_db

# Ollama 모델 설정
llm = ChatOllama(model="llama3.1")

# MongoDB 검색을 위한 커스텀 도구들
@tool
async def search_by_date(start_date: str = None, end_date: str = None) -> str:
    """Search data by date range. Input format: YYYY-MM-DD"""
    query = {}
    # if start_date:
    #     # query["detection_time"] = {"$gte": datetime.fromisoformat(start_date)}
    #     start_date_obj = datetime.fromisoformat(start_date)
    # if end_date:
    #     end_date_obj = datetime.fromisoformat(end_date)
    #     # if "detection_time" in query:
    #     #     query["detection_time"]["$lte"] = datetime.fromisoformat(end_date)
    #     # else:
    #     #     query["detection_time"] = {"$lte": datetime.fromisoformat(end_date)}
    ddd = start_date.split(",")
    print(ddd)
    s_d = ddd[0]
    e_d = ddd[1].strip()
    query = {"detection_time": {"$gte": s_d, "$lte": e_d}}
    print(query)
    results = await db["anomaly_data"].find(query).to_list(length=10)
    print(results)
    return json.dumps([{k: str(v) for k, v in doc.items()} for doc in results], ensure_ascii=False)

@tool
async def search_by_attributes(upper: str = None, lower: str = None, hair: str = None) -> str:
    """사람의 특징으로 데이터를 검색합니다. 예: upper='빨간색', lower='청바지', hair='검정색'"""
    query = {}
    if upper:
        query["upper"] = upper
    if lower:
        query["lower"] = lower
    if hair:
        query["hair"] = hair
    
    results = await db.posts.find(query).to_list(length=10)
    return json.dumps([{k: str(v) for k, v in doc.items()} for doc in results], ensure_ascii=False)

# 도구 목록 생성
tools = [search_by_date, search_by_attributes]

# 프롬프트 템플릿 생성
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}"""

# 에이전트 생성
agent = create_react_agent(llm, tools, PromptTemplate.from_template(template))
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    max_iterations=3  # 최대 반복 횟수 제한
)

# 실행 함수
async def process_query(user_input: str):
    try:
        response = await agent_executor.ainvoke({"input": user_input})
        if isinstance(response, AgentFinish):
            return response.return_values.get("output")
        elif "output" in response:
            return response["output"]
        else:
            steps = response.get("intermediate_steps", [])
            if steps:
                last_step = steps[-1]
                if isinstance(last_step[1], str):
                    return last_step[1]
            return "검색 결과를 찾을 수 없습니다."
    except Exception as e:
        return f"Error occurred: {str(e)}"

# 사용 예시
if __name__ == "__main__":
    async def main():
        queries = [
            "Find data from October 1, 2024 to December 1, 2024"
        ]
        
        for query in queries:
            print(f"\n질문: {query}")
            result = await process_query(query)
            print(f"\n응답: {result}")
    
    asyncio.run(main())