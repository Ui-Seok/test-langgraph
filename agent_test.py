from typing import Dict, List, Optional, Union
from datetime import datetime
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from typing import TypedDict
from IPython.display import Image, display
import json
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# MongoDB 연결 설정
MONGODB_URI = "mongodb://localhost:27017/"
DB_NAME = "test_db"
CHAT_DB_NAME = "chat_history_db"
CHAT_COLLECTION_NAME = "conversations"

# 사용 가능한 컬렉션 목록
AVAILABLE_COLLECTIONS = [
    "AOD_Info",
    "Patch_Bounding_Box",
    "Patch_Image_Path",
    "Patch_Info",
    "Pedestrian_Attributes",
    "anomaly_data",
    "video"
]

mongodb_client = MongoClient(MONGODB_URI)

checkpointer = MongoDBSaver(mongodb_client)

class ConversationState(TypedDict):
    messages: List[Dict]
    current_step: str
    chat_history: List[Dict]
'''Answer the following questions as best you can. You have access to the following tools:

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

Begin!

Question: {input}
Thought:{agent_scratchpad}'''
class ProcessUserInput:
    def __init__(self, llm_model, tools):
        self.llm_model = llm_model
        self.tools = tools

        # 프롬프트 템플릿 설정
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 MongoDB 데이터베이스 전문가입니다. 
            사용자의 질문에 따라 MongoDB에서 데이터를 검색하고 분석하는 것을 도와줍니다.

            다음 도구들을 사용할 수 있습니다:
            1. search_by_date: 날짜 범위로 데이터 검색 (형식: YYYY-MM-DD)
            2. list_available_collections: 사용 가능한 컬렉션 목록 조회

            사용자의 질문을 이해하고 적절한 도구를 선택하여 응답하세요.
            날짜 검색 시에는 반드시 YYYY-MM-DD 형식을 사용해야 합니다."""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(llm_model, tools, prompt)

        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    def __call__(self, state: ConversationState):
        messages = state.get("messages", [])
        chat_history = state.get("chat_history", [])
        
        # 현재 메시지 처리
        current_message = messages[-1] if isinstance(messages, list) else messages
        
        # 대화 이력을 포함한 프롬프트 생성
        full_prompt = f"""Previous conversation:
{chat_history}

Current question: {current_message}

Please help the user with their request. You can use the following tools:
- search_by_date: Search data by date range in MongoDB
- list_available_collections: List all available collections in MongoDB

Remember the context of the conversation when responding."""

        try:
            # Agent Executor를 통한 실행
            result = self.agent_executor.invoke({"input": full_prompt})
            response = result.get("output", "I couldn't process that request. Could you please rephrase?")
            
            # 대화 이력 업데이트
            chat_history.append({"role": "user", "content": current_message})
            chat_history.append({"role": "assistant", "content": response})
            
            return {
                "messages": messages,
                "current_step": "user_chat",
                "chat_history": chat_history,
                "response": response
            }
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            return {
                "messages": messages,
                "current_step": "user_chat",
                "chat_history": chat_history,
                "response": "I encountered an error. Could you please try again?"
            }


def user_chat(state: ConversationState) -> str:
    input_message = input("User: ")
    return {
        "messages": input_message,
        "current_step": "input",
        "chat_history": state.get("chat_history", [])
    }
    
# 종료 조건 설정
def should_end(state: ConversationState) -> bool:
    # print("CHECK SHOULD END")
    messages = state["messages"]
    if isinstance(messages, list):
        last_message = messages[-1].lower() if messages else ""
    else:
        last_message = str(messages).lower()
    return any(word in last_message for word in ["exit", "quit", "q"])

def build_conversation_graph(llm_model, tools) -> StateGraph:
    process_user_input = ProcessUserInput(llm_model, tools)

    workflow = StateGraph(ConversationState)
    memory = MemorySaver()
    
    # 대화 처리 노드 정의
    workflow.add_node("process_input", process_user_input)
    workflow.add_node("user_chat", user_chat)
    
    # 시작 노드 설정
    # workflow.set_entry_point("user_chat")
    
    # 엣지 연결
    workflow.add_edge(START, "user_chat")
    workflow.add_conditional_edges(
        "user_chat",
        should_end,
        {
            True: END,
            False: "process_input"
        }
    )
    workflow.add_edge("process_input", "user_chat")
    
    return workflow.compile(checkpointer=memory)

@tool
def list_available_collections() -> str:
    """List all available collections in the MongoDB database.

    Returns:
        str: A string containing the list of collections.
    """
    db = mongodb_client[DB_NAME]
    collection_list = []
    for collection_name in db.list_collection_names():
        collection_list.append(collection_name)
    # print(collection_list)
    return collection_list

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
    
    results = list(mongodb_client[DB_NAME][collection_name].find(query).limit(10))
    
    return results

# def save_conversation_history(state: ConversationState):
#     """대화가 종료되면 전체 대화 내용을 MongoDB에 저장합니다."""
#     try:
#         conversation_doc = {
#             "timestamp": datetime.utcnow(),
#             "messages": state.messages
#         }
#         state.db[COLLECTION_NAME].insert_one(conversation_doc)
#         print("대화 내용이 성공적으로 저장되었습니다.")
#     except Exception as e:
#         print(f"대화 내용 저장 중 오류 발생: {str(e)}")

def main():
    # LLM 설정
    llm_model = ChatOllama(model="llama3.1")
    
    tools = [list_available_collections, search_by_date]
    llm_model = llm_model.bind_tools(tools)

    # 대화 그래프 생성
    graph = build_conversation_graph(llm_model, tools)
    graph_image = graph.get_graph(xray=True).draw_mermaid_png()
    display(Image(graph_image))
    
    with open("main_graph.png", "wb") as f:
        f.write(graph_image)
    print("그래프 이미지 저장 완료")
    
    # 초기 상태 설정
    initial_state = {
        "messages": [],
        "current_step": START,
        "chat_history": []
    }
    config_t = {"configurable": {"thread_id": "user_1"}}
    
    print("대화를 시작합니다. ('q' 또는 'exit' 또는 'quit' 를 입력하면 대화가 종료됩니다)")
    
    while True:
        # user_input = input("User: ")
        result = graph.invoke(initial_state, config_t)
        
        # print(result)
        # print("#######################")
        print("End of Conversation")
        
        initial_state = result
        
        if should_end(initial_state):
            break
        
        # if state.current_step == END:
        #     save_conversation_history(state)
        #     break

if __name__ == "__main__":
    main()
    
# class MongoDBTools:
#     def __init__(self, mongo_client):
#         self.client = mongo_client
#         self.db = self.client[DB_NAME]
#         self.chat_db = self.client[CHAT_DB_NAME]
    
#     @tool
#     def list_available_collections(self) -> str:
#         """데이터베이스에서 사용 가능한 컬렉션 목록을 반환합니다."""
#         return f"사용 가능한 컬렉션: {', '.join(AVAILABLE_COLLECTIONS)}"
    
#     @tool
#     def query_collection(self, collection_name: str, query_params: str) -> str:
#         """특정 컬렉션에서 조건에 맞는 데이터를 조회합니다.
#         Args:
#             collection_name: 조회할 컬렉션 이름
#             query_params: JSON 형식의 쿼리 매개변수
#         """
#         try:
#             if collection_name not in AVAILABLE_COLLECTIONS:
#                 return f"오류: {collection_name}은(는) 유효하지 않은 컬렉션입니다."
            
#             # 문자열로 된 쿼리 매개변수를 딕셔너리로 변환
#             query_dict = json.loads(query_params)
            
#             # 쿼리 실행
#             result = list(self.db[collection_name].find(query_dict, {'_id': 0}).limit(5))
            
#             if not result:
#                 return f"{collection_name}에서 조건에 맞는 데이터를 찾을 수 없습니다."
                
#             return json.dumps(result, ensure_ascii=False, indent=2)
#         except Exception as e:
#             return f"쿼리 실행 중 오류 발생: {str(e)}"
    
#     @tool
#     def get_collection_schema(self, collection_name: str) -> str:
#         """특정 컬렉션의 스키마(필드 구조)를 반환합니다."""
#         try:
#             if collection_name not in AVAILABLE_COLLECTIONS:
#                 return f"오류: {collection_name}은(는) 유효하지 않은 컬렉션입니다."
            
#             # 컬렉션에서 첫 번째 문서를 가져와 필드 구조 확인
#             sample_doc = self.db[collection_name].find_one()
#             if not sample_doc:
#                 return f"{collection_name} 컬렉션이 비어있습니다."
            
#             # _id 필드 제외
#             if '_id' in sample_doc:
#                 del sample_doc['_id']
            
#             # 필드 이름과 타입 추출
#             schema = {key: type(value).__name__ for key, value in sample_doc.items()}
#             return json.dumps(schema, ensure_ascii=False, indent=2)
#         except Exception as e:
#             return f"스키마 조회 중 오류 발생: {str(e)}"
    
#     @tool
#     def aggregate_collection(self, collection_name: str, pipeline: str) -> str:
#         """특정 컬렉션에 대해 집계 파이프라인을 실행합니다."""
#         try:
#             if collection_name not in AVAILABLE_COLLECTIONS:
#                 return f"오류: {collection_name}은(는) 유효하지 않은 컬렉션입니다."
            
#             # 문자열로 된 파이프라인을 리스트로 변환
#             pipeline_list = json.loads(pipeline)
            
#             # 집계 실행
#             result = list(self.db[collection_name].aggregate(pipeline_list))
            
#             if not result:
#                 return f"{collection_name}에서 집계 결과가 없습니다."
                
#             return json.dumps(result, ensure_ascii=False, indent=2)
#         except Exception as e:
#             return f"집계 실행 중 오류 발생: {str(e)}"

# def setup_agent(tools):
#     # LLM 설정
#     llm = ChatOllama(model="llama3.1")
    
#     # 프롬프트 템플릿 설정
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """당신은 MongoDB 데이터베이스 전문가입니다. 
#         사용 가능한 컬렉션: AOD_Info, Patch_Bounding_Box, Patch_Image_Path, Patch_Info, 
#         Pedestrian_Attributes, anomaly_data, video

#         다음 도구들을 사용할 수 있습니다:
#         1. list_available_collections: 사용 가능한 컬렉션 목록 조회
#         2. query_collection: 특정 컬렉션에서 데이터 조회
#         3. get_collection_schema: 컬렉션의 스키마 구조 확인
#         4. aggregate_collection: 컬렉션에 대한 집계 연산 수행
        
#         사용자의 질문을 이해하고 적절한 도구를 선택하여 응답하세요.
#         쿼리나 집계 작업 시에는 먼저 스키마를 확인하는 것이 좋습니다."""),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ])
    
#     # 에이전트 생성
#     agent = create_react_agent(llm, tools, prompt)
#     return AgentExecutor(agent=agent, tools=tools, verbose=True)