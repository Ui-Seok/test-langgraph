from typing import Dict, List, Optional, Union
from datetime import datetime
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
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
    
    # def __init__(self):
    #     self.messages: List[Dict] = []
    #     self.current_step: str = "START"
    #     self.mongo_client = MongoClient(MONGODB_URI)
    #     self.db = self.mongo_client[DB_NAME]
        
    # def add_message(self, message: Dict):
    #     self.messages.append(message)
        
# graph = StateGraph(ConversationState)

# def increment(state):
#     return {"counter": state["counter"] + 1}

# graph.add_node("increment", increment)

# graph.add_edge(START, "increment")
# graph.add_edge("increment", END)

# app = graph.compile()

# result = app.invoke({"counter": 2})

# print(result)

def process_user_input(state):
    messages = state.get("messages", [])
    chat_history = state.get("chat_history", [])
    
    if not messages:
        return state
    
    llm = ChatOllama(model="llama3.1")
    conversation_messages = []
    for hist in chat_history:
        if isinstance(hist, dict):
            if hist.get("role") == "user":
                conversation_messages.append(HumanMessage(content=hist["content"]))
            elif hist.get("role") == "assistant":
                conversation_messages.append(AIMessage(content=hist["content"]))
                
    current_message = messages if isinstance(messages, str) else messages[-1]
    conversation_messages.append(HumanMessage(content=current_message))
    # user_input = messages[-1] if isinstance(messages, list) else messages
        
    result = llm.invoke(conversation_messages)
    print(f"AI Agent Response: {result.content}")
    
    updated_history = chat_history + [
        {"role": "user", "content": current_message},
        {"role": "assistant", "content": result.content}
    ]
    
    return {
        "messages": result.content,
        "current_step": "processing",
        "chat_history": updated_history
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

def build_conversation_graph() -> StateGraph:
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
    # 대화 그래프 생성
    graph = build_conversation_graph()
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
        
        if initial_state.get("messages"):
            print("AI:", initial_state["messages"][-1]["content"])

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