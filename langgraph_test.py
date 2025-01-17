from typing import Optional, TypedDict, Literal, Annotated, List
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
import operator

from typing_extensions import TypedDict

class MainState(TypedDict):
   messages: Annotated[List[AnyMessage], operator.add]  # 메시지 히스토리
   context: str  # 컨텍스트 정보
   subgraph_result: Optional[str]
   processing_status: str

class SubGraphState(TypedDict):
    """서브그래프의 상태를 정의하는 클래스"""
    messages: Annotated[List[AnyMessage], operator.add]  # 메시지 히스토리
    context: str  # 컨텍스트 정보

def preprocessing(state: MainState) -> MainState:
    """데이터 전처리를 수행하는 노드"""
    return {
        "context": f"Context from: {state['messages'][-1].content}",
        "processing_status": "preprocessing_complete"
    }

def postprocessing(state: MainState) -> MainState:
    """서브그래프 실행 결과를 후처리하는 노드"""
    context = state.get("context", "")
    return {
        "subgraph_result": f"Final result based on context: {context}",
        "processing_status": "complete"
    }

def route_next(state: MainState) -> Literal["postprocessing", "reprocess"]:
    """다음 단계를 결정하는 라우터"""
    if state["processing_status"] == "preprocessing_complete":
        return "postprocessing"
    return "reprocess"
    
def process_node(state: SubGraphState):
    """입력 메시지를 처리하는 노드"""
    current_message = state["messages"][-1]
    # 메시지 처리 로직
    processed_result = f"Processed: {current_message.content}"
    return {"context": processed_result}

def respond_node(state: SubGraphState):
    """응답을 생성하는 노드"""
    context = state["context"]
    # 응답 생성 로직
    response = AIMessage(content=f"Response based on: {context}")
    return {"messages": [response]}

# 메인 그래프 구성
main_graph = StateGraph(MainState)

# 서브그래프 인스턴스 생성
subgraph = StateGraph(SubGraphState)

# 서브그래프 노드 추가
subgraph.add_node("process", process_node)
subgraph.add_node("respond", respond_node)

# 서브그래프 엣지 추가
subgraph.add_edge(START, "process")      # 시작 -> 처리
subgraph.add_edge("process", "respond")  # 처리 -> 응답
subgraph.add_edge("respond", END)        # 응답 -> 종료

# 메인 그래프 노드 추가
main_graph.add_node("preprocessing", preprocessing)
main_graph.add_node("subgraph", subgraph.compile())  # 기존 서브그래프
main_graph.add_node("postprocessing", postprocessing)

# 메인 그래프 엣지 추가
main_graph.add_edge(START, "preprocessing")
main_graph.add_edge("preprocessing", "subgraph")
main_graph.add_conditional_edges(
    "subgraph", 
    route_next,
    {
        "postprocessing": "postprocessing",
        "reprocess": "preprocessing"
    }
)
main_graph.add_edge("postprocessing", END)

# 메인 그래프 컴파일
compiled_maingraph = main_graph.compile()

# 서브 그래프 컴파일
compiled_subgraph = subgraph.compile()

# 메인 그래프 시각화
maingraph_image = compiled_maingraph.get_graph(xray=True).draw_mermaid_png()
display(Image(maingraph_image))

# 메인 그래프 이미지 파일로 저장
with open("main_graph.png", "wb") as f:
    f.write(maingraph_image)
print("그래프 이미지 저장 완료")

initial_state = {
   "messages": [HumanMessage(content="Hello!")],  
   "processing_status": "started"
}

try:
   result = compiled_maingraph.invoke(initial_state)
   print("실행 결과:")
   print(f"처리 상태: {result['processing_status']}")
   print(f"최종 결과: {result['subgraph_result']}")
except Exception as e:
   print(f"그래프 실행 실패: {e}")
