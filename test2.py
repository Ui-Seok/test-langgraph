from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent

# template = """Question: {question}

# Answer: Let's think step by step."""

# prompt = ChatPromptTemplate.from_template(template)

model = ChatOllama(model="llama3.1")

# chain = prompt | model

# response = chain.invoke({"question": "What is LangChain?"})

# print(response)
print("#######################")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together. For example, to multiply 2 and 3, input should be: {"a": 2, "b": 3}

    Args:
        a: The first number to multiply
        b: The second number to multiply

    Returns:
        The product of the two numbers
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two integers together. For example, to add 2 and 3, input should be: {"a": 2, "b": 3}

    Args:
        a: The first number to add
        b: The second number to add

    Returns:
        The sum of the two numbers
    """
    return a + b

# multiply.invoke({"a": 2, "b": 3})

# print(f"multiply name: {multiply.name}", sep="\n")
# print(f"multiply description: {multiply.description}", sep="\n")
# print(f"multiply args: {multiply.args}", sep="\n")

# print("#######################")

query = "What is 23 multiplied by 3?"

messages = [HumanMessage(query)]

tool_list = [multiply, add]

model_with_tools = model.bind_tools(tool_list)

result = model_with_tools.invoke(query).tool_calls

print(result, sep="\n")

messages.append(result)

for tool_call in result:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)
    
print(messages)