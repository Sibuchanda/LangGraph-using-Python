from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import TypedDict, Literal
from langgraph.checkpoint.memory import InMemorySaver
import time

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

class CrashState(TypedDict):
    input: str
    step1: str
    step2: str


def step_1(state: CrashState) -> CrashState:
    print("Step 1 executed")
    return {"step1": "done", "input": state["input"]}

def step_2(state: CrashState) -> CrashState:
    print("Step 2 hanging... now manually interrupt from the notebook toolbar (STOP button)")
    time.sleep(1000)  # created manually long running point
    return {"step2": "done"}

def step_3(state: CrashState) -> CrashState:
    print("Step 3 executed")
    return {"done": True}


graph = StateGraph(CrashState)
graph.add_node("step_1", step_1)
graph.add_node("step_2", step_2)
graph.add_node("step_3", step_3)

graph.set_entry_point("step_1")
graph.add_edge("step_1", "step_2")
graph.add_edge("step_2", "step_3")
graph.add_edge("step_3", END)

checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)
thread1 = {"configurable" : {"thread_id" : "1"}}


# try:
#     print("Running graph: Please manually interrupt during Step 2...")
#     workflow.invoke({"input": "start"}, config={"configurable": {"thread_id": 'thread1'}})
# except KeyboardInterrupt:
#     print("Kernel manually interrupted (crash simulated).")



# 6. Re-run to show fault-tolerant resume
print("\n Re-running the graph to demonstrate fault tolerance...")
final_state = workflow.invoke(None, config={"configurable": {"thread_id": 'thread1'}})
print("\nFinal State:", final_state)