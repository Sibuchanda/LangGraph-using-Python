from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import TypedDict, Literal
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str


def generate_joke(state: JokeState):
    prompt = f"Generate a joke on topic : {state["topic"]}"
    res = llm.invoke(prompt)

    return {"joke": res.content}


def generate_explanation(state: JokeState):
    prompt = f"Geneate explanation of the given joke : {state["joke"]} within 3 to 4 lines"
    res = llm.invoke(prompt)

    return {"explanation": res.content}


graph = StateGraph(JokeState)

graph.add_node("generate_joke",generate_joke)
graph.add_node("generate_explanation",generate_explanation)

graph.add_edge(START,"generate_joke")
graph.add_edge("generate_joke","generate_explanation")
graph.add_edge("generate_explanation",END)


# Creating in-memory checkpointer
checkpointer = InMemorySaver()

# Creating workflow
workflow = graph.compile(checkpointer=checkpointer)

# Creating Thread 
config1 = {"configurable" : {"thread_id" : "1"}}

res = workflow.invoke({"topic": "AI"}, config=config1)
# print(res["topic"])
# print(res["joke"])
# print(res["explanation"])

state = workflow.get_state(config1)
# print(state)

history = list(workflow.get_state_history(config1))
print(history)