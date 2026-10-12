from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import TypedDict, Literal


load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# Defining Schema
class UrgencySchema(BaseModel):
    urgency : Literal["high","medium","low"] = Field(description="Urgency of the mail")

class DepartmentSchema(BaseModel):
    department : Literal["finance","shipping","technical"] = Field(description="Related department of the mail")

# Structured model
urgency_model = llm.with_structured_output(UrgencySchema)
department_model = llm.with_structured_output(DepartmentSchema)



class ReportState(TypedDict):
    user_message: str
    urgency : Literal["high","medium","low"]
    department : Literal["finance","shipping","technical"]
    summery: str


def find_urgency(state: ReportState) -> Literal["high","medium","low"]:
    prompt = f"Find the urgency of the following user message : {state["user_message"]}"
    res = urgency_model.invoke(prompt).urgency
    return {"urgency" : res}


def find_department(state: ReportState):
    prompt = f"What could be the suitable department based on the following message : {state["user_message"]}"
    res = department_model.invoke(prompt).department

    return {"department" : res}


def generate_summery(state: ReportState):
    prompt = f"Generate a short replay mail to department : {state["department"]} baed on the urgency level : {state["urgency"]}"

    res = llm.invoke(prompt)
    return {"summery": res.content}


# Graph Creation

graph = StateGraph(ReportState)

graph.add_node("find_urgency",find_urgency)
graph.add_node("find_department",find_department)
graph.add_node("generate_summery",generate_summery)

# Adding graph edge
graph.add_edge(START,"find_urgency")
graph.add_edge(START,"find_department")

graph.add_edge("find_urgency","generate_summery")
graph.add_edge("find_department","generate_summery")

graph.add_edge("generate_summery",END)


workflow = graph.compile()



mail1 = """Hello,
I was checking my credit card statement today and noticed an unexpected charge from your service. 
Could someone please look into this and let me know why I was billed twice? 
I'd like to get this resolved as soon as possible."""

mail2 = """Hi support team,
My tracking link says my package was delivered yesterday, but nothing has arrived at my doorstep yet. 
Could you check with the carrier to see where it was left? 
Thank you for your help."""

mail3 = """Hey team,
Ever since the latest app update, I keep getting logged out automatically every time I open a new tab. 
Is this a known issue, or is there a setting I need to change on my end? 
Let me know if you need any error logs."""


initial_state = {"user_message" : mail2}
res = workflow.invoke(initial_state)

print(f"Urgency : {res["urgency"]}")
print(f"Department : {res["department"]}")
print(f"Summery Mail : {res["summery"]} \n")
