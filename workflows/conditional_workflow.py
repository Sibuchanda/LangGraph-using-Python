from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import TypedDict, Literal


load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# Defining Schema
class PaymentMethodSchema(BaseModel):
    payment : Literal["upi","card","cod"] = Field(description="Payment method of an order")


# Structured model
payment_model = llm.with_structured_output(PaymentMethodSchema)



class ReportState(TypedDict):
    user_message: str
    paymentmode : Literal["upi","card","cod"]
    response: str


def find_payment_method(state: ReportState) -> Literal["upi","card","cod"]:
    prompt = f"Find the payment method of the following message : {state["user_message"]}"
    res = payment_model.invoke(prompt).payment
    return {"paymentmode" : res}


def check_payment_method(state: ReportState) -> Literal["upi_payment","card_payment","cod_payment"]:
    if state["paymentmode"]=="upi":
        return "upi_payment"
    elif state["paymentmode"]=="card":
        return "card_payment"
    else:
        return "cod_payment"


def upi_payment(state: ReportState):
    prompt = f"Generate a short one line message for the choosed payment method : {state["paymentmode"]}"
    res = llm.invoke(prompt).content

    return {"response" : res}
def card_payment(state: ReportState):
    prompt = f"Generate a short one line message for the choosed payment method : {state["paymentmode"]}"
    res = llm.invoke(prompt).content

    return {"response" : res}
def cod_payment(state: ReportState):
    prompt = f"Generate a short one line message for the choosed payment method : {state["paymentmode"]}"
    res = llm.invoke(prompt).content

    return {"response" : res}



# Graph Creation

graph = StateGraph(ReportState)

graph.add_node("find_payment_method",find_payment_method)
graph.add_node("upi_payment",upi_payment)
graph.add_node("card_payment",card_payment)
graph.add_node("cod_payment",cod_payment)

# Adding graph edge
graph.add_edge(START,"find_payment_method")
graph.add_conditional_edges("find_payment_method",check_payment_method)

graph.add_edge("upi_payment",END)
graph.add_edge("card_payment",END)
graph.add_edge("cod_payment",END)


workflow = graph.compile()

message1 = "I want to pay using my Google Pay UPI ID."
message2 = "Can I enter my Visa credit card details to complete this order?"
message3 = "I prefer to pay in cash when the delivery agent arrives at my door."

initial_state = {"user_message" : message3}
res = workflow.invoke(initial_state)

print(f"User message : {res["user_message"]}")
print(f"Payment Mode : {res["paymentmode"]}")
print(f"Response : {res["response"]}")

