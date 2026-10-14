from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import TypedDict, Literal


load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# Defining Schema
class ReviewSchema(BaseModel):
    is_approved: bool = Field("True if the artical satisfied all conditions else False")
    feedback: str = Field("Actionable feedback on how to improve the artical")


# Structured model
review_model = llm.with_structured_output(ReviewSchema)



class ArticalState(TypedDict):
    topic: str  # User input message
    article : str # AI generated atrical
    is_approved: bool
    feedback: str
    count: int


def write_article(state: ArticalState):
    revision_count = state["count"] + 1
    get_feedback = state["feedback"]

    if(len(get_feedback) > 0):
        prompt = f"Rewrite the following topic {state["topic"]} to fix the issues : {get_feedback}. Current article is : {state["article"]}"
    else:
        prompt = f"Write an article on topic : {state["topic"]} including two bullet points"

    res = llm.invoke(prompt)
    return {"article": res.content, "count": revision_count}
        


def review_article(state: ArticalState):
    prompt = f"""Review the article :{state["article"]} based on the criteria : 1. Must contain at least 2 bullet points.
    2. Must be concise (under 80 words).
    3. Must end with a key takeaway or conclusion sentence.
    based on the topic : {state["topic"]}
    """

    review = review_model.invoke(prompt)
    return {"is_approved" : review.is_approved, "feedback" : review.feedback}



def check_approval(state: ArticalState) -> Literal["write_artical", "__end__"]:
    if state["is_approved"] or state["count"] >=5 :
        return END
    else:
        return "write_article"


graph = StateGraph(ArticalState)

graph.add_node("write_article", write_article)
graph.add_node("review_article", review_article)

graph.add_edge(START,"write_article")
graph.add_edge("write_article","review_article")

# Cyclic conditional check
graph.add_conditional_edges(
    "review_article",
    check_approval,{
        "write_article" : "write_article",
        END : END
    }
)

workflow = graph.compile()

# 6. Execution
initial_state = {
    "topic": "Why Developers Should Learn LangGraph in 2026",
    "article": "",
    "is_approved": False,
    "feedback": "",
    "count": 0
}

result = workflow.invoke(initial_state)


print(result["article"])
print("\n=== EXECUTION METRICS ===")
print(f"Approved: {result['is_approved']}")
print(f"Total Revisions/Loops: {result['count']}")



