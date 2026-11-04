from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from research_agent import run_research

class AgentState(TypedDict):
    query: str
    research: str
    analysis: str
    next_agent: str


def supervisor(state: AgentState):
    if not state.get("research"):
        return {"next_agent": "research"}
    if not state.get("analysis"):
        return {"next_agent": "analysis"}
    return {"next_agent": "finish"}

def research_agent(state: AgentState):
    print("Research agent executing...")

    result = run_research(state["query"])

    return {
        "research": result["summary"]
    }

def analyst_agent(state: AgentState):
    print("Analyst agent executing.....")
    return {
        "analysis":{
            "Preliminary implication: organizations "
            "should evaluate automation opportunities. "
            "Evidence is still required."
        }
    }

def route(state: AgentState) -> Literal[
    "research",
    "analysis",
    "finish"
]:
    return state["next_agent"]

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("research", research_agent)
workflow.add_node("analysis", analyst_agent)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    route,
    {
        "research": "research",
        "analysis": "analysis",
        "finish": END
    }
)
workflow.add_edge("research", "supervisor")
workflow.add_edge("analysis", "supervisor")

app = workflow.compile()

result = app.invoke({
    "query": "Analyze AI trends in banking",
    "research": "",
    "analysis": "",
    "next_agent": ""
})

print(result)
