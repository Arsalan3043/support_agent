from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
import operator

from core.config import settings
from core.models import AgentResponse
from tools.rag_tool import rag_tool


class AgentState(TypedDict):
    """State for the support agent graph"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    intent: str
    needs_rag: bool
    rag_results: str
    confidence: float
    answer: str
    needs_clarification: bool
    clarification_question: str
    iteration_count: int


class SupportAgent:
    """Intelligent support agent using LangGraph with ReAct pattern"""
    
    def __init__(self):
        self.llm = self._initialize_llm()
        self.graph = self._build_graph()
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration"""
        if settings.llm_provider == "anthropic":
            return ChatAnthropic(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                anthropic_api_key=settings.anthropic_api_key
            )
        else:
            return ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                openai_api_key=settings.openai_api_key
            )
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("decide_rag", self.decide_rag)
        workflow.add_node("retrieve_knowledge", self.retrieve_knowledge)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("validate_answer", self.validate_answer)
        workflow.add_node("ask_clarification", self.ask_clarification)
        
        # Define edges
        workflow.set_entry_point("classify_intent")
        
        workflow.add_edge("classify_intent", "decide_rag")
        
        workflow.add_conditional_edges(
            "decide_rag",
            self.route_after_rag_decision,
            {
                "retrieve": "retrieve_knowledge",
                "direct": "generate_answer"
            }
        )
        
        workflow.add_edge("retrieve_knowledge", "generate_answer")
        workflow.add_edge("generate_answer", "validate_answer")
        
        workflow.add_conditional_edges(
            "validate_answer",
            self.route_after_validation,
            {
                "clarify": "ask_clarification",
                "complete": END
            }
        )
        
        workflow.add_edge("ask_clarification", END)
        
        return workflow.compile()
    
    def classify_intent(self, state: AgentState) -> AgentState:
        """Classify user intent"""
        logger.info("Node: classify_intent")
        
        messages = [
            SystemMessage(content="""You are an intent classifier for a customer support system.
Classify the user's intent into one of these categories:
- greeting: General greetings, hello, hi
- question: Asking about products, services, policies
- complaint: Expressing dissatisfaction
- request: Requesting specific action
- feedback: Providing feedback
- other: Anything else

Respond with just the category name."""),
            HumanMessage(content=state["user_query"])
        ]
        
        response = self.llm.invoke(messages)
        intent = response.content.strip().lower()
        
        logger.info(f"Classified intent: {intent}")
        
        return {
            **state,
            "intent": intent,
            "messages": state.get("messages", []) + [HumanMessage(content=state["user_query"])]
        }
    
    def decide_rag(self, state: AgentState) -> AgentState:
        """Decide if RAG is needed"""
        logger.info("Node: decide_rag")
        
        # Simple heuristic: greetings don't need RAG
        needs_rag = state["intent"] not in ["greeting", "other"]
        
        # Could use LLM for more sophisticated decision
        if needs_rag:
            prompt = f"""Does answering this query require searching documentation?
Query: {state['user_query']}
Intent: {state['intent']}

Answer with just 'yes' or 'no'."""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            needs_rag = response.content.strip().lower() == "yes"
        
        logger.info(f"Needs RAG: {needs_rag}")
        
        return {**state, "needs_rag": needs_rag}
    
    def retrieve_knowledge(self, state: AgentState) -> AgentState:
        """Retrieve relevant knowledge using RAG"""
        logger.info("Node: retrieve_knowledge")
        
        rag_results = rag_tool._run(state["user_query"])
        
        logger.info(f"Retrieved knowledge (length: {len(rag_results)} chars)")
        
        return {**state, "rag_results": rag_results}
    
    def generate_answer(self, state: AgentState) -> AgentState:
        """Generate answer based on available information"""
        logger.info("Node: generate_answer")
        
        system_prompt = """You are a professional customer support agent. Your role is to:

1. Provide accurate, helpful answers based on the information available
2. Be empathetic and understanding
3. Acknowledge when you don't have enough information
4. Ask clarifying questions when needed
5. Always cite your sources when using knowledge base information
6. Keep responses concise but complete
7. Never hallucinate or make up information

Guidelines:
- If using knowledge base results, cite the source
- If information is uncertain, acknowledge it
- If you cannot answer, explain why and suggest next steps
- Be conversational and human-like"""
        
        user_prompt = f"""User Query: {state['user_query']}
Intent: {state['intent']}

"""
        
        if state.get("needs_rag") and state.get("rag_results"):
            user_prompt += f"""
Knowledge Base Results:
{state['rag_results']}

Based on the above information, provide a helpful answer. If the knowledge base doesn't contain enough information, be honest about it.
"""
        else:
            user_prompt += "\nProvide a direct, helpful response based on your general knowledge.\n"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        logger.info(f"Generated answer (length: {len(answer)} chars)")
        
        return {
            **state,
            "answer": answer,
            "messages": state.get("messages", []) + [AIMessage(content=answer)]
        }
    
    def validate_answer(self, state: AgentState) -> AgentState:
        """Validate the answer quality"""
        logger.info("Node: validate_answer")
        
        validation_prompt = f"""Review this support interaction:

User Query: {state['user_query']}
Agent Answer: {state['answer']}

Evaluate:
1. Is the answer complete and helpful?
2. Does it require clarification from the user?
3. Rate confidence (0-1)

Respond in this format:
COMPLETE: yes/no
CLARIFICATION_NEEDED: yes/no
CONFIDENCE: 0.X
CLARIFICATION_QUESTION: [if needed, ask a specific question]"""
        
        response = self.llm.invoke([HumanMessage(content=validation_prompt)])
        validation = response.content
        
        # Parse validation
        needs_clarification = "CLARIFICATION_NEEDED: yes" in validation
        confidence_line = [l for l in validation.split('\n') if 'CONFIDENCE:' in l]
        confidence = float(confidence_line[0].split(':')[1].strip()) if confidence_line else 0.8
        
        clarification_question = ""
        if needs_clarification:
            clarification_lines = [l for l in validation.split('\n') if 'CLARIFICATION_QUESTION:' in l]
            if clarification_lines:
                clarification_question = clarification_lines[0].split(':', 1)[1].strip()
        
        logger.info(f"Validation - Confidence: {confidence}, Needs clarification: {needs_clarification}")
        
        return {
            **state,
            "confidence": confidence,
            "needs_clarification": needs_clarification and confidence < settings.confidence_threshold,
            "clarification_question": clarification_question
        }
    
    def ask_clarification(self, state: AgentState) -> AgentState:
        """Ask clarifying question"""
        logger.info("Node: ask_clarification")
        
        clarification = state.get("clarification_question") or "Could you provide more details about your question?"
        
        return {
            **state,
            "answer": f"{state['answer']}\n\n{clarification}"
        }
    
    def route_after_rag_decision(self, state: AgentState) -> str:
        """Route based on RAG decision"""
        return "retrieve" if state.get("needs_rag") else "direct"
    
    def route_after_validation(self, state: AgentState) -> str:
        """Route based on validation"""
        if state.get("needs_clarification") and state["confidence"] < settings.confidence_threshold:
            return "clarify"
        return "complete"
    
    def process_message(self, user_message: str, session_id: str = "default") -> AgentResponse:
        """Process a user message and return response"""
        logger.info(f"Processing message: {user_message[:100]}...")
        
        initial_state = {
            "messages": [],
            "user_query": user_message,
            "intent": "",
            "needs_rag": False,
            "rag_results": "",
            "confidence": 1.0,
            "answer": "",
            "needs_clarification": False,
            "clarification_question": "",
            "iteration_count": 0
        }
        
        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            # Extract sources from RAG results if available
            sources = []
            if final_state.get("rag_results"):
                # Simple extraction - could be more sophisticated
                import re
                source_pattern = r'Source: (.+?)(?:\n|$)'
                sources = re.findall(source_pattern, final_state.get("rag_results", ""))
            
            return AgentResponse(
                content=final_state["answer"],
                confidence=final_state.get("confidence", 1.0),
                sources=sources,
                needs_clarification=final_state.get("needs_clarification", False),
                clarification_question=final_state.get("clarification_question"),
                metadata={
                    "intent": final_state.get("intent", "unknown"),
                    "used_rag": final_state.get("needs_rag", False)
                }
            )
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return AgentResponse(
                content="I apologize, but I encountered an error processing your request. Please try again or contact support.",
                confidence=0.0,
                sources=[],
                metadata={"error": str(e)}
            )


# Global agent instance
support_agent = SupportAgent()