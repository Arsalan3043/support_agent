"""
Example script to test the support agent directly
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.support_agent import support_agent
from utils.load_knowledge import load_sample_knowledge_base
from storage.vector_store import vector_store
from loguru import logger
import time


def test_agent():
    """Test the agent with various queries"""
    
    # First, ensure we have some knowledge
    stats = vector_store.get_collection_stats()
    if stats.get("count", 0) == 0:
        logger.info("Knowledge base is empty. Loading sample documents...")
        load_sample_knowledge_base()
    
    # Test queries
    test_queries = [
        "Hi there!",
        "What's your return policy?",
        "How long does shipping take?",
        "Do you offer gift cards?",
        "I want to track my order",
        "What payment methods do you accept?",
        "Tell me about your warranty",
    ]
    
    print("\n" + "="*80)
    print("TESTING SUPPORT AGENT")
    print("="*80 + "\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_queries)}")
        print(f"{'='*80}")
        print(f"\n👤 USER: {query}\n")
        
        # Get response
        start_time = time.time()
        response = support_agent.process_message(query, session_id="test_session")
        elapsed = time.time() - start_time
        
        # Display response
        print(f"🤖 AGENT: {response.content}\n")
        
        # Display metadata
        print(f"📊 METADATA:")
        print(f"   • Confidence: {response.confidence:.2%}")
        print(f"   • Intent: {response.metadata.get('intent', 'unknown')}")
        print(f"   • Used RAG: {'Yes' if response.metadata.get('used_rag') else 'No'}")
        print(f"   • Response Time: {elapsed:.2f}s")
        
        if response.sources:
            print(f"   • Sources: {', '.join(response.sources)}")
        
        if response.needs_clarification:
            print(f"   ⚠️  Needs Clarification: {response.clarification_question}")
        
        print()
        
        # Pause between queries
        if i < len(test_queries):
            time.sleep(1)
    
    print("="*80)
    print("TESTING COMPLETE")
    print("="*80 + "\n")


def test_conversation():
    """Test multi-turn conversation"""
    
    print("\n" + "="*80)
    print("TESTING MULTI-TURN CONVERSATION")
    print("="*80 + "\n")
    
    conversation = [
        "Hello!",
        "I want to return a product",
        "How long do I have?",
        "What if it's been opened?",
        "Thanks for your help!"
    ]
    
    session_id = "conversation_test"
    
    for i, message in enumerate(conversation, 1):
        print(f"\n--- Turn {i} ---")
        print(f"👤 USER: {message}")
        
        response = support_agent.process_message(message, session_id=session_id)
        
        print(f"🤖 AGENT: {response.content}")
        print(f"   Confidence: {response.confidence:.2%}")
        
        time.sleep(1)
    
    print("\n" + "="*80 + "\n")


def test_low_confidence():
    """Test queries that should have low confidence"""
    
    print("\n" + "="*80)
    print("TESTING LOW CONFIDENCE SCENARIOS")
    print("="*80 + "\n")
    
    ambiguous_queries = [
        "Tell me about it",  # Ambiguous
        "What about the thing?",  # Vague
        "How much?",  # Lacks context
        "When will it arrive?"  # Missing order info
    ]
    
    for query in ambiguous_queries:
        print(f"\n👤 USER: {query}")
        response = support_agent.process_message(query, session_id="low_conf_test")
        print(f"🤖 AGENT: {response.content}")
        print(f"   Confidence: {response.confidence:.2%}")
        print(f"   Needs Clarification: {response.needs_clarification}")
        time.sleep(1)
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the support agent")
    parser.add_argument(
        "--mode",
        choices=["queries", "conversation", "confidence", "all"],
        default="all",
        help="Test mode to run"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "queries" or args.mode == "all":
            test_agent()
        
        if args.mode == "conversation" or args.mode == "all":
            test_conversation()
        
        if args.mode == "confidence" or args.mode == "all":
            test_low_confidence()
        
        print("\n✅ All tests completed successfully!\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user\n")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ Test failed: {e}\n")