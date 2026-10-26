"""
UNIVERSAL CHATBOT EVALUATOR - MULTI-TURN METRICS ONLY
Turn Relevancy + Conversation Completeness + Knowledge Retention
"""

from deepeval.models import OllamaModel
ollama_model = OllamaModel(model="mistral")

try:
    from deepeval.test_case import ConversationalTestCase, Turn
except ImportError:
    class Turn:
        def __init__(self, role: str, content: str):
            self.role = role
            self.content = content
    
    from deepeval.test_case import LLMTestCase as ConversationalTestCase

class ConversationTurn:
    def __init__(self, user: str, assistant: str):
        self.user = user
        self.assistant = assistant
    
    def to_turn(self):
        """Convert to DeepEval Turn objects"""
        return [
            Turn(role="user", content=self.user),
            Turn(role="assistant", content=self.assistant)
        ]

from deepeval.metrics import (
    TurnRelevancyMetric,
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
)

from dotenv import load_dotenv
import json
from datetime import datetime
import os

load_dotenv()


class UniversalChatbotEvaluator:
    """Evaluate multi-turn chatbot conversations"""
    
    def __init__(self, chatbot_name: str, model=None):
        self.chatbot_name = chatbot_name
        self.model = model or ollama_model
        self.results = {}
        self.conversation_turns = []
        self.start_time = datetime.now()
    
    def add_conversation(self, turns):
        self.conversation_turns = turns
    
    def evaluate_all_metrics(self):
        """Evaluate 3 multi-turn metrics"""
        
        if not self.conversation_turns:
            print("❌ No conversation to evaluate!")
            return {}
        
        # Convert ConversationTurn to DeepEval Turn objects
        turns = []
        for turn in self.conversation_turns:
            turns.extend(turn.to_turn())
        
        test_case = ConversationalTestCase(turns=turns)
        
        print("\n" + "=" * 110)
        print(f"🤖 EVALUATING {self.chatbot_name.upper()}")
        print("=" * 110)
        print(f"Analyzing {len(self.conversation_turns)} conversation turns")
        print(f"Using model: Ollama (mistral)")
        print(f"Metrics: Turn Relevancy + Conversation Completeness + Knowledge Retention\n")
        
        print("📊 MULTI-TURN METRICS")
        print("-" * 110)
        
        # 3 Multi-Turn Metrics ONLY
        metrics_data = [
            ("Turn Relevancy", TurnRelevancyMetric(threshold=0.7, model=self.model)),
            ("Conversation Completeness", ConversationCompletenessMetric(threshold=0.7, model=self.model)),
            ("Knowledge Retention", KnowledgeRetentionMetric(threshold=0.7, model=self.model)),
        ]
        
        for name, metric in metrics_data:
            try:
                print(f"{name:30s}: ", end="", flush=True)
                metric.measure(test_case)
                self.results[name.lower().replace(' ', '_')] = {
                    'score': metric.score,
                    'passed': metric.success,
                    'reason': metric.reason
                }
                status = "✅" if metric.success else "❌"
                print(f"{metric.score:.2f}/1.0 {status}")
            except Exception as e:
                print(f"⚠️ Error - {str(e)[:40]}")
        
        return self.results
    
    def print_summary(self):
        if not self.results:
            print("❌ No metrics evaluated!")
            return
        
        print("\n" + "=" * 110)
        print("📋 SUMMARY")
        print("=" * 110)
        
        all_scores = [r['score'] for r in self.results.values()]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        
        print(f"Chatbot: {self.chatbot_name}")
        print(f"Conversation Turns: {len(self.conversation_turns)}")
        print(f"Metrics Evaluated: {len(self.results)}")
        print(f"Average Score: {avg_score:.2f}/1.0")
        
        if avg_score >= 0.9:
            verdict = "⭐⭐⭐ EXCELLENT"
        elif avg_score >= 0.8:
            verdict = "⭐⭐ GOOD"
        elif avg_score >= 0.7:
            verdict = "⭐ ACCEPTABLE"
        else:
            verdict = "❌ NEEDS IMPROVEMENT"
        
        print(f"Verdict: {verdict}")
        print("=" * 110)
    
    def save_report(self):
        if not self.results:
            print("No results to save!")
            return
        
        report = {
            "timestamp": self.start_time.isoformat(),
            "chatbot": self.chatbot_name,
            "model_used": "ollama/mistral",
            "conversation_turns": len(self.conversation_turns),
            "metrics_used": list(self.results.keys()),
            "metrics": self.results,
            "average_score": sum(r['score'] for r in self.results.values()) / len(self.results) if self.results else 0,
            "conversation": [
                {"turn": i+1, "user": turn.user, "assistant": turn.assistant}
                for i, turn in enumerate(self.conversation_turns)
            ]
        }
        
        filename = f"eval_{self.chatbot_name.lower()}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Report saved: {filename}")


def method1_manual_input():
    """METHOD 1: Manual text input"""
    
    print("\n" + "=" * 110)
    print("📝 METHOD 1: MANUAL TEXT INPUT")
    print("=" * 110)
    print("\nEnter your conversation turn by turn.")
    print("Type 'done' when finished.\n")
    
    turns = []
    
    while True:
        user_msg = input("👤 User: ").strip()
        
        if user_msg.lower() == 'done':
            break
        
        if not user_msg:
            continue
        
        bot_msg = input("🤖 Bot: ").strip()
        
        if not bot_msg:
            continue
        
        turns.append(ConversationTurn(user=user_msg, assistant=bot_msg))
        print()
    
    return turns


def method2_from_file():
    """METHOD 2: Load from text file"""
    
    print("\n" + "=" * 110)
    print("📝 METHOD 2: LOAD FROM TEXT FILE")
    print("=" * 110)
    
    filename = input("\n📁 Enter filename (e.g., my_conversation.txt): ").strip()
    
    if not filename or not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        turns = []
        lines = content.split('\n')
        current_user = ""
        current_bot = ""
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('USER:') or line.startswith('user:'):
                if current_user and current_bot:
                    turns.append(ConversationTurn(user=current_user, assistant=current_bot))
                current_user = line.replace('USER:', '').replace('user:', '').strip()
                current_bot = ""
            
            elif line.startswith('BOT:') or line.startswith('bot:'):
                current_bot = line.replace('BOT:', '').replace('bot:', '').strip()
            
            elif line and current_user and not current_bot:
                current_bot += ' ' + line
        
        if current_user and current_bot:
            turns.append(ConversationTurn(user=current_user, assistant=current_bot))
        
        print(f"✅ Loaded {len(turns)} conversation turns")
        return turns
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def method3_from_json():
    """METHOD 3: Load from JSON file"""
    
    print("\n" + "=" * 110)
    print("📝 METHOD 3: LOAD FROM JSON FILE")
    print("=" * 110)
    
    filename = input("\nEnter JSON filename (e.g., conversation.json): ").strip()
    
    if not filename or not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return []
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        turns = []
        
        if isinstance(data, list):
            for item in data:
                turns.append(ConversationTurn(
                    user=item.get('user', item.get('question', '')),
                    assistant=item.get('assistant', item.get('response', item.get('answer', '')))
                ))
        
        elif isinstance(data, dict) and 'turns' in data:
            for turn in data['turns']:
                turns.append(ConversationTurn(
                    user=turn.get('user', ''),
                    assistant=turn.get('assistant', '')
                ))
        
        print(f"✅ Loaded {len(turns)} conversation turns")
        return turns
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def main():
    print("\n" + "=" * 110)
    print("🎯 UNIVERSAL CHATBOT EVALUATOR")
    print("=" * 110)
    print("Multi-Turn Metrics: Turn Relevancy + Conversation Completeness + Knowledge Retention")
    print("Using Ollama (mistral) - Free, Local, No API Keys\n")
    
    print("📥 CHOOSE INPUT METHOD:")
    print("1️⃣  Manual text input")
    print("2️⃣  Load from text file")
    print("3️⃣  Load from JSON file")
    
    choice = input("\nSelect (1-3): ").strip()
    
    turns = []
    
    if choice == '1':
        turns = method1_manual_input()
    elif choice == '2':
        turns = method2_from_file()
    elif choice == '3':
        turns = method3_from_json()
    else:
        print("❌ Invalid choice!")
        return
    
    if not turns:
        print("❌ No conversation loaded!")
        return
    
    print("\n" + "=" * 110)
    chatbot_name = input("What chatbot? (ChatGPT/Claude/Gemini/MyBot): ").strip() or "Unknown"
    
    print("\n⏳ Evaluating your conversation...\n")
    
    evaluator = UniversalChatbotEvaluator(chatbot_name, model=ollama_model)
    evaluator.add_conversation(turns)
    evaluator.evaluate_all_metrics()
    evaluator.print_summary()
    evaluator.save_report()


if __name__ == "__main__":
    main()