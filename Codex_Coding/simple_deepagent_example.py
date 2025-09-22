from typing import Dict, Any, List
from dataclasses import dataclass
import json

@dataclass
class AgentState:
    """Shared state between orchestrator and sub-agents"""
    image_path: str = ""
    messages: List[Dict] = None
    ocr_results: Dict = None
    current_phase: str = "init"

    def __post_init__(self):
        if self.messages is None:
            self.messages = []

class OCRSubAgent:
    """Sub-agent specialized in OCR text extraction"""

    def __init__(self):
        self.agent_name = "OCR_SubAgent"

    def process(self, state: AgentState) -> AgentState:
        """Process OCR extraction on the image"""
        print(f"🔍 {self.agent_name}: Starting OCR processing...")

        try:
            # Import the OCR tool
            from id_verification_tools import extract_ocr_text

            # Run OCR extraction
            print(f"   THINKING: Applying PaddleOCR to extract text from {state.image_path}")
            ocr_result = extract_ocr_text(state.image_path)

            # Store results in state
            state.ocr_results = ocr_result
            state.current_phase = "ocr_complete"

            # Add message to state
            status = ocr_result.get('status', 'unknown')
            text_count = len(ocr_result.get('raw_text', []))

            state.messages.append({
                "agent": self.agent_name,
                "status": status,
                "text_extracted": text_count,
                "message": f"OCR processing complete - Status: {status}, Text items: {text_count}"
            })

            print(f"   ✅ {self.agent_name}: Complete - Status: {status}")

        except Exception as e:
            print(f"   ❌ {self.agent_name}: Error - {e}")
            state.messages.append({
                "agent": self.agent_name,
                "status": "error",
                "error": str(e)
            })

        return state

class DeepAgentOrchestrator:
    """Orchestrator that manages sub-agents and workflow"""

    def __init__(self):
        self.orchestrator_name = "DeepAgent_Orchestrator"
        self.ocr_agent = OCRSubAgent()

    def process_document(self, image_path: str) -> Dict[str, Any]:
        """Main orchestrator method that coordinates sub-agents"""

        print("="*60)
        print(f"🎯 {self.orchestrator_name}: Starting document processing")
        print("="*60)

        # Initialize state
        state = AgentState(image_path=image_path)
        state.current_phase = "orchestrator_started"

        # PHASE 1: Orchestrator analysis
        print(f"\n🧠 PHASE 1: Orchestrator analyzing document...")
        print(f"   Document: {image_path}")

        # PHASE 2: Delegate to OCR sub-agent
        print(f"\n🚀 PHASE 2: Delegating to OCR sub-agent...")
        state = self.ocr_agent.process(state)

        # PHASE 3: Orchestrator review and decision
        print(f"\n📊 PHASE 3: Orchestrator reviewing results...")
        final_result = self._review_and_decide(state)

        print("="*60)
        print(f"🏁 {self.orchestrator_name}: Processing complete")
        print("="*60)

        return final_result

    def _review_and_decide(self, state: AgentState) -> Dict[str, Any]:
        """Orchestrator reviews sub-agent results and makes decisions"""

        print(f"   THINKING: Reviewing OCR sub-agent results...")

        # Analyze OCR results
        ocr_status = state.ocr_results.get('status', 'unknown') if state.ocr_results else 'no_results'
        text_count = len(state.ocr_results.get('raw_text', [])) if state.ocr_results else 0

        # Make orchestrator decision
        if ocr_status == 'success' and text_count > 0:
            orchestrator_decision = "OCR_SUCCESS_PROCEED"
            confidence = 0.8
        elif ocr_status == 'success' and text_count == 0:
            orchestrator_decision = "OCR_SUCCESS_NO_TEXT"
            confidence = 0.3
        else:
            orchestrator_decision = "OCR_FAILED_REVIEW_NEEDED"
            confidence = 0.1

        print(f"   DECISION: {orchestrator_decision} (confidence: {confidence})")

        # Compile final result
        return {
            "orchestrator": self.orchestrator_name,
            "status": "complete",
            "decision": orchestrator_decision,
            "confidence": confidence,
            "sub_agents_used": ["OCR_SubAgent"],
            "ocr_results": state.ocr_results,
            "messages": state.messages,
            "final_phase": state.current_phase
        }

# Usage example:
if __name__ == "__main__":
    # Create orchestrator
    orchestrator = DeepAgentOrchestrator()

    # Process document
    image_path = "/path/to/your/document.png"
    result = orchestrator.process_document(image_path)

    # Print results
    print(f"\nFinal Decision: {result['decision']}")
    print(f"Confidence: {result['confidence']}")
    print(f"OCR Status: {result['ocr_results']['status']}")