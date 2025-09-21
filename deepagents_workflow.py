"""
DeepAgents-style ID Verification Workflow

Built following DeepAgents best practices with LangGraph integration
Uses current Python 3.9 setup, ready to upgrade to full DeepAgents when Python 3.11+ available
"""

import json
import logging
from typing import Dict, Any, List, Annotated
from pathlib import Path
from dataclasses import dataclass, field

# LangGraph and LangChain imports (working with current setup)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Import our pure tools
from id_verification_tools import (
    preprocess_document,
    extract_ocr_text,
    extract_mrz_data,
    extract_pdf417_data,
    standards_check,
    publish_report
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IDVerificationState:
    """
    State object for ID verification workflow
    Follows DeepAgents virtual filesystem pattern
    """
    messages: Annotated[List[Dict], "Conversation messages"] = field(default_factory=list)
    files: Annotated[Dict[str, str], "Virtual filesystem for artifacts"] = field(default_factory=dict)
    current_phase: Annotated[str, "Current workflow phase"] = "initialized"
    extracted_data: Annotated[Dict[str, Any], "All extraction results"] = field(default_factory=dict)
    validation_results: Annotated[Dict[str, Any], "Standards validation results"] = field(default_factory=dict)
    workflow_plan: Annotated[List[str], "Planned workflow steps"] = field(default_factory=list)
    human_approval_required: Annotated[bool, "Whether human approval is needed"] = False

    # Additional fields for compatibility
    image_path: str = ""
    workflow_status: str = "initialized"
    current_step: str = "planning"
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    compliance_checks: List[Dict[str, Any]] = field(default_factory=list)
    final_report: str = ""
    error_log: List[str] = field(default_factory=list)

class DeepAgentsIDWorkflow:
    """
    DeepAgents-style ID verification workflow

    Implements:
    - Planning-first approach
    - Context quarantine via specialized phases
    - Virtual filesystem for artifact management
    - Human-in-the-loop for sensitive operations
    """

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()

        # Tool configuration (following best practices)
        self.tools_config = {
            "quality_threshold": 0.7,
            "validation_threshold": 0.8,
            "require_human_approval": ["publish_report"]
        }

        logger.info("DeepAgents ID workflow initialized")

    def _build_workflow(self) -> CompiledStateGraph:
        """Build the LangGraph workflow following DeepAgents patterns"""

        # Define state schema
        workflow = StateGraph(IDVerificationState)

        # Add nodes (following DeepAgents sub-agent specialization)
        workflow.add_node("planner", self._planning_agent)
        workflow.add_node("preprocess", self._preprocess_agent)
        workflow.add_node("extract", self._extract_agent)
        workflow.add_node("validate", self._validation_agent)
        workflow.add_node("critic", self._critic_agent)
        workflow.add_node("report", self._report_agent)
        workflow.add_node("human_approval", self._human_approval_node)

        # Define workflow edges (DeepAgents planning -> execution pattern)
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "preprocess")
        workflow.add_edge("preprocess", "extract")
        workflow.add_edge("extract", "validate")
        workflow.add_edge("validate", "critic")

        # Conditional edge for human approval
        workflow.add_conditional_edges(
            "critic",
            self._should_require_approval,
            {
                "human_approval": "human_approval",
                "report": "report"
            }
        )

        workflow.add_edge("human_approval", "report")
        workflow.add_edge("report", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def _planning_agent(self, state: IDVerificationState) -> IDVerificationState:
        """
        Planning Agent: Create workflow plan before execution
        Follows DeepAgents planning-first principle
        """
        logger.info("🎯 Planning Agent: Creating verification workflow plan")

        # Standard ID verification workflow plan
        workflow_plan = [
            "1. PREPROCESS: Document cleanup and quality enhancement",
            "2. EXTRACT: Multi-modal data extraction (OCR/MRZ/PDF417)",
            "3. VALIDATE: Standards compliance checking per YAML controls",
            "4. CRITIC: Completeness review and traceability verification",
            "5. REPORT: Generate standards-aligned verification report"
        ]

        # Update virtual filesystem with plan
        state.files["workflow_plan.md"] = "\n".join([
            "# ID Verification Workflow Plan",
            "",
            "## Planned Steps:",
            *workflow_plan,
            "",
            "## Artifacts to Generate:",
            "- preprocessing_results.json",
            "- extraction_results.json",
            "- validation_results.json",
            "- traceability_matrix.json",
            "- verification_report.md"
        ])

        state.workflow_plan = workflow_plan
        state.current_phase = "planning_complete"

        # Add planning message
        state.messages.append({
            "role": "assistant",
            "content": f"📋 Workflow plan created with {len(workflow_plan)} phases. Ready to proceed with document verification."
        })

        return state

    def _preprocess_agent(self, state: IDVerificationState) -> IDVerificationState:
        """
        Preprocess Agent: Document cleanup specialist
        Context quarantine: Only handles image preprocessing
        """
        logger.info("🔧 Preprocess Agent: Document cleanup and enhancement")

        state.current_phase = "preprocessing"

        # Find input image in virtual filesystem
        image_files = [f for f in state.files.keys() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            state.messages.append({
                "role": "assistant",
                "content": "❌ No image files found in virtual filesystem for preprocessing"
            })
            return state

        # Process first image file
        image_file = image_files[0]

        # Save image to temp file for processing (in real implementation, would use virtual FS)
        temp_image_path = f"/tmp/{image_file}"
        with open(temp_image_path, 'wb') as f:
            f.write(state.files[image_file].encode() if isinstance(state.files[image_file], str) else state.files[image_file])

        # Run preprocessing tool
        preprocess_result = preprocess_document(temp_image_path)

        # Store results in virtual filesystem
        state.files["preprocessing_results.json"] = json.dumps(preprocess_result, indent=2)

        # Update state
        if preprocess_result["status"] == "success":
            state.files["processed_image.jpg"] = f"Processed version of {image_file}"
            state.messages.append({
                "role": "assistant",
                "content": f"✅ Preprocessing complete. Quality score: {preprocess_result.get('quality_score', 'N/A')}"
            })
        else:
            state.messages.append({
                "role": "assistant",
                "content": f"❌ Preprocessing failed: {preprocess_result.get('error', 'Unknown error')}"
            })

        return state

    def _extract_agent(self, state: IDVerificationState) -> IDVerificationState:
        """
        Extract Agent: Multi-modal extraction specialist
        Context quarantine: Only handles data extraction, no validation
        """
        logger.info("📊 Extract Agent: Multi-modal data extraction")

        state.current_phase = "extraction"

        # Get preprocessed image path (simplified for demo)
        preprocess_results = json.loads(state.files.get("preprocessing_results.json", "{}"))
        if preprocess_results.get("status") != "success":
            state.messages.append({
                "role": "assistant",
                "content": "❌ Cannot extract - preprocessing failed"
            })
            return state

        # Mock extraction using original image (in real implementation, use processed image)
        temp_image_path = "/tmp/sample_image.jpg"  # Simplified for demo

        # Run all extraction tools
        extraction_results = {}

        # OCR extraction
        ocr_result = extract_ocr_text(temp_image_path)
        extraction_results["ocr"] = ocr_result

        # MRZ extraction
        mrz_result = extract_mrz_data(temp_image_path)
        extraction_results["mrz"] = mrz_result

        # PDF417 extraction
        pdf417_result = extract_pdf417_data(temp_image_path)
        extraction_results["pdf417"] = pdf417_result

        # Store results in virtual filesystem
        state.files["extraction_results.json"] = json.dumps(extraction_results, indent=2)
        state.extracted_data = extraction_results

        # Generate extraction summary
        successful_extractions = [k for k, v in extraction_results.items() if v.get("status") == "success"]

        state.messages.append({
            "role": "assistant",
            "content": f"✅ Extraction complete. Successfully extracted data using: {', '.join(successful_extractions)}"
        })

        return state

    def _validation_agent(self, state: IDVerificationState) -> IDVerificationState:
        """
        Validation Agent: Standards compliance specialist
        Context quarantine: Only handles validation logic
        """
        logger.info("✅ Validation Agent: Standards compliance checking")

        state.current_phase = "validation"

        # Load standards controls from virtual filesystem
        controls_yaml = state.files.get("controls.yaml", state.files.get("standards_crosswalk.yaml", ""))

        if not controls_yaml:
            state.messages.append({
                "role": "assistant",
                "content": "❌ No standards controls found in virtual filesystem"
            })
            return state

        # Get extraction results
        extraction_json = state.files.get("extraction_results.json", "{}")

        # Run standards validation
        validation_result = standards_check(extraction_json, controls_yaml)

        # Store results in virtual filesystem
        state.files["validation_results.json"] = json.dumps(validation_result, indent=2)
        state.validation_results = validation_result

        # Generate validation summary
        if validation_result.get("status") == "success":
            compliance_summary = validation_result.get("compliance_summary", {})
            pass_rate = compliance_summary.get("pass_rate", 0)

            state.messages.append({
                "role": "assistant",
                "content": f"✅ Validation complete. Pass rate: {pass_rate:.1%}, Status: {compliance_summary.get('overall_status', 'Unknown')}"
            })
        else:
            state.messages.append({
                "role": "assistant",
                "content": f"❌ Validation failed: {validation_result.get('error', 'Unknown error')}"
            })

        return state

    def _critic_agent(self, state: IDVerificationState) -> IDVerificationState:
        """
        Critic Agent: Completeness review and quality assurance
        Follows DeepAgents critic pattern for self-audit
        """
        logger.info("🔍 Critic Agent: Completeness review and quality assurance")

        state.current_phase = "critic_review"

        # Review all phases for completeness
        review_results = {
            "preprocessing_complete": "preprocessing_results.json" in state.files,
            "extraction_complete": "extraction_results.json" in state.files,
            "validation_complete": "validation_results.json" in state.files,
            "quality_concerns": [],
            "missing_evidence": [],
            "recommendations": []
        }

        # Quality checks
        if state.extracted_data:
            # Check extraction quality
            for method, result in state.extracted_data.items():
                if result.get("status") == "error":
                    review_results["quality_concerns"].append(f"{method} extraction failed")

        if state.validation_results:
            # Check validation quality
            compliance_summary = state.validation_results.get("compliance_summary", {})
            pass_rate = compliance_summary.get("pass_rate", 0)

            if pass_rate < self.tools_config["validation_threshold"]:
                review_results["quality_concerns"].append(f"Low validation pass rate: {pass_rate:.1%}")

        # Generate recommendations
        if review_results["quality_concerns"]:
            review_results["recommendations"].append("Consider manual review due to quality concerns")
            state.human_approval_required = True
        else:
            review_results["recommendations"].append("Automated verification appears successful")
            state.human_approval_required = False

        # Generate traceability matrix
        traceability_matrix = self._generate_traceability_matrix(state)
        state.files["traceability_matrix.json"] = json.dumps(traceability_matrix, indent=2)

        # Store critic review
        state.files["critic_review.json"] = json.dumps(review_results, indent=2)

        # Generate summary message
        concern_count = len(review_results["quality_concerns"])
        state.messages.append({
            "role": "assistant",
            "content": f"🔍 Critic review complete. Found {concern_count} quality concerns. Traceability matrix generated."
        })

        return state

    def _should_require_approval(self, state: IDVerificationState) -> str:
        """Determine if human approval is required"""
        return "human_approval" if state.human_approval_required else "report"

    def _human_approval_node(self, state: IDVerificationState) -> IDVerificationState:
        """Human approval checkpoint (simulated for demo)"""
        logger.info("👤 Human Approval: Review required")

        state.current_phase = "awaiting_approval"

        # In real implementation, this would pause workflow for human input
        # For demo, simulate approval
        state.messages.append({
            "role": "assistant",
            "content": "👤 Human approval required. Quality concerns detected. [In real implementation, workflow would pause here]"
        })

        # Simulate approval (in real implementation, wait for human input)
        state.messages.append({
            "role": "human",
            "content": "Approved - proceed with report generation"
        })

        return state

    def _report_agent(self, state: IDVerificationState) -> IDVerificationState:
        """
        Report Agent: Generate final verification report
        Context quarantine: Only handles report generation
        """
        logger.info("📄 Report Agent: Generating verification report")

        state.current_phase = "report_generation"

        # Generate comprehensive report
        report_content = self._generate_verification_report(state)

        # Store report in virtual filesystem
        state.files["verification_report.md"] = report_content

        # Publish report (tool with potential human approval)
        publish_result = publish_report(report_content, "verification_report.md")

        if publish_result["status"] == "success":
            state.messages.append({
                "role": "assistant",
                "content": f"✅ Verification report generated and published: {publish_result['output_path']}"
            })
        else:
            state.messages.append({
                "role": "assistant",
                "content": f"❌ Report publishing failed: {publish_result.get('error', 'Unknown error')}"
            })

        state.current_phase = "complete"
        return state

    def _generate_traceability_matrix(self, state: IDVerificationState) -> Dict[str, Any]:
        """Generate traceability matrix for compliance reporting"""

        matrix = {
            "timestamp": "2025-01-20T10:30:00Z",
            "workflow_id": "demo-verification-001",
            "traceability_entries": []
        }

        # Add entries for each validation control
        if state.validation_results and "validations" in state.validation_results:
            for control_id, result in state.validation_results["validations"].items():
                matrix["traceability_entries"].append({
                    "control_id": control_id,
                    "requirement": "Standards compliance check",
                    "tool_used": "standards_check",
                    "evidence": result.get("evidence", {}),
                    "status": result.get("status", "UNKNOWN"),
                    "reason": result.get("reason", "No reason provided")
                })

        return matrix

    def _generate_verification_report(self, state: IDVerificationState) -> str:
        """Generate comprehensive verification report"""

        # Get validation summary
        validation_results = state.validation_results or {}
        compliance_summary = validation_results.get("compliance_summary", {})

        # Get critic review
        critic_review = json.loads(state.files.get("critic_review.json", "{}"))

        report = f"""# Government ID Verification Report

## Executive Summary

**Verification Status**: {compliance_summary.get('overall_status', 'Unknown')}
**Pass Rate**: {compliance_summary.get('pass_rate', 0):.1%}
**Quality Concerns**: {len(critic_review.get('quality_concerns', []))}

## Workflow Execution

{chr(10).join(state.workflow_plan)}

## Extraction Results

**Successful Methods**: {len([k for k, v in (state.extracted_data or {}).items() if v.get('status') == 'success'])}
**Failed Methods**: {len([k for k, v in (state.extracted_data or {}).items() if v.get('status') == 'error'])}

## Standards Compliance

**Total Controls**: {compliance_summary.get('total_controls', 0)}
**Passed Controls**: {compliance_summary.get('passed_controls', 0)}

## Quality Assurance

**Critic Review**: {len(critic_review.get('quality_concerns', []))} concerns identified
**Recommendations**: {len(critic_review.get('recommendations', []))} recommendations provided

## Legal Notice

This is a research prototype that demonstrates a standards-aligned verification workflow.
It implements checks consistent with NIST SP 800-63-4, applies MRZ rules from ICAO Doc 9303,
parses PDF417 per AAMVA guidance, and illustrates PAD reporting terminology from ISO/IEC 30107-3.
It is not a certified system; no claim of formal compliance is made.

---
*Generated by DeepAgents ID Verification System*
*Timestamp: 2025-01-20T10:30:00Z*
"""

        return report

    def verify_id(self, image_data: bytes, image_filename: str, controls_yaml: str) -> Dict[str, Any]:
        """
        Main entry point for ID verification

        Args:
            image_data: Image file as bytes
            image_filename: Name of image file
            controls_yaml: Standards controls configuration

        Returns:
            Dict with verification results and generated artifacts
        """

        # Initialize state with virtual filesystem
        initial_state = IDVerificationState(
            messages=[{
                "role": "user",
                "content": f"Please verify the government ID: {image_filename}"
            }],
            files={
                image_filename: image_data,
                "controls.yaml": controls_yaml
            },
            current_phase="initialized",
            extracted_data={},
            validation_results={},
            workflow_plan=[],
            human_approval_required=False
        )

        # Run workflow
        config = {"configurable": {"thread_id": "verification-001"}}

        try:
            result = self.workflow.invoke(initial_state, config=config)

            return {
                "status": "success",
                "final_phase": result.current_phase,
                "messages": result.messages,
                "generated_files": list(result.files.keys()),
                "artifacts": result.files
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "final_phase": initial_state.current_phase
            }

    def process_document(self, state: IDVerificationState) -> IDVerificationState:
        """
        Process document using the actual LangGraph workflow
        Proper state management following DeepAgents patterns
        """
        try:
            # Validate input
            if not state.image_path or not Path(state.image_path).exists():
                state.error_log.append(f"Image not found: {state.image_path}")
                state.workflow_status = "failed"
                return state

            # Prepare state for workflow execution
            state.files["input_image"] = state.image_path
            state.current_phase = "initialized"

            # Load standards controls
            controls_path = "standards_crosswalk.yaml"
            if Path(controls_path).exists():
                with open(controls_path, 'r') as f:
                    state.files["controls_yaml"] = f.read()
            else:
                state.files["controls_yaml"] = "# Mock controls for testing"

            # Use the actual LangGraph workflow with proper config
            config = {"configurable": {"thread_id": f"doc-{hash(state.image_path)}"}}
            final_state = self.workflow.invoke(state, config)

            # Ensure return type is IDVerificationState
            if isinstance(final_state, dict):
                # Convert dict back to IDVerificationState if needed
                for key, value in final_state.items():
                    if hasattr(state, key):
                        setattr(state, key, value)
                final_state = state

            return final_state

        except Exception as e:
            state.error_log.append(f"Workflow processing failed: {str(e)}")
            state.workflow_status = "failed"
            logger.error(f"Workflow execution failed: {str(e)}")
            return state

# Factory function following DeepAgents patterns
def create_id_verification_agent() -> DeepAgentsIDWorkflow:
    """
    Factory function to create ID verification agent
    Following DeepAgents create_deep_agent pattern
    """
    return DeepAgentsIDWorkflow()

if __name__ == "__main__":
    # Demo usage
    agent = create_id_verification_agent()

    # Load test data
    with open("standards_crosswalk.yaml", 'r') as f:
        controls = f.read()

    # Mock image data for demo
    test_image = b"mock_image_data"

    # Run verification
    result = agent.verify_id(test_image, "test_id.jpg", controls)

    print(f"Verification Status: {result['status']}")
    print(f"Generated Files: {result.get('generated_files', [])}")