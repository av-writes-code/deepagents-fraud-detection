#!/usr/bin/env python3
"""
Test DeepAgents Workflow End-to-End

Use only empirically validated tools:
- OpenCV (100% success rate)
- zxing-cpp (100% success rate)
- Standards validation (rule-based)
"""

import os
import json
from pathlib import Path
from deepagents_workflow import DeepAgentsIDWorkflow, IDVerificationState

def create_test_cases():
    """Create test cases for DeepAgents workflow"""

    print("🧪 Creating test cases for DeepAgents workflow...")

    test_cases = [
        {
            "name": "Simple Government ID",
            "image_path": "test_data/simple_text.png",
            "expected_elements": ["text_extraction", "document_structure"],
            "test_type": "ocr_validation"
        },
        {
            "name": "Mock MRZ Document",
            "image_path": "test_data/mock_mrz.png",
            "expected_elements": ["mrz_pattern", "passport_format"],
            "test_type": "mrz_validation"
        },
        {
            "name": "Mock Barcode Document",
            "image_path": "test_data/mock_barcode.png",
            "expected_elements": ["barcode_pattern", "structured_data"],
            "test_type": "barcode_validation"
        }
    ]

    # Verify test images exist
    valid_cases = []
    for case in test_cases:
        if Path(case["image_path"]).exists():
            valid_cases.append(case)
            print(f"  ✅ {case['name']}: {case['image_path']}")
        else:
            print(f"  ❌ {case['name']}: Missing {case['image_path']}")

    return valid_cases

def test_agent_workflow():
    """Test the complete DeepAgents workflow"""

    print("\n🚀 TESTING DEEPAGENTS WORKFLOW")
    print("=" * 50)

    # Initialize workflow
    workflow = DeepAgentsIDWorkflow()
    results = []

    # Get test cases
    test_cases = create_test_cases()

    if not test_cases:
        print("❌ No valid test cases found")
        return

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print("-" * 30)

        try:
            # Create initial state
            initial_state = IDVerificationState(
                image_path=test_case["image_path"],
                workflow_status="initialized",
                current_step="planning",
                agent_outputs={},
                extracted_data={},
                validation_results={},
                compliance_checks=[],
                final_report="",
                error_log=[]
            )

            # Run workflow
            print("🔄 Executing DeepAgents workflow...")
            final_state = workflow.process_document(initial_state)

            # Analyze results
            test_result = {
                "test_case": test_case["name"],
                "image_path": test_case["image_path"],
                "workflow_status": final_state.workflow_status,
                "agents_executed": list(final_state.agent_outputs.keys()),
                "data_extracted": bool(final_state.extracted_data),
                "validation_completed": bool(final_state.validation_results),
                "compliance_checks": len(final_state.compliance_checks),
                "report_generated": bool(final_state.final_report),
                "errors": final_state.error_log,
                "success": final_state.workflow_status == "completed"
            }

            # Display results
            if test_result["success"]:
                print(f"  ✅ Status: {final_state.workflow_status}")
                print(f"  📊 Agents: {', '.join(test_result['agents_executed'])}")
                print(f"  📋 Compliance checks: {test_result['compliance_checks']}")
                print(f"  📄 Report generated: {test_result['report_generated']}")
            else:
                print(f"  ❌ Status: {final_state.workflow_status}")
                print(f"  ⚠️ Errors: {len(test_result['errors'])}")
                for error in test_result["errors"]:
                    print(f"    - {error}")

            results.append(test_result)

        except Exception as e:
            print(f"  ❌ Workflow failed: {str(e)}")
            results.append({
                "test_case": test_case["name"],
                "success": False,
                "error": str(e)
            })

    return results

def analyze_workflow_performance(results):
    """Analyze DeepAgents workflow performance"""

    print(f"\n📊 DEEPAGENTS WORKFLOW ANALYSIS")
    print("=" * 50)

    if not results:
        print("❌ No results to analyze")
        return

    # Calculate metrics
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r.get("success", False))
    success_rate = (successful_tests / total_tests) * 100

    print(f"📈 Overall Success Rate: {success_rate:.0f}% ({successful_tests}/{total_tests})")

    # Agent execution analysis
    agent_counts = {}
    for result in results:
        if result.get("success"):
            for agent in result.get("agents_executed", []):
                agent_counts[agent] = agent_counts.get(agent, 0) + 1

    print(f"\n🤖 Agent Execution Success:")
    for agent, count in agent_counts.items():
        print(f"  {agent}: {count}/{successful_tests} successful tests")

    # Compliance analysis
    total_compliance_checks = sum(r.get("compliance_checks", 0) for r in results if r.get("success"))
    avg_compliance_checks = total_compliance_checks / successful_tests if successful_tests > 0 else 0

    print(f"\n📋 Compliance Analysis:")
    print(f"  Average compliance checks per document: {avg_compliance_checks:.1f}")
    print(f"  Total compliance validations: {total_compliance_checks}")

    # Error analysis
    all_errors = []
    for result in results:
        all_errors.extend(result.get("errors", []))

    if all_errors:
        print(f"\n⚠️ Error Analysis:")
        print(f"  Total errors encountered: {len(all_errors)}")
        for error in all_errors[:5]:  # Show first 5 errors
            print(f"    - {error}")

    return {
        "success_rate": success_rate,
        "successful_tests": successful_tests,
        "total_tests": total_tests,
        "agent_execution": agent_counts,
        "avg_compliance_checks": avg_compliance_checks,
        "total_errors": len(all_errors)
    }

def generate_workflow_report(results, analysis):
    """Generate comprehensive workflow test report"""

    print(f"\n📄 GENERATING WORKFLOW REPORT")
    print("=" * 40)

    report = {
        "test_metadata": {
            "test_date": "2025-09-20",
            "test_type": "DeepAgents Workflow End-to-End",
            "tools_used": ["OpenCV", "zxing-cpp", "Standards validation"],
            "test_scope": "Mock government ID documents"
        },
        "performance_summary": analysis,
        "detailed_results": results,
        "conclusions": {
            "workflow_viability": analysis["success_rate"] >= 50,
            "ready_for_production": False,
            "next_steps": [
                "Fix tool integration issues" if analysis["success_rate"] < 75 else "Scale testing",
                "Add real dataset testing",
                "Improve error handling",
                "Optimize agent coordination"
            ]
        }
    }

    # Save report
    report_path = Path("test_data") / "deepagents_workflow_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"📁 Report saved to: {report_path}")

    # Key findings
    print(f"\n🎯 KEY FINDINGS:")
    print(f"  Workflow Success Rate: {analysis['success_rate']:.0f}%")
    print(f"  Agent Coordination: {'✅ Working' if analysis['successful_tests'] > 0 else '❌ Failed'}")
    print(f"  Standards Compliance: {'✅ Implemented' if analysis['avg_compliance_checks'] > 0 else '❌ Missing'}")
    print(f"  Error Handling: {'⚠️ Needs improvement' if analysis['total_errors'] > 0 else '✅ Stable'}")

    return report

def main():
    """Main workflow testing function"""

    print("🚀 DEEPAGENTS WORKFLOW TESTING")
    print("=" * 50)
    print("Testing empirically validated DeepAgents implementation")

    # Step 1: Test workflow
    results = test_agent_workflow()

    # Step 2: Analyze performance
    analysis = analyze_workflow_performance(results)

    # Step 3: Generate report
    report = generate_workflow_report(results, analysis)

    # Step 4: Next steps
    print(f"\n🚀 READY FOR:")
    if analysis["success_rate"] >= 75:
        print("  ✅ Medium article creation")
        print("  ✅ Performance comparison")
        print("  ✅ GitHub repository setup")
    elif analysis["success_rate"] >= 50:
        print("  ⚠️ Debugging and improvements")
        print("  ✅ Partial article creation")
        print("  ✅ Architecture documentation")
    else:
        print("  ❌ Major debugging required")
        print("  ✅ Error analysis")
        print("  ✅ Tool integration fixes")

if __name__ == "__main__":
    main()