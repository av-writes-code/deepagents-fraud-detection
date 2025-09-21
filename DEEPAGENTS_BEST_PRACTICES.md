# DeepAgents Best Practices Guide

## Core Architecture Principles

### 1. Tool Design Patterns

**Pure & Idempotent Tools**
```python
# ✅ GOOD: Pure function, predictable output
def extract_mrz_data(image_path: str) -> Dict[str, Any]:
    # Always returns same result for same input
    # No side effects except logging

# ❌ BAD: Stateful, unpredictable
def extract_mrz_data(self, image_path: str):
    self.last_result = process()  # Mutates state
```

**Consistent Error Handling**
```python
# ✅ GOOD: Consistent return structure
return {
    "status": "success|error",
    "tool": "tool_name",
    "data": {...},
    "error": "error_message" if failed
}

# ❌ BAD: Inconsistent returns
return data_dict  # No status info
raise Exception()  # Crashes agent
```

### 2. Context Quarantine Strategy

**Sub-agent Specialization**
```python
subagents = [
    {
        "name": "extract",
        "description": "Multi-modal data extraction only",
        "prompt": "You extract fields with high recall. Never speculate outside pixels.",
        "tools": ["extract_ocr_text", "extract_mrz_data", "extract_pdf417_data"]
    },
    {
        "name": "checks",
        "description": "Standards validation only",
        "prompt": "Validate against YAML controls. Explain each pass/fail briefly.",
        "tools": ["standards_check"]
    }
]
```

**Benefits:**
- **Large/noisy data** stays quarantined in specialized sub-agents
- **Main agent context** remains clean and focused
- **Parallel processing** of independent tasks
- **Easier debugging** with isolated failure domains

### 3. Virtual Filesystem Patterns

**Use Files for Large Data**
```python
# ✅ GOOD: Store large content in virtual FS
files = {
    "controls.yaml": open("standards_crosswalk.yaml").read(),
    "sample_id.jpg": open("test_image.jpg", "rb").read(),
    "extraction_results.json": json.dumps(large_data)
}

# Reference files in prompts
prompt = "Read controls.yaml and validate extraction_results.json"

# ❌ BAD: Paste large content in prompt
prompt = f"Validate this data: {huge_json_blob}"  # Context pollution
```

**Artifact Management**
```python
# Write intermediate results to virtual FS
def write_extraction_artifacts(results):
    return {
        "mrz_data.json": json.dumps(mrz_results),
        "pdf417_data.json": json.dumps(barcode_results),
        "ocr_results.json": json.dumps(text_results)
    }
```

### 4. Planning-First Workflow

**Always Plan Before Execute**
```python
# ✅ GOOD: Explicit planning phase
planning_prompt = """
Plan the ID verification workflow:
1. preprocess -> extract -> validate -> report
2. Artifacts: extraction.json, validation.json, report.md
3. Standards: NIST/ICAO/AAMVA checks per controls.yaml
"""

# Use DeepAgents built-in planning tool
# Agent will write TODOs and follow them
```

**Task Breakdown Strategy**
- Break complex tasks into **discrete sub-tasks**
- Each sub-task maps to **one specialist sub-agent**
- **Clear handoffs** between agents with file artifacts
- **Checkpoints** for human-in-the-loop approval

### 5. Human-in-the-Loop Guardrails

**Gate Sensitive Operations**
```python
interrupt_config = {
    "publish_report": True,      # Require approval before publishing
    "standards_check": False,    # Auto-approve validation
    "extract_mrz_data": False    # Auto-approve extraction
}

# Critical: Use checkpointer for interrupts
checkpointer = InMemorySaver()
graph = agent.compile(checkpointer=checkpointer)
```

**Approval Patterns**
- **Low-risk tools**: Auto-approve (extraction, validation)
- **High-risk tools**: Require approval (publishing, external calls)
- **Sensitive data**: Human review before processing
- **Standards compliance**: Auto-validate, human review edge cases

### 6. Standards Compliance Integration

**YAML-Driven Configuration**
```python
# ✅ GOOD: Vendor-neutral standards mapping
standards_config = {
    "controls": [
        {
            "id": "ICAO_9303_MRZ_CHECKSUM",
            "requirement": "All MRZ check digits validate correctly",
            "toolchain": ["extract_mrz_data"],
            "agent_mapping": {
                "extract_agent": "MRZ data extraction",
                "checks_agent": "Checksum validation"
            }
        }
    ]
}

# ❌ BAD: Hardcoded compliance logic
if document_type == "passport":
    validate_icao_checksums()  # Not configurable
```

**Traceability Matrix Generation**
```python
# Each validation produces auditable evidence
validation_result = {
    "control_id": "ICAO_9303_MRZ_CHECKSUM",
    "status": "PASS|FAIL",
    "evidence": {"checksum_results": {...}},
    "tool_used": "extract_mrz_data",
    "timestamp": "2025-01-20T10:30:00Z"
}
```

### 7. Memory Management Best Practices

**Context Window Hygiene**
- Keep main agent prompts **< 40% context window**
- Use sub-agents for **context quarantine**
- Store large data in **virtual filesystem**
- **Compact errors** to essential information only

**State Persistence**
```python
# Use thread IDs for stateful conversations
config = {
    "configurable": {"thread_id": "verification-001"},
    "interrupt_config": interrupt_config
}

# Results persist across interrupts/resumes
result = graph.invoke(input_data, config=config)
```

### 8. Error Handling & Recovery

**Graceful Degradation**
```python
# ✅ GOOD: Partial success handling
def extract_all_data(image_path):
    results = {}

    # Try each extraction method independently
    results["ocr"] = extract_ocr_text(image_path)
    results["mrz"] = extract_mrz_data(image_path)  # May fail
    results["pdf417"] = extract_pdf417_data(image_path)  # May fail

    # Continue with available data
    return {
        "status": "partial_success",
        "available_data": [k for k, v in results.items() if v["status"] == "success"],
        "failed_extractions": [k for k, v in results.items() if v["status"] == "error"],
        "results": results
    }
```

**Recovery Strategies**
- **Tool failures**: Continue with available data
- **Sub-agent failures**: Retry with different approach
- **Validation failures**: Flag for human review
- **Critical failures**: Stop and request intervention

### 9. Production Deployment Patterns

**Environment Setup**
```python
# Use proper Python version for DeepAgents
# Python 3.11+ required
python3.11 -m venv deepagents_env
source deepagents_env/bin/activate
pip install deepagents langgraph langchain-openai

# Keep existing OSS tools in separate venv if needed
# Avoid version conflicts
```

**Configuration Management**
```python
# Separate config from code
TOOLS_CONFIG = {
    "ocr_engine": "paddleocr",
    "mrz_processor": "passporteye",
    "barcode_decoder": "zxing-cpp",
    "quality_threshold": 0.7
}

AGENT_CONFIG = {
    "model": "claude-3-sonnet",
    "temperature": 0.0,
    "max_tokens": 4000
}
```

### 10. Testing & Validation

**Tool Testing**
```python
# Test each tool independently
def test_extract_mrz_data():
    result = extract_mrz_data("test_passport.jpg")
    assert result["status"] == "success"
    assert result["mrz_found"] == True
    assert "document_number" in result
```

**End-to-End Testing**
```python
# Test complete workflow
def test_id_verification_workflow():
    files = {"test_id.jpg": load_test_image()}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Verify this ID"}]},
        config={"configurable": {"thread_id": "test-001"}}
    )
    assert "verification_report.md" in result["files"]
```

## Anti-Patterns to Avoid

### ❌ Context Pollution
```python
# BAD: Dumping large data in prompts
prompt = f"Analyze this OCR data: {massive_json_dump}"

# GOOD: Reference files
prompt = "Analyze OCR data in ocr_results.json"
```

### ❌ Stateful Tools
```python
# BAD: Tools that modify global state
class StatefulExtractor:
    def __init__(self):
        self.cache = {}  # Shared state

    def extract(self, image):
        self.cache[image] = result  # Modifies state

# GOOD: Pure functions
def extract_data(image_path: str) -> Dict[str, Any]:
    return {...}  # Same input = same output
```

### ❌ Monolithic Agents
```python
# BAD: One agent doing everything
agent_prompt = """You are an ID verification system.
Extract text, parse barcodes, validate checksums, generate reports..."""

# GOOD: Specialized sub-agents
subagents = [
    {"name": "extract", "description": "Data extraction only"},
    {"name": "validate", "description": "Standards validation only"},
    {"name": "report", "description": "Report generation only"}
]
```

### ❌ Hardcoded Compliance
```python
# BAD: Hardcoded validation rules
if document_type == "passport":
    check_mrz_checksums()
elif document_type == "license":
    check_pdf417_format()

# GOOD: YAML-driven validation
for control in yaml_controls:
    validate_control(extracted_data, control)
```

## Quick Reference Checklist

### Before Building
- [ ] Python 3.11+ environment set up
- [ ] DeepAgents installed without disrupting existing tools
- [ ] Standards YAML configuration created
- [ ] Test images and data prepared

### During Development
- [ ] Tools are pure functions (no side effects)
- [ ] Sub-agents have clear, narrow responsibilities
- [ ] Large data stored in virtual filesystem
- [ ] Planning phase before execution
- [ ] Error handling returns consistent structure

### Before Deployment
- [ ] Human-in-the-loop approvals configured
- [ ] All tools tested independently
- [ ] End-to-end workflow validated
- [ ] Traceability matrix generation verified
- [ ] Legal disclaimers included in outputs

### Monitoring in Production
- [ ] Context window utilization < 60%
- [ ] Sub-agent performance metrics tracked
- [ ] Validation pass rates monitored
- [ ] Human approval patterns analyzed
- [ ] Error rates and recovery success tracked

---

## 📚 Essential Links & References

### 🔧 Core DeepAgents Resources
| Resource | Link | Purpose |
|----------|------|---------|
| **DeepAgents Python Repo** | [github.com/langchain-ai/deepagents](https://github.com/langchain-ai/deepagents) | Main implementation, README with examples |
| **DeepAgents Documentation** | [docs.langchain.com/labs/deep-agents/overview](https://docs.langchain.com/labs/deep-agents/overview) | Complete API reference and concepts |
| **DeepAgents Quickstart** | [docs.langchain.com/labs/deep-agents/quickstart](https://docs.langchain.com/labs/deep-agents/quickstart) | Get started in 5 minutes |
| **Built-in Components** | [docs.langchain.com/labs/deep-agents/built-in-components](https://docs.langchain.com/labs/deep-agents/built-in-components) | Planning tools, virtual FS, sub-agents |
| **DeepAgents JavaScript** | [github.com/langchain-ai/deepagentsjs](https://github.com/langchain-ai/deepagentsjs) | TypeScript/JavaScript implementation |

### 🔄 LangGraph Integration
| Resource | Link | Purpose |
|----------|------|---------|
| **LangGraph Persistence** | [langchain-ai.github.io/langgraph/concepts/persistence/](https://langchain-ai.github.io/langgraph/concepts/persistence/) | Checkpointers, threads, state management |
| **Human-in-the-Loop** | [langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/add-human-in-the-loop/](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/add-human-in-the-loop/) | Interrupts, approvals, HIL patterns |
| **Add Memory to Agents** | [docs.langchain.com/oss/python/langgraph/add-memory](https://docs.langchain.com/oss/python/langgraph/add-memory) | Memory management best practices |
| **LangGraph Studio** | [langchain-ai.github.io/langgraph/concepts/langgraph_studio/](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) | Visual debugging and monitoring |

### 🎯 Quick Navigation by Use Case

**🚀 Getting Started**
1. [DeepAgents Quickstart](https://docs.langchain.com/labs/deep-agents/quickstart) - 5-minute setup
2. [Create Deep Agent API](https://docs.langchain.com/labs/deep-agents/quickstart#create-deep-agent) - Basic factory pattern
3. [Sub-agent Configuration](https://docs.langchain.com/labs/deep-agents/built-in-components#subagents) - Specialization setup

**🛠️ Building Production Systems**
1. [Virtual Filesystem](https://docs.langchain.com/labs/deep-agents/built-in-components#virtual-filesystem) - Artifact management
2. [Checkpointers Setup](https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpointers) - State persistence
3. [Interrupt Configuration](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/add-human-in-the-loop/#interrupt-before-tools) - Human approvals

**🔍 Debugging & Monitoring**
1. [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) - Visual workflow debugging
2. [Streaming Responses](https://langchain-ai.github.io/langgraph/how-tos/streaming/) - Real-time monitoring
3. [Error Handling Patterns](https://langchain-ai.github.io/langgraph/how-tos/error-handling/) - Graceful degradation

### 📋 Implementation Checklist with Links

**Phase 1: Setup**
- [ ] Install DeepAgents: `pip install deepagents` ([Installation Guide](https://docs.langchain.com/labs/deep-agents/quickstart#installation))
- [ ] Configure tools: ([Tool Configuration](https://docs.langchain.com/labs/deep-agents/built-in-components#tools))
- [ ] Set up checkpointer: ([Persistence Setup](https://langchain-ai.github.io/langgraph/concepts/persistence/#setup))

**Phase 2: Architecture**
- [ ] Define sub-agents: ([Sub-agent Patterns](https://docs.langchain.com/labs/deep-agents/built-in-components#subagents))
- [ ] Configure virtual FS: ([Virtual Filesystem](https://docs.langchain.com/labs/deep-agents/built-in-components#virtual-filesystem))
- [ ] Set up planning tool: ([Planning Tool](https://docs.langchain.com/labs/deep-agents/built-in-components#planning-tool))

**Phase 3: Production**
- [ ] Configure HIL: ([Human-in-the-Loop](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/add-human-in-the-loop/))
- [ ] Set up monitoring: ([LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/))
- [ ] Deploy with FastAPI: ([Deployment Guide](https://langchain-ai.github.io/langgraph/concepts/deployment/))

---

## 💡 Why These Links Matter

### **DeepAgents GitHub Repo**
The single source of truth. Contains:
- Working code examples
- Latest architecture patterns
- Issue discussions for edge cases
- Integration examples with LangGraph

### **DeepAgents Documentation**
Comprehensive guide covering:
- `create_deep_agent()` factory function
- Sub-agent configuration schemas
- Built-in tools (planning, virtual FS)
- Model configuration options

### **LangGraph Persistence**
Critical for production deployment:
- Thread-based state management
- Checkpoint/resume capabilities
- Memory optimization patterns
- Database integration options

### **Human-in-the-Loop Guides**
Essential for sensitive operations:
- Tool approval workflows
- Interrupt configuration
- Manual review processes
- Approval UI patterns

---

*📌 **Bookmark this section** - Reference these links before any major DeepAgents implementation decisions.*

*🔄 **Keep updated** - DeepAgents is actively developed. Check GitHub for latest patterns.*