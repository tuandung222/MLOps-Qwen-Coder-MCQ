# app/main.py
import os
import time
import yaml
import json
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Set up OpenTelemetry
resource = Resource(attributes={SERVICE_NAME: "qwen-coder-mcq-api"})
provider = TracerProvider(resource=resource)
jaeger_exporter = JaegerExporter(
    agent_host_name=os.getenv("JAEGER_HOST", "jaeger"),
    agent_port=int(os.getenv("JAEGER_PORT", "6831")),
)
processor = BatchSpanProcessor(jaeger_exporter)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# Model loading
MODEL_PATH = os.getenv("MODEL_PATH", "tuandunghcmut/Qwen25_Coder_MultipleChoice_v4")
DEVICE = os.getenv("DEVICE", "cuda:0")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "2048"))

# API Models
class MCQChoice(BaseModel):
    A: str
    B: str
    C: str
    D: Optional[str] = None
    E: Optional[str] = None

class MCQRequest(BaseModel):
    question: str
    choices: MCQChoice
    streaming: bool = False
    max_length: Optional[int] = None

class MCQResponse(BaseModel):
    understanding: str
    analysis: str
    reasoning: str
    conclusion: str
    answer: str

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    model_loaded: bool

# Application setup
app = FastAPI(
    title="Qwen Coder MCQ API",
    description="API for answering multiple-choice coding questions with step-by-step reasoning",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=[".*admin.*", "/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="inprogress",
    inprogress_labels=True,
)
instrumentator.add(
    metrics.request_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace="qwen",
        metric_subsystem="api",
    )
)
instrumentator.add(
    metrics.response_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace="qwen",
        metric_subsystem="api",
    )
)
instrumentator.add(
    metrics.latency(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace="qwen",
        metric_subsystem="api",
    )
)

# Global model and tokenizer
tokenizer = None
model = None
pipe = None

def format_prompt(question: str, choices: MCQChoice) -> str:
    """Format the prompt for the model."""
    choices_list = []
    for key, value in choices.dict(exclude_none=True).items():
        choices_list.append(f"{key}. {value}")
    
    choices_text = "\n".join(choices_list)
    
    prompt = f"""Question: {question}

Choices:
{choices_text}

Think through this step-by-step:
- Understand what the question is asking
- Analyze each option carefully
- Reason about why each option might be correct or incorrect
- Select the most appropriate answer

Your response MUST be in YAML format:
understanding: |
  <your understanding of the question>
analysis: |
  <your analysis of each option>
reasoning: |
  <your reasoning about the correct answer>
conclusion: |
  <your final conclusion>
answer: <single letter A through D>"""
    
    return prompt

@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    global tokenizer, model, pipe
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map=DEVICE,
            torch_dtype="auto",
            trust_remote_code=True
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
            do_sample=False,
            trust_remote_code=True
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        # We'll continue even if model fails to load, but API endpoints will return errors

# Register Prometheus metrics
instrumentator.instrument(app).expose(app, include_in_schema=True, should_gzip=True)

# Telemetry instrumentation
FastAPIInstrumentor.instrument_app(app)

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    with tracer.start_as_current_span("health_check"):
        return {
            "status": "ok",
            "timestamp": time.time(),
            "version": "1.0.0",
            "model_loaded": model is not None and tokenizer is not None
        }

@app.post("/api/v1/mcq/answer", response_model=MCQResponse)
async def answer_mcq(request: MCQRequest):
    """Answer a multiple-choice question with reasoning."""
    with tracer.start_as_current_span("answer_mcq") as span:
        span.set_attribute("question_length", len(request.question))
        
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            # Format the prompt
            prompt = format_prompt(request.question, request.choices)
            span.set_attribute("prompt_length", len(prompt))
            
            # Generate response
            max_len = request.max_length or MAX_LENGTH
            response = pipe(prompt, max_length=max_len)[0]['generated_text']
            
            # Extract the YAML part from the response
            yaml_part = response.split("Your response MUST be in YAML format:")[1].strip()
            result = yaml.safe_load(yaml_part)
            
            # Validate and return the response
            return MCQResponse(
                understanding=result.get("understanding", ""),
                analysis=result.get("analysis", ""),
                reasoning=result.get("reasoning", ""),
                conclusion=result.get("conclusion", ""),
                answer=result.get("answer", "")
            )
        except Exception as e:
            span.record_exception(e)
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

async def generate_stream(prompt, max_len):
    """Generate streaming response."""
    partial_text = ""
    result = None
    
    for response in pipe(prompt, max_length=max_len, streamer=True):
        partial_text += response
        
        try:
            if "Your response MUST be in YAML format:" in partial_text:
                yaml_part = partial_text.split("Your response MUST be in YAML format:")[1].strip()
                result = yaml.safe_load(yaml_part)
                if result and isinstance(result, dict) and "answer" in result:
                    yield f"data: {json.dumps(result)}\n\n"
        except yaml.YAMLError:
            # YAML is not complete yet, continue
            pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    # Send final result
    if result:
        yield f"data: {json.dumps(result)}\n\n"
    
    yield "data: [DONE]\n\n"

@app.post("/api/v1/mcq/explain")
async def explain_mcq(request: MCQRequest):
    """Answer MCQ with streaming or non-streaming response."""
    with tracer.start_as_current_span("explain_mcq") as span:
        span.set_attribute("question_length", len(request.question))
        span.set_attribute("streaming", request.streaming)
        
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Format the prompt
        prompt = format_prompt(request.question, request.choices)
        span.set_attribute("prompt_length", len(prompt))
        
        max_len = request.max_length or MAX_LENGTH
        
        if request.streaming:
            return StreamingResponse(
                generate_stream(prompt, max_len),
                media_type="text/event-stream"
            )
        else:
            return await answer_mcq(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)