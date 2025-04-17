import json
import os
import re
import sys
import yaml
from typing import List, Optional, Union, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM
import logging
from dataclasses import dataclass
from enum import Enum
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Constants
MODEL_PATH = "tuandunghcmut/Qwen25_Coder_MultipleChoice_v4"
BASE_MODEL_PATH = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Setup logging
def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with consistent formatting and configuration.

    Args:
        name: Name of the logger (typically __name__)
        level: Optional logging level (defaults to INFO)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)

    # Set level if specified
    if level is not None:
        logger.setLevel(level)
    elif logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)

    # Only add handler if logger doesn't already have handlers
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Set formatter for handler
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

    return logger

logger = get_logger(__name__)

# Enums and Data Classes
class ModelSource(str, Enum):
    """Model source enumeration"""
    HUGGINGFACE = "huggingface"

@dataclass
class HubConfig:
    """Configuration for Hugging Face Hub integration"""
    model_id: str
    token: Optional[str] = None
    private: bool = False
    save_method: str = "lora"  # lora, merged_16bit, merged_4bit, gguf

# Pydantic Models for API
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
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True

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

# Model Handler Classes
class PromptCreator:
    """
    Creates and formats prompts for multiple choice questions
    Supports different prompt styles for training and inference
    """

    # Prompt types
    BASIC = "basic"  # Simple answer-only format
    YAML_REASONING = "yaml"  # YAML formatted reasoning
    TEACHER_REASONED = (
        "teacher"  # Same YAML format as YAML_REASONING but using teacher completions for training
    )
    OPTIONS = "options"  # Includes only lettered options in prompt

    VALID_PROMPT_TYPES = [
        BASIC,
        YAML_REASONING,
        TEACHER_REASONED,
        OPTIONS,
    ]

    def __init__(self, prompt_type: str = BASIC):
        """
        Initialize with specified prompt type

        Args:
            prompt_type: Type of prompt to use

        Raises:
            ValueError: If prompt_type is not one of the valid types
        """
        if prompt_type not in self.VALID_PROMPT_TYPES:
            raise ValueError(
                f"Invalid prompt type: {prompt_type}. Must be one of {self.VALID_PROMPT_TYPES}"
            )

        # For prompt formatting, teacher_reasoned is equivalent to yaml_reasoning
        # The difference only matters during training when using teacher completions
        if prompt_type == self.TEACHER_REASONED:
            prompt_type = self.YAML_REASONING

        self.prompt_type = prompt_type
        # Store the original prompt type to track if we're using teacher mode
        self.original_type = prompt_type

    def format_choices(self, choices: Union[List[str], str]) -> str:
        """
        Format choices into a string

        Args:
            choices: List of choices or pre-formatted string

        Returns:
            Formatted string of choices

        Raises:
            ValueError: If choices is empty or invalid
        """
        if not choices:
            raise ValueError("Choices cannot be empty")

        if isinstance(choices, str):
            return choices

        if not isinstance(choices, list):
            raise ValueError(f"Choices must be a list or string, got {type(choices)}")

        if not all(isinstance(choice, str) for choice in choices):
            raise ValueError("All choices must be strings")

        return "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))

    def get_max_letter(self, choices: Union[List[str], str]) -> str:
        """
        Get the maximum letter for the given number of choices

        Args:
            choices: List of choices or pre-formatted string

        Returns:
            Maximum letter (A, B, C, etc.)

        Raises:
            ValueError: If choices is empty or invalid
        """
        if not choices:
            raise ValueError("Choices cannot be empty")

        if isinstance(choices, str):
            # Try to count the number of lines in the formatted string
            num_choices = len([line for line in choices.split("\n") if line.strip()])
            if num_choices == 0:
                raise ValueError("No valid choices found in string")
            return chr(64 + num_choices)

        if not isinstance(choices, list):
            raise ValueError(f"Choices must be a list or string, got {type(choices)}")

        if not all(isinstance(choice, str) for choice in choices):
            raise ValueError("All choices must be strings")

        return chr(64 + len(choices))

    def create_inference_prompt(self, question: str, choices: Union[List[str], str]) -> str:
        """
        Create a prompt for inference

        Args:
            question: The question text
            choices: List of choices or pre-formatted string

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If question or choices are empty or invalid
        """
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")

        formatted_choices = self.format_choices(choices)
        max_letter = self.get_max_letter(choices)

        # Basic prompt types
        if self.prompt_type == self.BASIC:
            return self._create_basic_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type in [self.YAML_REASONING, self.TEACHER_REASONED]:
            return self._create_yaml_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.OPTIONS:
            return self._create_options_prompt(question, formatted_choices, max_letter)
        else:
            raise ValueError(f"Unknown prompt type: {self.prompt_type}")

    def _create_basic_prompt(self, question: str, formatted_choices: str, max_letter: str) -> str:
        """Create a basic prompt that only asks for the answer"""
        return f"""Question: {question}

Choices:
{formatted_choices}

Answer with a single letter from A through {max_letter} without any additional explanation or commentary."""

    def _create_yaml_prompt(self, question: str, formatted_choices: str, max_letter: str) -> str:
        """Create a YAML-formatted prompt that asks for reasoning"""
        return f"""Question: {question}

Choices:
{formatted_choices}

Analyze this question step-by-step and provide a detailed explanation.
Your response MUST be in YAML format as follows:

understanding: |
  <your understanding of what the question is asking>
analysis: |
  <your analysis of each option>
reasoning: |
  <your step-by-step reasoning process>
conclusion: |
  <your final conclusion>
answer: <single letter A through {max_letter}>

The answer field MUST contain ONLY a single character letter."""

    def _create_options_prompt(self, question: str, formatted_choices: str, max_letter: str) -> str:
        """Create a prompt that focuses on lettered options"""
        return f"""Question: {question}

Choices:
{formatted_choices}

Please select the best answer from the options above. Provide a brief explanation for your choice and clearly state the letter of your answer (A through {max_letter})."""

    def set_prompt_type(self, prompt_type: str) -> "PromptCreator":
        """
        Set the prompt type

        Args:
            prompt_type: Type of prompt to use (BASIC, YAML_REASONING, or TEACHER_REASONED)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If prompt_type is not one of the valid types
        """
        if prompt_type not in self.VALID_PROMPT_TYPES:
            raise ValueError(
                f"Invalid prompt type: {prompt_type}. Must be one of {self.VALID_PROMPT_TYPES}"
            )

        # Store the original type
        self.original_type = prompt_type

        # For prompt formatting, teacher_reasoned is equivalent to yaml_reasoning
        if prompt_type == self.TEACHER_REASONED:
            prompt_type = self.YAML_REASONING

        self.prompt_type = prompt_type
        return self

    def is_teacher_mode(self) -> bool:
        """Check if using teacher-reasoned mode"""
        return self.original_type == self.TEACHER_REASONED

class ResponseParser:
    """
    Parser for model responses with support for different formats
    Extracts answers and reasoning from model outputs
    """

    # Parser modes
    BASIC = "basic"  # Extract single letter answer
    YAML = "yaml"  # Parse YAML formatted response with reasoning

    def __init__(self, parser_mode: str = BASIC):
        """
        Initialize with specified parser mode

        Args:
            parser_mode: Mode of parsing to use (BASIC or YAML)
        """
        if parser_mode not in [self.BASIC, self.YAML]:
            raise ValueError(f"Unknown parser mode: {parser_mode}")
        self.parser_mode = parser_mode

    def parse(self, response_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse the response text to extract answer and reasoning

        Args:
            response_text: Raw response text from the model

        Returns:
            Tuple of (answer, reasoning)
        """
        if not response_text:
            return None, None

        if self.parser_mode == self.BASIC:
            return self._parse_basic_response(response_text)
        elif self.parser_mode == self.YAML:
            return self._parse_yaml_response(response_text)
        else:
            raise ValueError(f"Unknown parser mode: {self.parser_mode}")

    def _parse_basic_response(self, response_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse basic response format (just the answer)"""
        # Clean up the response
        response_text = response_text.strip()

        # Try to find a single letter answer
        answer_match = re.search(r"^[A-Za-z]$", response_text)
        if answer_match:
            return answer_match.group(0).upper(), None

        # Try to find answer after "Answer:" or similar
        answer_match = re.search(r"(?:answer|Answer):\s*([A-Za-z])", response_text)
        if answer_match:
            return answer_match.group(1).upper(), None

        # Try to find any single letter in the response
        answer_match = re.search(r"[A-Za-z]", response_text)
        if answer_match:
            return answer_match.group(0).upper(), None

        return None, None

    def _parse_yaml_response(self, response_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse YAML-formatted response with reasoning"""
        # Clean up the response
        response_text = response_text.strip()

        # Remove any markdown code block markers
        response_text = re.sub(r"```yaml\s*", "", response_text)
        response_text = re.sub(r"```\s*", "", response_text)

        try:
            # Try to parse as YAML
            yaml_content = yaml.safe_load("---\n" + response_text)
            if isinstance(yaml_content, dict):
                answer = yaml_content.get("answer")
                reasoning = self._extract_reasoning_from_yaml(yaml_content)

                # Clean up answer if needed
                if answer:
                    answer = answer.strip().upper()
                    if len(answer) > 1:
                        # Extract first letter if multiple characters
                        answer = answer[0]

                return answer, reasoning
        except yaml.YAMLError:
            # If YAML parsing fails, try to extract using regex
            answer_match = re.search(r"answer:\s*([A-Za-z])", response_text)
            reasoning_match = re.search(r"reasoning:\s*\|\s*([\s\S]+?)(?:\n\w+:|$)", response_text)

            answer = answer_match.group(1).upper() if answer_match else None
            reasoning = reasoning_match.group(1).strip() if reasoning_match else None

            return answer, reasoning

        return None, None

    def _extract_reasoning_from_yaml(self, yaml_content: Dict[str, Any]) -> Optional[str]:
        """Extract and format reasoning from YAML content"""
        reasoning_parts = []

        # Add understanding if present
        if "understanding" in yaml_content:
            reasoning_parts.append(f"Understanding:\n{yaml_content['understanding']}")

        # Add analysis if present
        if "analysis" in yaml_content:
            reasoning_parts.append(f"Analysis:\n{yaml_content['analysis']}")

        # Add reasoning if present
        if "reasoning" in yaml_content:
            reasoning_parts.append(f"Reasoning:\n{yaml_content['reasoning']}")

        # Add conclusion if present
        if "conclusion" in yaml_content:
            reasoning_parts.append(f"Conclusion:\n{yaml_content['conclusion']}")

        return "\n\n".join(reasoning_parts) if reasoning_parts else None

    def set_parser_mode(self, parser_mode: str) -> "ResponseParser":
        """Set the parser mode"""
        if parser_mode not in [self.BASIC, self.YAML]:
            raise ValueError(f"Unknown parser mode: {parser_mode}")
        self.parser_mode = parser_mode
        return self

    @classmethod
    def from_prompt_type(cls, prompt_type: str) -> "ResponseParser":
        """
        Create a ResponseParser instance from a prompt type

        Args:
            prompt_type: Type of prompt (from PromptCreator)

        Returns:
            ResponseParser instance with appropriate mode
        """
        if prompt_type == PromptCreator.BASIC:
            return cls(cls.BASIC)
        elif prompt_type in [PromptCreator.YAML_REASONING, PromptCreator.TEACHER_REASONED]:
            return cls(cls.YAML)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

class QwenModelHandler:
    """Handles loading, configuration, and inference with Qwen models"""

    def __init__(
        self,
        model_name: str,
        base_model_name: str,
        max_seq_length: int = 2048,
        model_source: str = ModelSource.HUGGINGFACE,
        device_map: str = "cpu",  # Default to CPU
        source_hub_config: Optional[HubConfig] = None,
        destination_hub_config: Optional[HubConfig] = None,
    ):
        """
        Initialize a Qwen model handler.

        Args:
            model_name: Name or path of the model to load
            base_model_name: Name or path of the base model to load
            max_seq_length: Maximum sequence length for tokenizer and model
            model_source: Source of the model (huggingface or unsloth)
            device_map: Device mapping strategy for the model (default: "cpu")
            source_hub_config: Configuration for the source model on Hugging Face Hub
            destination_hub_config: Configuration for the destination model on Hugging Face Hub
        """
        self.model_name = model_name
        self.base_model_name = base_model_name
        self.max_seq_length = max_seq_length
        self.model_source = model_source
        self.device_map = device_map
        self.source_hub_config = source_hub_config
        self.destination_hub_config = destination_hub_config

        # Initialize model and tokenizer
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

        # Log model configuration
        logger.info(f"Loading {model_name} from {model_source}, max_seq_length={max_seq_length}, device={device_map}")

        # Load the model based on the source
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer based on the specified source"""
        try:
            self._load_from_huggingface()

            # Ensure tokenizer has pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Log model info
            logger.info(f"Model loaded successfully: {self.model_name}")
            if hasattr(self.model, "config"):
                logger.info(f"Model type: {self.model.config.model_type}")
                for key, value in self.model.config.to_dict().items():
                    if key in [
                        "hidden_size",
                        "intermediate_size",
                        "num_attention_heads",
                        "num_hidden_layers",
                        "torch_dtype",
                    ]:
                        logger.info(f"{key}: {value}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_from_huggingface(self):
        """Load model from HuggingFace Hub"""
        # Load base model
        logger.info(f"Loading base model: {self.base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map=self.device_map,
            trust_remote_code=True
        )
        
        # Load adapter model
        logger.info(f"Loading adapter model: {self.model_name}")
        self.model = PeftModel.from_pretrained(
            base_model, 
            self.model_name,
            device_map=self.device_map
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            token=self.source_hub_config.token if self.source_hub_config else None,
            trust_remote_code=True,
            padding_side="right",
            model_max_length=self.max_seq_length,
        )

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeated tokens
            do_sample: Whether to use sampling or greedy generation

        Returns:
            str: The generated text response
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the output
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        response = response[len(prompt) :].strip()
        return response

    def generate_with_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 768,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        min_p: float = 0.1,
        stream: bool = True,
    ):
        """
        Generate a response from the model with streaming support.

        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeated tokens
            do_sample: Whether to use sampling or greedy generation
            min_p: Minimum probability for sampling (recommended 0.1)
            stream: Whether to stream the output or return the full response

        Returns:
            If stream=True: TextIteratorStreamer object that yields tokens as they're generated
            If stream=False: Complete response as string
        """
        import threading
        from transformers import TextIteratorStreamer

        # Format the prompt using chat template
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        # Create attention mask
        attention_mask = torch.ones_like(inputs)

        if stream:
            # Use TextIteratorStreamer for streaming output
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            # Generation args
            generation_args = {
                "input_ids": inputs,
                "attention_mask": attention_mask,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "min_p": min_p,
                "use_cache": True,
            }

            # Start generation in a separate thread
            thread = threading.Thread(target=self.model.generate, kwargs=generation_args)
            thread.start()

            # Return the streamer object
            return streamer
        else:
            # Generate without streaming
            outputs = self.model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                min_p=min_p,
                use_cache=True,
            )

            # Decode the output
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response
            prompt_text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
            response = response[len(prompt_text) :].strip()

            return response

# Create FastAPI app
app = FastAPI(
    title="Qwen Coder MCQ API",
    description="API for answering multiple-choice coding questions with step-by-step reasoning",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model handler
model_handler = None
prompt_creator = None
response_parser = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model and related components on startup"""
    global model_handler, prompt_creator, response_parser
    
    try:
        # Initialize model handler
        model_handler = QwenModelHandler(
            model_name=MODEL_PATH,
            base_model_name=BASE_MODEL_PATH,
            max_seq_length=2048,
            device_map="cpu",  # Use CPU by default
            model_source="huggingface",
        )
        
        # Initialize prompt creator and response parser
        prompt_creator = PromptCreator(prompt_type=PromptCreator.YAML_REASONING)
        response_parser = ResponseParser.from_prompt_type(prompt_creator.prompt_type)
        
        logger.info("Model and components initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import time
    return {
        "status": "ok",
        "timestamp": time.time(),
        "version": "1.0.0",
        "model_loaded": model_handler is not None
    }

def format_choices_for_prompt(choices: MCQChoice) -> str:
    """Format MCQ choices for the prompt"""
    choices_list = []
    for key, value in choices.dict(exclude_none=True).items():
        choices_list.append(f"{key}. {value}")
    
    return "\n".join(choices_list)

@app.post("/api/v1/mcq/answer", response_model=MCQResponse)
async def answer_mcq(request: MCQRequest):
    """Answer a multiple-choice question with reasoning"""
    if model_handler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format the prompt
        choices_text = format_choices_for_prompt(request.choices)
        prompt = prompt_creator.create_inference_prompt(request.question, choices_text)
        
        # Generate response
        max_len = request.max_length or 2048
        response = model_handler.generate_response(
            prompt=prompt,
            max_new_tokens=max_len,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
        )
        
        # Parse the response
        answer, reasoning = response_parser.parse(response)
        
        if not answer:
            raise HTTPException(status_code=500, detail="Failed to extract answer from model response")
        
        # Extract YAML content
        try:
            yaml_content = yaml.safe_load(response)
            if isinstance(yaml_content, dict):
                return MCQResponse(
                    understanding=yaml_content.get("understanding", ""),
                    analysis=yaml_content.get("analysis", ""),
                    reasoning=yaml_content.get("reasoning", ""),
                    conclusion=yaml_content.get("conclusion", ""),
                    answer=yaml_content.get("answer", "")
                )
            else:
                raise HTTPException(status_code=500, detail="Invalid YAML format in model response")
        except yaml.YAMLError:
            # If YAML parsing fails, try to extract using regex
            understanding_match = re.search(r"understanding:\s*\|\s*([\s\S]+?)(?:\n\w+:|$)", response)
            analysis_match = re.search(r"analysis:\s*\|\s*([\s\S]+?)(?:\n\w+:|$)", response)
            reasoning_match = re.search(r"reasoning:\s*\|\s*([\s\S]+?)(?:\n\w+:|$)", response)
            conclusion_match = re.search(r"conclusion:\s*\|\s*([\s\S]+?)(?:\n\w+:|$)", response)
            answer_match = re.search(r"answer:\s*([A-Za-z])", response)
            
            return MCQResponse(
                understanding=understanding_match.group(1).strip() if understanding_match else "",
                analysis=analysis_match.group(1).strip() if analysis_match else "",
                reasoning=reasoning_match.group(1).strip() if reasoning_match else "",
                conclusion=conclusion_match.group(1).strip() if conclusion_match else "",
                answer=answer_match.group(1).strip() if answer_match else ""
            )
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

async def generate_stream(prompt, max_len, temperature, top_p, top_k, repetition_penalty, do_sample):
    """Generate streaming response"""
    try:
        streamer = model_handler.generate_with_streaming(
            prompt=prompt,
            max_new_tokens=max_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            min_p=0.1,
            stream=True,
        )
        
        partial_text = ""
        for chunk in streamer:
            if chunk:
                partial_text += chunk
                try:
                    # Try to parse YAML from partial text
                    yaml_content = yaml.safe_load("---\n" + partial_text)
                    if isinstance(yaml_content, dict) and "answer" in yaml_content:
                        yield f"data: {json.dumps(yaml_content)}\n\n"
                except yaml.YAMLError:
                    # YAML is not complete yet, continue
                    pass
        
        # Send final result
        try:
            yaml_content = yaml.safe_load("---\n" + partial_text)
            if isinstance(yaml_content, dict):
                yield f"data: {json.dumps(yaml_content)}\n\n"
        except yaml.YAMLError:
            pass
        
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error in streaming: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

@app.post("/api/v1/mcq/explain")
async def explain_mcq(request: MCQRequest):
    """Answer MCQ with streaming or non-streaming response"""
    if model_handler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Format the prompt
    choices_text = format_choices_for_prompt(request.choices)
    prompt = prompt_creator.create_inference_prompt(request.question, choices_text)
    
    max_len = request.max_length or 2048
    
    if request.streaming:
        return StreamingResponse(
            generate_stream(
                prompt=prompt,
                max_len=max_len,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
            ),
            media_type="text/event-stream"
        )
    else:
        return await answer_mcq(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8000, reload=True) 