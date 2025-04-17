import json
import os
import re
import sys
from typing import List, Optional, Union
import gradio as gr
import spaces
import torch

import yaml

MODEL_PATH = "tuandunghcmut/Qwen25_Coder_MultipleChoice_v4"


"""
Contains 200 example coding multiple choice questions for the demo application,
organized by category.
"""

# Define the examples by category
CODING_EXAMPLES_BY_CATEGORY = {
    "Python": [
        {
            "question": "Which of the following is NOT a valid way to iterate through a list in Python?",
            "choices": [
                "for item in my_list:",
                "for i in range(len(my_list)):",
                "for index, item in enumerate(my_list):",
                "for item from my_list:",
            ],
            "answer": "D",
        },
        {
            "question": "In Python, what does the `__str__` method do?",
            "choices": [
                "Returns a string representation of an object for developers",
                "Returns a string representation of an object for end users",
                "Converts a string to an object",
                "Checks if an object is a string",
            ],
            "answer": "B",
        },
    ]}

# Flatten the examples for easy access by index
CODING_EXAMPLES = []
for category, examples in CODING_EXAMPLES_BY_CATEGORY.items():
    for example in examples:
        example["category"] = category
        CODING_EXAMPLES.append(example)


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

    def create_training_prompt(self, question: str, choices: Union[List[str], str]) -> str:
        """
        Create a prompt for training

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
            return self._create_basic_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type in [self.YAML_REASONING, self.TEACHER_REASONED]:
            return self._create_yaml_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.OPTIONS:
            return self._create_options_training_prompt(question, formatted_choices, max_letter)
        else:
            raise ValueError(f"Unknown prompt type: {self.prompt_type}")

    def _create_basic_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a basic training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

The answer is a single letter (A, B, C, etc.). Only provide ONE character as your answer:"""

    def _create_yaml_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a YAML-formatted training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

Analyze this question step-by-step and provide a detailed explanation.
Follow the YAML format in your response:

understanding: |
  <your understanding of the question>
analysis: |
  <your analysis of each option>
reasoning: |
  <your reasoning about the correct answer>
conclusion: |
  <your final conclusion>
answer: <single letter A through {max_letter}>"""

    def _create_options_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a training prompt for options format"""
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


import re
from typing import Any, Dict, Optional, Tuple

import yaml

try:
    from .prompt_creator import PromptCreator
except ImportError:
    pass


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


import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers import BitsAndBytesConfig
import logging
import sys
from typing import Optional


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


class ModelSource(str, Enum):
    """Model source enumeration"""

    HUGGINGFACE = "huggingface"
    # UNSLOTH = "unsloth"


@dataclass
class HubConfig:
    """Configuration for Hugging Face Hub integration"""

    model_id: str
    token: Optional[str] = None
    private: bool = False
    save_method: str = "lora"  # lora, merged_16bit, merged_4bit, gguf

# @spaces.GPU()
def create_model():
    class QwenModelHandler:
        """Handles loading, configuration, and inference with Qwen models"""

        HUGGINGFACE = "huggingface"
        # UNSLOTH = "unsloth"

        # @spaces.GPU()
        def __init__(
            self,
            model_name: str,
            max_seq_length: int = 2048,
            # quantization: Union[str, BitsAndBytesConfig] = "4bit",
            quantization: Union[str, BitsAndBytesConfig, None] = None,
            # quantization: Union[str, None] = None,

            model_source: str = ModelSource.HUGGINGFACE,
            device_map: str = "cuda",
            source_hub_config: Optional[HubConfig] = None,
            destination_hub_config: Optional[HubConfig] = None,
            # attn_implementation: str = "default",
            # force_attn_implementation: bool = False,
        ):
            """
            Initialize a Qwen model handler.

            Args:
                model_name: Name or path of the model to load
                max_seq_length: Maximum sequence length for tokenizer and model
                quantization: Quantization level (4bit, 8bit, or none) or BitsAndBytesConfig object
                model_source: Source of the model (huggingface or unsloth)
                device_map: Device mapping strategy for the model
                source_hub_config: Configuration for the source model on Hugging Face Hub
                destination_hub_config: Configuration for the destination model on Hugging Face Hub
                # attn_implementation: Attention implementation to use (default, flash_attention_2, sdpa, eager, xformers)
                # force_attn_implementation: Whether to force the attention implementation even if not optimal
            """
            self.model_name = model_name
            self.max_seq_length = max_seq_length
            self.quantization = quantization
            self.model_source = model_source
            self.device_map = device_map
            self.source_hub_config = source_hub_config
            self.destination_hub_config = destination_hub_config
            # self.attn_implementation = attn_implementation
            # self.force_attn_implementation = force_attn_implementation

            # Initialize model and tokenizer
            self.model: Optional[PreTrainedModel] = None
            self.tokenizer: Optional[PreTrainedTokenizer] = None

            # Log model configuration
            logger.info(f"Loading {model_name} from {model_source}, max_seq_length={max_seq_length}")

            # Load the model based on the source
            self._load_model()

        # @spaces.GPU()
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

        # @spaces.GPU()
        def _load_from_huggingface(self):
            """Load model from HuggingFace Hub"""
            # Configure quantization


            # NOTE: I don't know why bitsandbytes has tons of bugs with the spaces.
            # quantization_config = None
            if isinstance(self.quantization, str):
                if self.quantization == "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                elif self.quantization == "8bit":
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif isinstance(self.quantization, BitsAndBytesConfig):
                quantization_config = self.quantization

            # Check attention implementation
            # attn_implementation = self._check_attention_support()

            # NOTE: This is hard coded.
            attn_implementation = 'sdpa'

            model_kwargs = {
                "device_map": self.device_map,
                "token": self.source_hub_config.token if self.source_hub_config else None,
                "trust_remote_code": True,
            }

            #########################################################################################
            # NOTE: Code for disable quantization
            # quantization_config = None
            #########################################################################################


            # NOTE: Code for disable quantization
            # Add quantization config if specified
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

            # Add attention implementation if not default
            if attn_implementation != "default":
                model_kwargs["attn_implementation"] = attn_implementation
                logger.info(f"Using attention implementation: {attn_implementation}")

            # Import AutoModelForCausalLM here to avoid early import
            from transformers import AutoModelForCausalLM

            # Load model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16,
                # attn_implementation='sdpa',
                **model_kwargs
            ).to('cuda')

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
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
            # inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

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

        # @spaces.GPU()
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
            ).to('cuda')
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

        def calculate_perplexity(self, prompt: str, answer: str, temperature: float = 0.0) -> float:
            """
            Calculate perplexity of the given answer for a prompt.

            Args:
                prompt: The input prompt
                answer: The answer to evaluate
                temperature: Sampling temperature

            Returns:
                float: Perplexity score (lower is better)
            """
            import math

            # Combine prompt and answer
            full_text = prompt + answer

            # Tokenize
            encodings = self.tokenizer(full_text, return_tensors="pt")
            input_ids = encodings.input_ids.to(self.model.device)
            target_ids = input_ids.clone()

            # Determine where the answer starts
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

            prompt_length = prompt_ids.shape[1]

            # Set prompt part to -100 so it's ignored in loss calculation
            target_ids[:, :prompt_length] = -100

            # Calculate loss
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss.item()

            # Count tokens in answer
            answer_length = target_ids.shape[1] - prompt_length

            # Calculate perplexity: exp(average negative log-likelihood)
            perplexity = math.exp(neg_log_likelihood)

            return perplexity

        # @spaces.GPU()
        def calculate_answer_loss(self, prompt: str, answer: str) -> float:
            """
            Calculate the loss specifically on the answer portion of the text.

            Args:
                prompt: The input prompt
                answer: The answer to evaluate

            Returns:
                float: Loss value for the answer
            """
            # Combine prompt and answer
            full_text = prompt + answer

            # Tokenize
            encodings = self.tokenizer(full_text, return_tensors="pt")
            input_ids = encodings.input_ids.to(self.model.device)
            # input_ids = encodings.input_ids.to('cuda')

            target_ids = input_ids.clone()

            # Determine where the answer starts
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
            # prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
            prompt_length = prompt_ids.shape[1]

            # Set prompt part to -100 so it's ignored in loss calculation
            target_ids[:, :prompt_length] = -100

            # Calculate loss on answer only
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, labels=target_ids)
                loss = outputs.loss.item()

            return loss

        # @spaces.GPU()
        def save_to_hub(self, hub_config: HubConfig, merge_adapter: bool = False):
            """
            Save model to Hugging Face Hub.

            Args:
                hub_config: Configuration for Hub saving
                merge_adapter: Whether to merge the adapter weights before saving

            Returns:
                str: URL of the saved model on the Hub
            """
            try:
                logger.info(f"Saving model to {hub_config.model_id}...")

                # Create repository if needed
                if hub_config.token:
                    from huggingface_hub import create_repo

                    try:
                        create_repo(
                            hub_config.model_id, private=hub_config.private, token=hub_config.token
                        )
                        logger.info(f"Created repository: {hub_config.model_id}")
                    except Exception as e:
                        # Repository likely already exists
                        logger.info(f"Repository exists or couldn't be created: {str(e)}")

                # Save based on method
                if hub_config.save_method == "lora":
                    # Save LoRA adapter only
                    if hasattr(self.model, "peft_config"):
                        logger.info("Saving LoRA adapter...")
                        self.model.save_pretrained(
                            hub_config.model_id, token=hub_config.token, push_to_hub=True
                        )

                        # Save tokenizer
                        self.tokenizer.save_pretrained(
                            hub_config.model_id, token=hub_config.token, push_to_hub=True
                        )
                    else:
                        logger.warning("Model doesn't have LoRA adapter, saving full model...")
                        self.model.save_pretrained(
                            hub_config.model_id, token=hub_config.token, push_to_hub=True
                        )

                elif hub_config.save_method == "merged_16bit":
                    # Merge adapter and save in 16-bit
                    if hasattr(self.model, "merge_and_unload"):
                        logger.info("Merging adapter and saving in 16-bit...")
                        merged_model = self.model.merge_and_unload()
                        merged_model.save_pretrained(
                            hub_config.model_id, token=hub_config.token, push_to_hub=True
                        )

                        # Save tokenizer
                        self.tokenizer.save_pretrained(
                            hub_config.model_id, token=hub_config.token, push_to_hub=True
                        )
                    else:
                        logger.warning("Model doesn't support merge_and_unload, saving as is...")
                        self.model.save_pretrained(
                            hub_config.model_id, token=hub_config.token, push_to_hub=True
                        )

                elif hub_config.save_method == "merged_4bit":
                    # Create optimized 4-bit model
                    logger.info("Saving 4-bit quantized model is not fully supported yet")
                    logger.info("Falling back to standard saving...")
                    self.model.save_pretrained(
                        hub_config.model_id, token=hub_config.token, push_to_hub=True
                    )

                elif hub_config.save_method == "gguf":
                    logger.warning("GGUF export not yet supported, saving in standard format")
                    self.model.save_pretrained(
                        hub_config.model_id, token=hub_config.token, push_to_hub=True
                    )

                else:
                    raise ValueError(f"Unsupported save method: {hub_config.save_method}")

                # Generate model URL
                hf_hub_url = f"https://huggingface.co/{hub_config.model_id}"
                logger.info(f"Model saved successfully to {hf_hub_url}")

                return hf_hub_url

            except Exception as e:
                logger.error(f"Error saving model to Hub: {str(e)}")
                raise

        def save_model(self, output_dir: str, save_method: str = "lora") -> str:
            """
            Save model to disk

            Args:
                output_dir: Directory to save the model
                save_method: Method to use for saving ("lora", "merged_16bit", "merged_4bit", "gguf")

            Returns:
                Path to saved model
            """
            os.makedirs(output_dir, exist_ok=True)

            if self.model_source == ModelSource.UNSLOTH:
                # Use Unsloth's saving methods
                if save_method == "lora":
                    self.model.save_pretrained(output_dir)
                    self.tokenizer.save_pretrained(output_dir)
                elif save_method == "merged_16bit":
                    self.model.save_pretrained_merged(
                        output_dir, self.tokenizer, save_method="merged_16bit"
                    )
                elif save_method == "merged_4bit":
                    self.model.save_pretrained_merged(
                        output_dir, self.tokenizer, save_method="merged_4bit"
                    )
                elif save_method == "gguf":
                    self.model.save_pretrained_gguf(
                        output_dir, self.tokenizer, quantization_method="q4_k_m"
                    )
                else:
                    raise ValueError(f"Unknown save method: {save_method}")
            else:
                # Use Hugging Face's saving methods
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)

            print(f"Model saved to {output_dir} using method {save_method}")
            return output_dir

        def push_to_hub(self, hub_config: HubConfig) -> str:
            """
            Push model to Hugging Face Hub

            Args:
                hub_config: Configuration for pushing to HuggingFace Hub

            Returns:
                URL of the pushed model
            """
            if self.model_source == ModelSource.UNSLOTH:
                # Use Unsloth's hub methods
                if hub_config.save_method == "lora":
                    self.model.push_to_hub_merged(
                        hub_config.model_id, self.tokenizer, save_method="lora", token=hub_config.token
                    )
                elif hub_config.save_method == "merged_16bit":
                    self.model.push_to_hub_merged(
                        hub_config.model_id,
                        self.tokenizer,
                        save_method="merged_16bit",
                        token=hub_config.token,
                    )
                elif hub_config.save_method == "merged_4bit":
                    self.model.push_to_hub_merged(
                        hub_config.model_id,
                        self.tokenizer,
                        save_method="merged_4bit",
                        token=hub_config.token,
                    )
                elif hub_config.save_method == "gguf":
                    self.model.push_to_hub_gguf(
                        hub_config.model_id,
                        self.tokenizer,
                        quantization_method=["q4_k_m", "q5_k_m"],
                        token=hub_config.token,
                    )
                else:
                    raise ValueError(f"Unknown save method: {hub_config.save_method}")
            else:
                # Use Hugging Face's hub methods
                self.model.push_to_hub(
                    hub_config.model_id, token=hub_config.token, private=hub_config.private
                )
                self.tokenizer.push_to_hub(
                    hub_config.model_id, token=hub_config.token, private=hub_config.private
                )

            hub_url = f"https://huggingface.co/{hub_config.model_id}"
            print(f"Model successfully pushed to: {hub_url}")
            return hub_url


    # NOTE: Such a stupid way to load the model, but it works and not cause any error with spaces.GPU()
    """Load the model from Hugging Face Hub or local checkpoint"""
    model_handler = QwenModelHandler(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        # quantization=None,  # Disable quantization
        quantization='4bit',  # Disable quantization

        device_map="cuda",  # Automatically choose best device
        # attn_implementation="flash_attention_2",  # Use flash attention for better performance
        # force_attn_implementation=True,  # Force flash attention even if not optimal
        model_source="huggingface",  # Use Unsloth's optimized model
    )
    model_handler.model.to("cuda")
    return model_handler

# NOTE: First ugly code
MODEL_HANDLER = create_model()

"""Initialize the application with model"""
class TempObjectClass:
    def __init__(self):
        self.model_path = MODEL_PATH
        self.model_handler = MODEL_HANDLER
        self.prompt_creator = PromptCreator(prompt_type=PromptCreator.YAML_REASONING)
        self.response_parser = ResponseParser.from_prompt_type(self.prompt_creator.prompt_type)
        self.response_cache = {}  # Cache for model responses

# NOTE: Second ugly code   
self = TEMP_OBJECT = TempObjectClass()

# PROMPT_CREATOR = TEMP_OBJECT.prompt_creator

@spaces.GPU(duration=180)
def inference_fn(
    question,
    choices,
    temperature,
    max_new_tokens,
    top_p,
    top_k,
    repetition_penalty,
    do_sample,
):
    self = TEMP_OBJECT

    # """Run inference with the model"""
    if True:
        if True:

            try:
                print("\n=== Debug: Inference Process ===")
                print(f"Input Question: {question}")
                print(f"Input Choices: {choices}")

                # Create cache key
                cache_key = f"{question}|{choices}|{temperature}|{max_new_tokens}|{top_p}|{top_k}|{repetition_penalty}|{do_sample}"
                print(f"Cache Key: {cache_key}")

                # Check cache first
                if cache_key in self.response_cache:
                    print("Cache hit! Returning cached response")
                    return self.response_cache[cache_key]

                # Create the prompt using the standard format from prompt_creator
                print("\nCreating prompt with PromptCreator...")
                prompt = self.prompt_creator.create_inference_prompt(question, choices)
                print(f"Generated Prompt:\n{prompt}")

                # Get model response using streaming generation
                print("\nStarting streaming generation...")
                response_chunks = []

                # Get streamer object
                streamer = self.model_handler.generate_with_streaming(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    min_p=0.1,  # Recommended value for better generation
                    stream=True,
                )

                # Iterate through streaming chunks
                for chunk in streamer:
                    if chunk:  # Only append non-empty chunks
                        response_chunks.append(chunk)
                        # Yield partial response for real-time display
                        partial_response = "".join(response_chunks)
                        # Format partial response for display
                        formatted_response = f"""
Question: {question}

Choices:
{choices}

Model Completion:
{partial_response}"""

                        # Yield to Gradio for display
                        yield prompt, formatted_response, "", ""

                # Combine all chunks for final response
                response = "".join(response_chunks)
                print(f"Complete Model Response:\n{response}")

                # Format the final response
                final_response = f"""Question: {question}

    Choices:
    {choices}

    {response}"""

                # Parse YAML for structured display
                yaml_raw_display = f"```yaml\n{response}\n```"

                try:
                    # Try to parse the YAML
                    yaml_data = yaml.safe_load(response)
                    yaml_json_display = f"```json\n{json.dumps(yaml_data, indent=2)}\n```"
                except Exception as e:
                    print(f"Error parsing YAML: {e}")
                    yaml_json_display = (
                        f"**Error parsing YAML:** {str(e)}\n\n**Raw Response:**\n```\n{response}\n```"
                    )

                print("\nFinal Formatted Response:")
                print(final_response)

                result = (prompt, final_response, yaml_raw_display, yaml_json_display)

                # Cache the result
                self.response_cache[cache_key] = result
                print("\nCached result for future use")

                # Yield final response with structured YAML
                yield result

            except Exception as e:
                print(f"\nError during inference: {e}")
                # Format error response in YAML format
                error_response = f"""There are some bugs in streaming model response"""
                yield prompt, error_response, "", ""



def process_example(example_idx):
    """Process an example from the preset list"""
    if example_idx is None:
        return "", ""

    # Convert string index to integer if needed
    if isinstance(example_idx, str):
        try:
            # Extract the number from the string (e.g., "Example 13: ..." -> 13)
            example_idx = int(example_idx.split(":")[0].split()[-1]) - 1
        except (ValueError, IndexError) as e:
            print(f"Error converting example index: {e}")
            return "", ""

    try:
        if not isinstance(example_idx, int):
            print(f"Invalid example index type: {type(example_idx)}")
            return "", ""

        if example_idx < 0 or example_idx >= len(CODING_EXAMPLES):
            print(f"Example index out of range: {example_idx}")
            return "", ""

        example = CODING_EXAMPLES[example_idx]
        question = example["question"]
        choices = "\n".join(example["choices"])

        return question, choices

    except (ValueError, IndexError) as e:
        print(f"Error processing example: {e}")
        return "", ""

def get_category_examples(category_name):
    """Get examples for a specific category"""
    if category_name == "All Categories":
        choices = [f"Example {i+1}: {ex['question']}" for i, ex in enumerate(CODING_EXAMPLES)]
    elif category_name in CODING_EXAMPLES_BY_CATEGORY:
        # Find the starting index for this category in the flattened list
        start_idx = 0
        for cat, examples in CODING_EXAMPLES_BY_CATEGORY.items():
            if cat == category_name:
                break
            start_idx += len(examples)

        choices = [
            f"Example {start_idx+i+1}: {ex['question']}"
            for i, ex in enumerate(CODING_EXAMPLES_BY_CATEGORY[category_name])
        ]
    else:
        choices = []

    return gr.Dropdown(choices=choices, value=None, interactive=True)



# @spaces.GPU()
def create_interface_ui():
    """Create the Gradio interface"""
    with gr.Blocks(title="Coding Multiple Choice Q&A with YAML Reasoning") as interface:
        gr.Markdown("# Coding Multiple Choice Q&A with YAML Reasoning")
        gr.Markdown(
            """
        This app uses a fine-tuned Qwen2.5-Coder-1.5B model to answer multiple-choice coding questions with structured YAML reasoning.

        The model breaks down its thought process in a structured way, providing:
        - Understanding of the question
        - Analysis of all options
        - Detailed reasoning process
        - Clear conclusion
        """
        )

        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown(
                    "### Examples (from the bank of 200 high-quality MCQs by Claude 3.7 Sonnet)"
                )

                # Category selector
                category_dropdown = gr.Dropdown(
                    choices=["All Categories"] + list(CODING_EXAMPLES_BY_CATEGORY.keys()),
                    value="All Categories",
                    label="Select a category",
                )

                # Example selector
                example_dropdown = gr.Dropdown(
                    choices=[
                        f"Example {i+1}: {q['question']}" for i, q in enumerate(CODING_EXAMPLES)
                    ],
                    label="Select an example question",
                    value=None,
                )

                gr.Markdown("### Your Question (or you can manually enter your input)")

                # Question and choices inputs
                question_input = gr.Textbox(
                    label="Question", lines=3, placeholder="Enter your coding question here..."
                )
                choices_input = gr.Textbox(
                    label="Choices (one per line)",
                    lines=4,
                    placeholder="Enter each choice on a new line, e.g.:\nOption A\nOption B\nOption C\nOption D",
                )

                # Parameters
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.001,
                    step=0.005,
                    label="Temperature (higher = more creative, lower = more deterministic)",
                )

                # Additional generation parameters
                max_new_tokens_slider = gr.Slider(
                    minimum=128,
                    maximum=2048,
                    value=512,
                    step=128,
                    label="Max New Tokens (maximum length of generated response)",
                )

                top_p_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="Top-p (nucleus sampling probability)",
                )

                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=90,
                    step=1,
                    label="Top-k (number of highest probability tokens to consider)",
                )

                repetition_penalty_slider = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.1,
                    step=0.1,
                    label="Repetition Penalty (higher = less repetition)",
                )

                do_sample_checkbox = gr.Checkbox(
                    value=True,
                    label="Enable Sampling (unchecked for greedy generation)",
                )

                # Submit button
                submit_btn = gr.Button("Submit", variant="primary")

            with gr.Column(scale=6):
                gr.Markdown("### Model Input")
                prompt_display = gr.Textbox(
                    label="Prompt sent to model",
                    lines=8,
                    interactive=False,
                    show_copy_button=True,
                )

                gr.Markdown("### Model Streaming Response")
                output = gr.Textbox(label="Response")

                gr.Markdown("### Structured YAML Response")
                with gr.Accordion("", open=True):
                    gr.Markdown(
                        "Once the model completes its response, the YAML will be displayed here in a structured format."
                    )
                    yaml_raw = gr.Markdown(label="Raw YAML")
                    yaml_json = gr.Markdown(label="YAML as JSON")

        # Set up category selection
        category_dropdown.change(
            fn=get_category_examples,
            inputs=[category_dropdown],
            outputs=[example_dropdown],
        )

        # Set up example selection
        example_dropdown.change(
            fn=process_example,
            inputs=[example_dropdown],
            outputs=[question_input, choices_input],
        )

        # Update prompt display when question or choices change
        def update_prompt(question, choices):
            print("\n=== Debug: Prompt Update ===")
            print(f"Question Input: {question}")
            print(f"Choices Input: {choices}")

            if not question or not choices:
                print("Empty question or choices, returning empty prompt")
                return ""

            try:
                print("\nCreating prompt with PromptCreator...")
                prompt = self.prompt_creator.create_inference_prompt(question, choices)
                print(f"Generated Prompt:\n{prompt}")
                return prompt
            except Exception as e:
                print(f"Error creating prompt: {e}")
                return ""

        # Add prompt update on question/choices change
        question_input.change(
            fn=update_prompt, inputs=[question_input, choices_input], outputs=[prompt_display]
        )

        choices_input.change(
            fn=update_prompt, inputs=[question_input, choices_input], outputs=[prompt_display]
        )

        # Set up submission with loading indicator
        submit_btn.click(
            fn=inference_fn,
            inputs=[
                question_input,
                choices_input,
                temperature_slider,
                max_new_tokens_slider,
                top_p_slider,
                top_k_slider,
                repetition_penalty_slider,
                do_sample_checkbox,
            ],
            outputs=[prompt_display, output, yaml_raw, yaml_json],
            show_progress=True,  # Show progress bar
            queue=True,  # Enable queueing for better handling of multiple requests
        )

    return interface




# NOTE: Third ugly code
INTERFACE = create_interface_ui()
INTERFACE.queue().launch()


