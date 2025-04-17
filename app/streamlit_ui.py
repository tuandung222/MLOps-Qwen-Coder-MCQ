import json
import time
import streamlit as st
import requests
import yaml
from typing import Dict, List, Optional, Union

# Constants
API_URL = "http://localhost:8000"
MODEL_PATH = "tuandunghcmut/Qwen25_Coder_MultipleChoice_v4"

# Example questions
CODING_EXAMPLES_BY_CATEGORY = {
    "Python": [
        {
            "question": "Which of the following is NOT a valid way to iterate through a list in Python?",
            "choices": {
                "A": "for item in my_list:",
                "B": "for i in range(len(my_list)):",
                "C": "for index, item in enumerate(my_list):",
                "D": "for item from my_list:",
            },
            "answer": "D",
        },
        {
            "question": "In Python, what does the `__str__` method do?",
            "choices": {
                "A": "Returns a string representation of an object for developers",
                "B": "Returns a string representation of an object for end users",
                "C": "Converts a string to an object",
                "D": "Checks if an object is a string",
            },
            "answer": "B",
        },
    ]
}

# Flatten the examples for easy access by index
CODING_EXAMPLES = []
for category, examples in CODING_EXAMPLES_BY_CATEGORY.items():
    for example in examples:
        example["category"] = category
        CODING_EXAMPLES.append(example)

def check_api_health():
    """Check if the API is running and healthy"""
    try:
        response = requests.get(f"{API_URL}/api/v1/health")
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API returned status code {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to API. Make sure the FastAPI server is running."

def format_choices_for_display(choices: Dict[str, str]) -> str:
    """Format choices for display in the UI"""
    return "\n".join([f"{key}. {value}" for key, value in choices.items()])

def process_example(example_idx: int) -> tuple:
    """Process an example from the preset list"""
    if example_idx < 0 or example_idx >= len(CODING_EXAMPLES):
        return "", ""
    
    example = CODING_EXAMPLES[example_idx]
    question = example["question"]
    choices = format_choices_for_display(example["choices"])
    
    return question, choices

def get_category_examples(category_name: str) -> List[Dict]:
    """Get examples for a specific category"""
    if category_name == "All Categories":
        return CODING_EXAMPLES
    elif category_name in CODING_EXAMPLES_BY_CATEGORY:
        return [ex for ex in CODING_EXAMPLES if ex["category"] == category_name]
    else:
        return []

def call_api(question: str, choices: Dict[str, str], streaming: bool = False, **kwargs) -> Dict:
    """Call the FastAPI endpoint to get an answer"""
    # Convert choices from string to dict if needed
    if isinstance(choices, str):
        choices_dict = {}
        for line in choices.strip().split("\n"):
            if line.strip():
                key, value = line.split(". ", 1)
                choices_dict[key] = value
    else:
        choices_dict = choices
    
    # Prepare request payload
    payload = {
        "question": question,
        "choices": choices_dict,
        "streaming": streaming,
        **kwargs
    }
    
    # Call API
    if streaming:
        response = requests.post(
            f"{API_URL}/api/v1/mcq/explain",
            json=payload,
            stream=True
        )
        return response
    else:
        response = requests.post(
            f"{API_URL}/api/v1/mcq/answer",
            json=payload
        )
        return response.json()

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Coding Multiple Choice Q&A with YAML Reasoning",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("Coding Multiple Choice Q&A with YAML Reasoning")
    st.markdown(
        """
        This app uses a fine-tuned Qwen2.5-Coder-1.5B model to answer multiple-choice coding questions with structured YAML reasoning.
        
        The model breaks down its thought process in a structured way, providing:
        - Understanding of the question
        - Analysis of all options
        - Detailed reasoning process
        - Clear conclusion
        """
    )
    
    # Check API health
    api_healthy, health_info = check_api_health()
    if not api_healthy:
        st.error(f"API Error: {health_info}")
        st.info("Please make sure the FastAPI server is running on http://localhost:8000")
        return
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Question Input")
        
        # Category selector
        category = st.selectbox(
            "Select a category",
            ["All Categories"] + list(CODING_EXAMPLES_BY_CATEGORY.keys())
        )
        
        # Example selector
        examples = get_category_examples(category)
        example_options = [f"Example {i+1}: {ex['question']}" for i, ex in enumerate(examples)]
        example_idx = st.selectbox("Select an example question", range(len(examples)), format_func=lambda x: example_options[x])
        
        # Question input
        question = st.text_area("Question", value=examples[example_idx]["question"] if examples else "", height=100)
        
        # Choices input
        choices_text = format_choices_for_display(examples[example_idx]["choices"]) if examples else ""
        choices = st.text_area("Choices (one per line)", value=choices_text, height=150)
        
        # Generation parameters
        st.subheader("Generation Parameters")
        
        col_params1, col_params2 = st.columns(2)
        
        with col_params1:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05, 
                                   help="Higher = more creative, lower = more deterministic")
            max_new_tokens = st.slider("Max New Tokens", 128, 2048, 512, 128, 
                                      help="Maximum length of generated response")
            top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.05, 
                             help="Nucleus sampling probability")
        
        with col_params2:
            top_k = st.slider("Top-k", 1, 100, 50, 1, 
                             help="Number of highest probability tokens to consider")
            repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.1, 0.1, 
                                          help="Higher = less repetition")
            do_sample = st.checkbox("Enable Sampling", value=True, 
                                   help="Unchecked for greedy generation")
        
        # Streaming option
        streaming = st.checkbox("Enable Streaming", value=False, 
                              help="Stream the response in real-time")
        
        # Submit button
        submit = st.button("Submit", type="primary")
    
    with col2:
        st.subheader("Model Response")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Formatted Response", "Raw YAML", "JSON View"])
        
        if submit:
            # Show spinner while processing
            with st.spinner("Generating response..."):
                try:
                    # Call API
                    if streaming:
                        response = call_api(
                            question=question,
                            choices=choices,
                            streaming=True,
                            temperature=temperature,
                            max_new_tokens=max_new_tokens,
                            top_p=top_p,
                            top_k=top_k,
                            repetition_penalty=repetition_penalty,
                            do_sample=do_sample
                        )
                        
                        # Process streaming response
                        with tab1:
                            st.markdown("### Streaming Response")
                            response_container = st.empty()
                            
                            with tab2:
                                st.markdown("### Raw YAML")
                                yaml_container = st.empty()
                                
                            with tab3:
                                st.markdown("### JSON View")
                                json_container = st.empty()
                            
                            # Process streaming response
                            full_response = ""
                            for line in response.iter_lines():
                                if line:
                                    line = line.decode('utf-8')
                                    if line.startswith('data: '):
                                        data = line[6:]
                                        if data == '[DONE]':
                                            break
                                        
                                        try:
                                            json_data = json.loads(data)
                                            full_response = json.dumps(json_data, indent=2)
                                            
                                            # Update containers
                                            with response_container:
                                                st.markdown(f"""
                                                **Understanding:** {json_data.get('understanding', '')}
                                                
                                                **Analysis:** {json_data.get('analysis', '')}
                                                
                                                **Reasoning:** {json_data.get('reasoning', '')}
                                                
                                                **Conclusion:** {json_data.get('conclusion', '')}
                                                
                                                **Answer:** {json_data.get('answer', '')}
                                                """)
                                            
                                            with yaml_container:
                                                st.code(yaml.dump(json_data, default_flow_style=False), language="yaml")
                                            
                                            with json_container:
                                                st.json(json_data)
                                        except json.JSONDecodeError:
                                            pass
                    else:
                        # Non-streaming response
                        response_data = call_api(
                            question=question,
                            choices=choices,
                            streaming=False,
                            temperature=temperature,
                            max_new_tokens=max_new_tokens,
                            top_p=top_p,
                            top_k=top_k,
                            repetition_penalty=repetition_penalty,
                            do_sample=do_sample
                        )
                        
                        # Display response in tabs
                        with tab1:
                            st.markdown(f"""
                            **Understanding:** {response_data.get('understanding', '')}
                            
                            **Analysis:** {response_data.get('analysis', '')}
                            
                            **Reasoning:** {response_data.get('reasoning', '')}
                            
                            **Conclusion:** {response_data.get('conclusion', '')}
                            
                            **Answer:** {response_data.get('answer', '')}
                            """)
                        
                        with tab2:
                            st.code(yaml.dump(response_data, default_flow_style=False), language="yaml")
                        
                        with tab3:
                            st.json(response_data)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Please check the API connection and try again.")

if __name__ == "__main__":
    main() 