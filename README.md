# Falcon-7B-Instruct Chatbot with Gradio

This repository contains code for setting up a conversational AI chatbot using the Falcon-7B-Instruct model and Gradio for the user interface.

## Installation

Before running the code, make sure you have the required packages installed. You can install them using pip:

```bash
pip install -q transformers einops accelerate langchain bitsandbytes
pip install fastapi==0.104.1 typing_extensions==4.8.0 gradio==3.41.0
```

## Usage

1. Import necessary libraries and initialize the Falcon-7B-Instruct model and tokenizer:

```python
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch
import gradio as gr

model = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
```

2. Create a text generation pipeline using the Falcon-7B-Instruct model:

```python
pipeline = pipeline(
    "text-generation",  # task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)
```

3. Define the function to interact with the chatbot:

```python
def function(message, history):
    sequences = pipeline(
        message,
        max_length=200,
        do_sample=False,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    for seq in sequences:
        return f"{seq['generated_text']}"
```

4. Set up the Gradio interface for the chatbot:

```python
gr.ChatInterface(
    function,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me anything", container=False, scale=7),
    title="Your Chatbot",
    description="A simple conversational AI chatbot powered by Falcon-7B-Instruct.",
    theme="soft",
    examples=["Tell me a joke", "What is AI?", "Are tomatoes vegetables?"],
    cache_examples=True,
    retry_btn='Retry',
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch(share=True)
```

5. Launch the Gradio interface and start interacting with the chatbot.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---