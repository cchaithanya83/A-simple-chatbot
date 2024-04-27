#  pip install -q transformers einops accelerate langchain bitsandbytes
#  pip install fastapi==0.104.1 typing_extensions==4.8.0 gradio==3.41.0


from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch
import gradio as gr

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation", #task
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



def function(message , history):
  sequences = pipeline(
    message,
      max_length=200,
      do_sample=False,
      top_k=10,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
  )
  print(sequences)
  for seq in sequences:
      return(f"{seq['generated_text']}")




gr.ChatInterface(
    function,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me a anything", container=False, scale=7),
    title="Your chatbot",
    description="A chatbot",
    theme="soft",
    examples=["Tell me a joke", "What is ai?", "Are tomatoes vegetables?"],
    cache_examples=True,
    retry_btn='Retry',
    undo_btn="Delete Previous",
    clear_btn="Clear",

).launch(share=True)

