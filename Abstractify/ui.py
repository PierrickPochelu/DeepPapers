import gradio as gr
import fitz  # PyMuPDF
import time
import pickle
import requests # for prompting OpenAI models
from typing import Tuple, Optional, Callable

##########################
# GLOBAL DATA STRUCTURES #
##########################

# OpenAI prompt
import os
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] # <--- put your key

# scikit-learn model
with open(f"/tmp/model.pkl", 'rb') as f:
    loaded_model = pickle.load(f)
with open(f"/tmp/class_names.pkl", 'rb') as f:
    class_names = pickle.load(f)

# TinyBERT model
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("/tmp/saved_model")
tokenizer = AutoTokenizer.from_pretrained("/tmp/saved_model")
with open("/tmp/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Summarizer models
from transformers import pipeline
summarizer_t5 = pipeline("summarization", model="t5-small", device=-1)
summarizer_bart = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)


####################################
# DATA EXTRACTION STRATEGY         #
####################################

def _extract_text_from_pdf(pdf_path:str, max_chars:int=10000):
    pdf_document = fitz.open(pdf_path)

    num_chars=0
    pages_content = []

    num_pages=pdf_document.page_count
    for page_num in range(num_pages):
        page = pdf_document.load_page(page_num)
        page_text = page.get_text()

        print("Page uploaded:", page_num, " #chars:", len(page_text), f" text: '{page_text[:100]}' ...")

        if num_chars+len(page_text)>=max_chars:
            truncated_page_text= page_text[:(max_chars - len(page_text))]
            pages_content.append(truncated_page_text)
            break
        else:
            num_chars += len(page_text)
            pages_content.append(page_text)


    pdf_text = " ".join(pages_content)
    return pdf_text

def _summary_and_clean(text:str, summarizer:Callable)->str:
    max_length=100 # max #words
    min_length=50
    chucnk_size=512
    if len(text.split()) > chucnk_size:  # If it's too long, split into chunks
        chunks = [text[i:i + chucnk_size] for i in range(0, len(text), chucnk_size)]
        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        return ' '.join(summaries)  # Join all chunk summaries
    else:
        # Otherwise summarize normally
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    return out

def strategy_raw(text:str)->str:
    return text

def strategy_t5(text:str)->str:
    summarized_text=_summary_and_clean(text, summarizer_t5)
    return summarized_text

def strategy_bart(text:str)->str:
    summarized_text=_summary_and_clean(text, summarizer_bart)
    return summarized_text

##############################
# INFERENCE STRATEGY         #
##############################

def sklearn_inference(text:str, TOP_PRED = 5)->str:

    predicted_probs = loaded_model.predict_proba([text])
    top_5_indices = predicted_probs.argsort()[0][-TOP_PRED:][::-1]  # Get the indices of top 5 predictions

    top_5_classes = class_names[top_5_indices]  # Get the corresponding class names
    top_5_probs = predicted_probs[0][top_5_indices]  # Get the corresponding probabilities

    # Print the top 5 predicted classes and their probabilities
    strings_out = []
    for cls, prob in zip(top_5_classes, top_5_probs):
        strings_out.append(f"{cls}: {prob * 100:.2f}%")
    out = "\n".join(strings_out)
    return out

def openai_inference(sample_text):
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL_NAME = "mistralai/mistral-7b-instruct"

    prompt = f"""
    You are an expert in scientific paper classification. Given the abstract below, classify it into one of the official arXiv categories. 

    Below the paper:
    {sample_text}

    Return only a category code among: {class_names}.
    Your answer should be only 20 characters maximum.
    """

    # Prepare API Request
    http_headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # Lower temperature for more deterministic responses
        "max_tokens": 50
    }

    # Send Request
    response = requests.post(API_URL, headers=http_headers, json=payload)

    # Exctract the response
    if response.status_code == 200:
        result = response.json()
        llm_output = result["choices"][0]["message"]["content"].strip()
        class_name = "other"
        for class_name in class_names:
            if class_name in llm_output:
                break
        out = class_name
    else:
        out = response.text

    return out

def torch_inference(text):
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokens)
    logits = output.logits
    top5_probs, top5_indices = torch.topk(logits, k=5, dim=1)  # Get top 5 classes
    top5_labels = label_encoder.inverse_transform(top5_indices.cpu().numpy().flatten())
    top5_probs = torch.nn.functional.softmax(top5_probs, dim=1).cpu().numpy().flatten()
    res = [(label, float(prob)) for label, prob in zip(top5_labels, top5_probs)]
    res = sorted(res, reverse=True, key=lambda p:p[1])

    lines=[]
    for pair in res:
        label=f"{pair[0]}: {pair[1]*100:.2f}%"
        lines.append(label)
    out="\n".join(lines)
    return out


##########################
# PIPELINE STRATEGY      #
# a pipeline contains:   #
#  1) a data extraction  #
#  2) an inference model #
##########################

def strategy_constant(txt:str)->str:
    return "No category", "No description. Is it a valid PDF ?"

def strategy_ml(txt:str)->str:
    pred=sklearn_inference(txt)
    return pred

def strategy_bert(txt:str)->str:
    pred = torch_inference(txt)
    return pred

def strategy_openai(txt: str)->str:
    txt=txt[:2000] # hard coded limit
    pred=openai_inference(txt)
    return pred

# ADD A NEW STRATEGY AND ITS NAME  IN THE MAP WHEN YOU DEVELOP A NEW STRATEGY
MAP_PROCESSING_STRATEGY={
    "Raw": strategy_raw,
    "T5-small": strategy_t5,
    "Bart": strategy_bart
}

MAP_INFERENCE_STRATEGY={
     "TFIDF+MLP": strategy_ml,
     "TinyBERT": strategy_bert,
     "Prompt+Mistral7b":strategy_openai
     }


###################
# INTERFACE CODE  #
###################

def strategy_dispatcher(file_path:Optional[str], strategy_summary_selected:str, strategy_inference_selected:str):
    """
    Processes the uploaded PDF (content ignored in this proof-of-concept) and returns:
      - A classification string.
      - A paper summary.
    """

    enlapsed_time_start=time.time()

    if file_path:
        strategy_summary = MAP_PROCESSING_STRATEGY[strategy_summary_selected]
        strategy_inference = MAP_INFERENCE_STRATEGY[strategy_inference_selected]
    else:
        return "No file", "Have you selected a file ?"

    # 1 read the file
    txt=_extract_text_from_pdf(file_path, 10000)

    # 2 call the summarizer (if needed)
    descr=strategy_summary(txt)

    # 3 call the inference
    classif=strategy_inference(descr)

    enlapsed_time=time.time()-enlapsed_time_start
    time_text = f"{round(enlapsed_time,3)} seconds"

    return classif, descr, time_text

# Define the Gradio interface with an added radio input for strategy selection.
iface = gr.Interface(
    fn=strategy_dispatcher,
    inputs=[
        gr.File(label="Upload PDF"),
        gr.Radio(choices=list(MAP_PROCESSING_STRATEGY.keys()), label="Select Summarization", value=list(MAP_PROCESSING_STRATEGY.keys())[0]),
        gr.Radio(choices=list(MAP_INFERENCE_STRATEGY.keys()), label="Select Inference Model", value=list(MAP_INFERENCE_STRATEGY.keys())[0])
    ],
    outputs=[
        gr.Textbox(label="Topic Classification"),
        gr.Textbox(label="Paper Summary"),
        gr.Textbox(label="Inference time")
    ],
    title="Paper Analyzer",
    description=(
        "How it works ?\n\n"
        " 1) Upload a PDF\n"
        " 2) Choose a Summarization strategy: 'Raw' (0 sec.), T5-small (takes ~80 sec.), Bart (takes ~210 sec.)\n"
        " 3) Choose a ML strategy: TFIDF+MLP (0 sec.), TinyBert (0 sec.), Prompt+Mistral7b (~2 sec.)\n"
        " 4) Click on submit\n"
    )
)

if __name__ == "__main__":
    iface.launch()
