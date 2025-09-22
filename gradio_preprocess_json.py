import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import gradio as gr

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()["embeddings"]
    return embedding

def preprocess_jsons():
    jsons = os.listdir("jsons")
    my_dicts = []
    chunk_id = 0
    for json_file in jsons:
        with open(f"jsons/{json_file}") as f:
            content = json.load(f)
        print(f"Creating Embeddings for {json_file}")
        embeddings = create_embedding([c['text'] for c in content['chunks']])
        for i, chunk in enumerate(content['chunks']):
            chunk['chunk_id'] = chunk_id
            chunk['embedding'] = embeddings[i]
            chunk_id += 1
            my_dicts.append(chunk)
    df = pd.DataFrame.from_records(my_dicts)
    joblib.dump(df, 'embeddings.joblib')
    return "Preprocessing complete! Embeddings saved."

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    response = r.json()
    return response["response"]

def rag_query(user_query):
    try:
        df = joblib.load('embeddings.joblib')
    except Exception:
        return "Embeddings not found. Please run preprocessing first."
    question_embedding = create_embedding([user_query])[0]
    similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
    top_results = 5
    max_indx = similarities.argsort()[::-1][:top_results]
    new_df = df.iloc[max_indx]
    prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{user_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''
    return inference(prompt)

with gr.Blocks() as demo:
    gr.Markdown("# RAG-based Web Dev Q&A")
    preprocess_btn = gr.Button("Preprocess JSONs")
    preprocess_output = gr.Textbox(label="Preprocessing Status")
    query_input = gr.Textbox(label="Ask a Question")
    query_output = gr.Textbox(label="Answer")

    preprocess_btn.click(preprocess_jsons, outputs=preprocess_output)
    query_input.submit(rag_query, inputs=query_input, outputs=query_output)

if __name__ == "__main__":
    demo.launch()