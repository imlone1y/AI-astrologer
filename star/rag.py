import os, glob, json, pathlib
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import ollama
import pdfminer.high_level
from docx import Document

# --------------------- 1. 嵌入與索引 --------------------- #
EMBED_MODEL = 'bge-m3'        # Ollama 同名 embedding 模型
INDEX_FILE   = 'faiss.index'
META_FILE    = 'meta.json'
FILES_DIR    = './files'
TOP_K        = 4              # 每次抓幾筆 context 當知識

def _read_text(path: str) -> str:
    ext = pathlib.Path(path).suffix.lower()
    if ext == '.pdf':
        return pdfminer.high_level.extract_text(path)
    elif ext in ('.docx', '.doc'):
        return "\n".join(p.text for p in Document(path).paragraphs)
    else:  # .txt .md …
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

def embed_files():
    model = SentenceTransformer(f'BAAI/{EMBED_MODEL}')
    index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())
    metadata = []          # list[dict]: 儲存檔名與原文

    for file in glob.glob(os.path.join(FILES_DIR, '**'), recursive=True):
        if os.path.isdir(file):
            continue
        text = _read_text(file)
        if not text.strip():
            continue
        chunks = [text[i:i+2048]             # 每 2 k 字切一段，避免超長
                  for i in range(0, len(text), 2048)]
        embs = model.encode(chunks, normalize_embeddings=True)
        index.add(np.array(embs, dtype='float32'))
        metadata.extend([{'file': file, 'chunk': c} for c in chunks])

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f'完成嵌入 {len(metadata)} 段文字，索引存於 {INDEX_FILE}')

# --------------------- 2. 檢索 --------------------- #
def _load_index() -> Tuple[faiss.IndexFlatL2, List[dict], SentenceTransformer]:
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    embedder = SentenceTransformer(f'BAAI/{EMBED_MODEL}')
    return index, meta, embedder

def retrieve(query: str, top_k: int = TOP_K) -> List[str]:
    index, meta, embedder = _load_index()
    q_emb = embedder.encode([query], normalize_embeddings=True).astype('float32')
    D, I = index.search(q_emb, top_k)
    return [meta[i]['chunk'] for i in I[0]]

# --------------------- 3. RAG 生成 --------------------- #
OLLAMA_URL = 'http://localhost:11434'   # 預設服務位址
LLM_MODEL  = 'gemma3:12b'               # 回答用模型

def load_user_chart(username: str) -> str:
    """讀取用戶的星盤資料（例如 Justin.txt）"""
    user_file = f"./profiles/{username}.txt"
    if not os.path.exists(user_file):
        return "User chart data not found."
    with open(user_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

def rag_ollama(prompt: str, username: str) -> str:
    """
    基於使用者星盤與問題，在占星書中檢索相關內容並由 LLM 回答。
    """
    # 讀取使用者星盤資料
    user_chart = load_user_chart(username)

    # 使用問題做向量搜尋
    contexts = retrieve(prompt, TOP_K)

    # 組合英文提示詞（Prompt）
    full_prompt = (
        f"You are a warm and insightful astrologer. "
        f"Based on {username}'s natal chart and relevant astrology references, "
        f"respond like you're having a personal one-on-one consultation.\n\n"

        f"Greet {username} by name at the beginning of your answer, and explain your insights in a natural, conversational tone—"
        f"not like an assistant or a robot.\n\n"

        f"Keep your reply under 50 words. Just answer the question directly, based on the chart and astrology knowledge.\n"
        f"At the end, ask if they'd like to ask something else.\n\n"

        f"### User Name:\n{username}\n\n"
        f"### User Natal Chart:\n{user_chart}\n\n"
        f"### Relevant Astrology Book Excerpts:\n{chr(10).join(contexts)}\n\n"
        f"### User's Question:\n{prompt}\n\n"
        f"### Your Answer:"
    )
    # 呼叫 Ollama 模型
    response = ollama.generate(
        model=LLM_MODEL,
        prompt=full_prompt,
        stream=False
    )
    return response['response']


# --------------------- 用法示範 --------------------- #
if __name__ == "__main__":
    if not os.path.exists(INDEX_FILE):
        embed_files()                       # 第一次執行先建立索引
    while True:
        q = input("\n輸入問題(Enter 結束)：")
        if not q.strip():
            break
        print("\n>>> ChatGPT：", rag_ollama(q))
