# ==== 0) (필요시) 설치 ====
# pip install -U transformers datasets torch pandas sentencepiece tqdm

import os
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
import torch

# ==== 1) 입출력 경로 ====
INPUT_CSV  = "naver_finance_top10.csv"   # 업로드한 파일 이름
OUTPUT_CSV = "naver_finance_top10_summarized.csv"

# ==== 2) 어떤 모델을 쓸지 선택 (언어별 프리셋) ====
# 한국어 요약 (권장)
MODEL_ID = "gogamza/kobart-summarization"
# 영어 요약(CNN/DailyMail 기반)
# MODEL_ID = "facebook/bart-large-cnn"

# ==== 3) 요약 대상 컬럼 지정 ====
# 보통 본문은 'body' 또는 'content', 제목은 'title'류일 가능성이 큼
TEXT_COL_CANDIDATES = ["body", "content", "text"]
TITLE_COL_CANDIDATES = ["title_from_list", "title", "headline"]

# ==== 4) 디바이스 설정 ====
device = 0 if torch.cuda.is_available() else -1

# ==== 5) 데이터 로드 & 컬럼 결정 ====
df = pd.read_csv(INPUT_CSV)
cols_lower = {c.lower(): c for c in df.columns}

def pick_col(cands, default=None):
    for c in cands:
        if c in cols_lower:
            return cols_lower[c]
    return default

TEXT_COL  = pick_col(TEXT_COL_CANDIDATES, default=df.columns[-1])  # 그래도 없으면 마지막 컬럼 사용
TITLE_COL = pick_col(TITLE_COL_CANDIDATES, default=None)

# 빈값 처리
df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("")
if TITLE_COL is not None:
    df[TITLE_COL] = df[TITLE_COL].astype(str).fillna("")

# ==== 6) 요약 파이프라인 준비 ====
summarizer = pipeline(
    task="summarization",
    model=MODEL_ID,
    device=device
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 모델 토큰 최대 길이(대부분 1024 또는 1024±)
model_max = getattr(tokenizer, "model_max_length", 1024)
# 안전 마진 (헤더 토큰 등)
max_input_tokens = min(model_max, 1024)

# ==== 7) 긴 텍스트를 토큰 길이에 맞춰 '청크'로 자르고 각각 요약 → 합치기 ====
def chunk_by_tokens(text, max_tokens=1024, overlap_tokens=50):
    """토크나이저 기준으로 max_tokens 단위로 텍스트를 잘라 리스트로 반환"""
    if not text.strip():
        return []

    enc = tokenizer(text, return_tensors=None, add_special_tokens=False)
    ids = enc["input_ids"]
    chunks = []
    start = 0
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        piece_ids = ids[start:end]
        chunk_text = tokenizer.decode(piece_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == len(ids): break
        start = end - overlap_tokens  # 약간 겹치게 이어 붙이기
        if start < 0: start = 0
    return chunks

def summarize_text(text, max_summary_len=150, min_summary_len=30):
    """긴 텍스트도 알아서 청크 요약 후 문단으로 합침"""
    text = text.strip()
    if not text:
        return ""

    chunks = chunk_by_tokens(text, max_tokens=max_input_tokens-32, overlap_tokens=40)
    # 짧으면 그냥 한 번에
    if len(chunks) <= 1:
        try:
            out = summarizer(
                text,
                max_length=max_summary_len,
                min_length=min_summary_len,
                do_sample=False
            )[0]["summary_text"]
            return out.strip()
        except Exception:
            return ""
    # 길면 청크별 요약 → 합치기 → 다시 짧게 요약(선택)
    partial_summaries = []
    for ck in chunks:
        try:
            s = summarizer(
                ck,
                max_length=max_summary_len,
                min_length=min_summary_len,
                do_sample=False
            )[0]["summary_text"]
            partial_summaries.append(s.strip())
        except Exception:
            continue
    if not partial_summaries:
        return ""

    joined = "\n".join(partial_summaries)
    # 최종 압축 요약 (선택)
    try:
        final = summarizer(
            joined,
            max_length=max_summary_len,
            min_length=min_summary_len,
            do_sample=False
        )[0]["summary_text"]
        return final.strip()
    except Exception:
        return joined.strip()

# ==== 8) 전체 데이터 요약 실행 ====
summaries = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Summarizing"):
    base_text = row[TEXT_COL]
    # 제목이 있으면 앞에 붙여 힌트 제공 (요약 품질 상승)
    if TITLE_COL is not None and str(row[TITLE_COL]).strip():
        text = f"{row[TITLE_COL]}\n{base_text}"
    else:
        text = base_text

    try:
        s = summarize_text(text)
    except Exception as e:
        s = ""
    summaries.append(s)

df["summary"] = summaries

# ==== 9) 저장 ====
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"Saved: {OUTPUT_CSV}  (rows={len(df)})")
print(f"Columns: {list(df.columns)}")
