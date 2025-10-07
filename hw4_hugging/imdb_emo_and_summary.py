#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install transformers datasets sentencepiece accelerate torch')

import datasets
from datasets import load_dataset, DatasetDict
from transformers import pipeline
import torch
import pandas as pd # 데이터 확인용

# GPU 사용 가능 여부 확인 및 설정 (Colab에서는 보통 GPU 사용 가능)
device = 0 if torch.cuda.is_available() else -1
print(f"사용 가능한 디바이스: {'GPU' if device == 0 else 'CPU'}")


# In[ ]:


# imdb 데이터셋 로드
dataset = load_dataset("imdb", split="test")

# 100개 샘플만 선택하여 서브셋 생성
subset = dataset.shuffle().select(range(100))

# 데이터셋 구조 확인
print("선택된 데이터셋 구조:")
print(subset)

# 첫 번째 샘플 확인
print("\n첫 번째 샘플:")
print(subset[0])


# In[ ]:


# 1. 요약 파이프라인 (영어 텍스트용)
# 'sshleifer/distilbart-cnn-12-6' 모델을 사용. 영어 텍스트 요약에 특화되어 있음.
summarizer = pipeline(
    task="summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=device
)

# 2. 감정 분석 파이프라인
# 'SamLowe/roberta-base-go_emotions' 모델 사용
emotion_classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=1,
    device=device
)

# 3. 번역 파이프라인
# 'facebook/nllb-200-distilled-600M' 모델 사용. 영어 -> 한국어 번역에 사용.
translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    src_lang="eng_Latn", # 원본 언어: 영어
    tgt_lang="kor_Hang", # 대상 언어: 한국어
    device=device
)

print("모든 파이프라인 로드 완료.")


# In[ ]:


# 'subset'은 데이터셋 객체, 'summarizer'는 이미 정의되었다고 가정
def summarize_text(sample):
    # pipeline 호출 시 'truncation=True' 파라미터 추가
    summary = summarizer(
        sample["text"],
        max_length=50,  # 요약 결과의 최대 길이
        min_length=10,  # 요약 결과의 최소 길이
        do_sample=False,
        truncation=True  # 입력 텍스트가 모델의 최대 길이를 초과하면 잘라냅니다.
    )
    sample["summary"] = summary[0]['summary_text']
    return sample

def analyze_emotion(sample):
    # 요약된 영어 텍스트의 감정분석
    result = emotion_classifier(sample["summary"])
    sample["emotion"] = result[0][0]['label']
    return sample

def translate_summary_to_ko(sample):
    # 요약된 영어 텍스트를 한국어로 번역
    result = translator(sample["summary"])
    sample["korean_summary"] = result[0]['translation_text']
    return sample

print("--")
subset = subset.map(summarize_text)
print("--")
subset = subset.map(analyze_emotion)
print("--")
subset = subset.map(translate_summary_to_ko)

print("Data processing complete.")
print(subset.select(range(5)))


# In[ ]:


# 최종 결과를 Pandas DataFrame으로 변환하여 출력
result_df = pd.DataFrame(subset)

# 필요한 컬럼만 선택하여 깔끔하게 출력
final_result = result_df[['text', 'emotion', 'summary', 'korean_summary']]
print(final_result.head())

# 감정 분석 결과의 분포 확인 (선택 사항)
print("\n감정 분석 결과 분포:")
print(final_result['emotion'].value_counts())


# In[ ]:


final_dataset = subset

# 최종 결과 확인 (첫 5개 샘플)
for i in range(min(5, len(final_dataset))):
  print(f"\n--- 샘플 {i+1} ---")
  print(f"원문 일부: {final_dataset[i]['text'][:100]}...")
  print(f"요약: {final_dataset[i]['summary']}")
  print(f"한글 번역: {final_dataset[i]['korean_summary']}")
  print(f"감정 분석: {final_dataset[i]['emotion']}")


# In[ ]:


# # CSV 저장
# result_df.to_csv("klue_mrc_processed.csv", index=False, encoding='utf-8-sig')
# print("\n결과를 CSV 파일로 저장했습니다. (필요시 주석 해제)")

