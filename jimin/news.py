import datasets
from datasets import load_dataset, DatasetDict
from transformers import pipeline
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# GPU 사용 가능 여부 확인 및 설정
device = 0 if torch.cuda.is_available() else -1
print(f"사용 가능한 디바이스: {'GPU' if device == 0 else 'CPU'}")
# print(f"torch 버전: {torch.__version__}, datasets 버전: {datasets.__version__}")

# 데이터셋: KLUE MRC (기계 독해 데이터셋이지만, 기사 원문(context)을 활용)
DATASET_NAME = "klue"
SUBSET_NAME = "mrc"
NUM_SAMPLES = 100

# 1. KLUE MRC 데이터셋의 'train' split 로드
full_dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split="train")

# 2. 실습을 위해 앞 100개 샘플만 선택하여 서브셋 생성
processing_subset = full_dataset.select(range(NUM_SAMPLES))

print(f"로드된 데이터셋 정보 (총 {NUM_SAMPLES}개 샘플):")
print(processing_subset)
print("\n첫 번째 데이터 예시 (context 확인):")
print(processing_subset[0]['context'][:150] + "...") # 원문 일부 출력

# 데이터셋 확인을 위해 Pandas DataFrame으로 변환
df_initial_check = pd.DataFrame(processing_subset)
# display(df_initial_check.head(3)) # Colab에서 확인용


# 모델: gogamza/kobart-summarization (BART 기반 한국어 요약 모델)
summarizer = pipeline(
    task="summarization",
    model="gogamza/kobart-summarization",
    device=device
)
print("요약 파이프라인 로드 완료.")

def summarize_context(example):
    """'context'를 요약하고 'summary' 컬럼에 추가"""
    summary_result = summarizer(
        example['context'],
        max_length=120, # 최대 길이 제한
        min_length=20,
        do_sample=False # 확정적(deterministic) 결과
    )
    example['summary'] = summary_result[0]['summary_text']
    return example

# 데이터셋에 적용
processing_subset = processing_subset.map(summarize_context)
print("\n요약 작업 완료.")

# @title 3.2. 번역 파이프라인 로드 및 적용
# 모델: facebook/nllb-200-distilled-600M (다국어 번역 모델)
translator = pipeline(
    task="translation",
    model="facebook/nllb-200-distilled-600M",
    device=device
)
print("번역 파이프라인 로드 완료.")

def translate_summary_to_english(example):
    """'summary'를 영어로 번역하고 'english_summary' 컬럼에 추가"""
    translation_result = translator(
        example['summary'],
        src_lang="kor_Hang", # 한국어
        tgt_lang="eng_Latn", # 영어
        max_length=400
    )
    example['english_summary'] = translation_result[0]['translation_text']
    return example

# 데이터셋에 적용
processing_subset = processing_subset.map(translate_summary_to_english)
print("\n번역 작업 완료.")

# @title 3.3. 감정 분석 파이프라인 로드 및 적용
# 모델: SamLowe/roberta-base-go_emotions (영어 감정 분류 모델)
emotion_classifier = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=1, # 가장 확률 높은 감정 1개만 반환
    device=device
)
print("감정 분석 파이프라인 로드 완료.")

def analyze_emotion(example):
    """'english_summary'의 감정을 분석하고 'emotion' 컬럼에 추가"""
    # 감정 분석 파이프라인은 텍스트를 리스트 형태로 받으므로, 입력 텍스트를 리스트로 감싸야 합니다.
    emotion_result = emotion_classifier(example['english_summary'])
    # top_k=1 옵션으로 인해 결과는 [[{'label': '...', 'score': ...}]] 형태
    example['emotion'] = emotion_result[0][0]['label']
    example['emotion_score'] = emotion_result[0][0]['score'] # 점수도 함께 저장
    return example

# 데이터셋에 적용
final_dataset = processing_subset.map(analyze_emotion)
print("\n감정 분석 작업 완료.")

# @title 5. 결과 분석 및 확인
print("--- 최종 결과 데이터셋 (3개 샘플) ---")
df_final = pd.DataFrame(final_dataset)

# 필요한 컬럼만 선택하여 출력
columns_to_show = ['context', 'summary', 'english_summary', 'emotion', 'emotion_score']
df_final_preview = df_final[columns_to_show]

# 원문은 너무 길 수 있으므로 줄여서 출력
df_final_preview['context'] = df_final_preview['context'].apply(lambda x: x[:70] + '...')

display(df_final_preview.head(3)) # Colab에서 확인용

# 최종 결과 확인 (첫 3개 샘플)
for i in range(min(3, len(final_dataset))):
    print(f"\n--- 샘플 {i+1} ---")
    print(f"원문 일부: {final_dataset[i]['context'][:70]}...")
    print(f"요약: {final_dataset[i]['summary']}")
    print(f"영어 번역: {final_dataset[i]['english_summary']}")
    print(f"분석 감정: {final_dataset[i]['emotion']} (Score: {final_dataset[i]['emotion_score']:.4f})")

# 6. 시각화 (감정 비율 그래프)
plt.figure(figsize=(10, 6))
sns.countplot(y='emotion', data=df_final, order=df_final['emotion'].value_counts().index, palette='viridis')
plt.title(f'분석된 감정 분포 (총 {NUM_SAMPLES}개 샘플)')
plt.xlabel('빈도수')
plt.ylabel('감정 레이블')
plt.show() # Colab에서 확인용
print("\n감정 분석 결과 시각화 준비 완료. ")