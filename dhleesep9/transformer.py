import datasets
from datasets import load_dataset, DatasetDict
from transformers import pipeline
import torch
import pandas as pd # 데이터 확인용

# GPU 사용 가능 여부 확인 및 설정 (Colab에서는 보통 GPU 사용 가능)
device = 0 if torch.cuda.is_available() else -1
print(f"사용 가능한 디바이스: {'GPU' if device == 0 else 'CPU'}")


full_dataset = load_dataset("yelp_polarity", split="train")
full_dataset = full_dataset.remove_columns(["label"])

# 실습을 위해 데이터 일부만 선택 (예: 앞 10개)
num_samples_to_use = 20
full_dataset_subset = full_dataset.select(range(num_samples_to_use))

print("로드된 데이터셋 정보:")
print(full_dataset_subset)

print("\n첫 번째 데이터 예시 (text 확인):")
print(full_dataset_subset[0]['text'])

# 데이터셋 확인을 위해 Pandas DataFrame으로 변환 (선택 사항)
df_check = pd.DataFrame(full_dataset_subset)
print("\n데이터셋 일부 미리보기 (DataFrame):")
display(df_check.head(3)) # Colab 환경에서는 display()가 표 형태로 보여줍니다


translator = pipeline(
    task="translation",
    model="facebook/nllb-200-distilled-600M",
    device=device
)

print("번역 파이프라인 로드 완료.")


# 번역을 수행하는 함수 정의
def translate_english_to_korean(example):
  """데이터셋의 'summary'를 받아 영어 번역 결과를 반환하는 함수"""
  translation_result = translator(
        example['text'],
        tgt_lang="kor_Hang",
        src_lang="eng_Latn",
        max_length=400,
        min_length=30,
        do_sample=False  # deterministic 출력을 원하면 False, 다양성을 원하면 True
    )
    # 결과에서 번역 텍스트 추출
  example['korean_translate'] = translation_result[0]['translation_text']
  return example

print("번역 작업을 시작합니다... (모델 크기와 데이터 양에 따라 시간이 많이 소요될 수 있습니다)")
translated_dataset = full_dataset_subset.map(translate_english_to_korean)
print("번역 작업 완료.")

print("\n번역이 추가된 데이터셋 정보:")
print(translated_dataset)
print(translated_dataset[0]['text'])
print(translated_dataset[0]['korean_translate'])

# 데이터셋 확인 (Pandas)
df_check_translation = pd.DataFrame(translated_dataset)
print("\n번역 추가 후 데이터셋 미리보기 (DataFrame):")
display(df_check_translation.head(3))


emotion_classifier = pipeline(
    task="text-classification",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    top_k=1,
    device=device
)
print("감정 분석 파이프라인 로드 완료.")

# 감정 분석 테스트 (선택 사항)
test_emotion = emotion_classifier("좋았다")
print(f"감정 분석 테스트: {test_emotion[0][0]}")


# 감정 분석을 수행하는 함수 정의
def analyze_emotion(example):
  emotion_result = emotion_classifier(example['korean_translate'])
  example['emotion'] = emotion_result[0][0]['label']
  return example

print("감정 분석 작업을 시작합니다...")
final_dataset = translated_dataset.map(analyze_emotion)
print("감정 분석 작업 완료.")

print("\n최종 데이터셋 정보:")
print(final_dataset)

print(final_dataset[0]['korean_translate'])
print("\n--- 분석된 감정 (Emotion) ---")
print(final_dataset[0]['emotion'])

# 최종 데이터셋 확인 (Pandas)
df_final = pd.DataFrame(final_dataset)
print("\n최종 데이터셋 미리보기 (DataFrame):")
display(df_final) # 전체 선택된 샘플 표시


# 최종 결과 확인 (첫 5개 샘플)
for i in range(min(5, len(final_dataset))):
  print(f"\n--- 샘플 {i+1} ---")
  print(f"원문 일부: {final_dataset[i]['text'][:100]}...")
  print(f"한국어 번역: {final_dataset[i]['korean_translate']}")
  print(f"예측 평점: {final_dataset[i]['emotion']}")


