# SLAKE 기반 의료 VQA 모델 이미지 의존성 평가

SLAKE 데이터셋에서 **HuatuoGPT-Vision-7B-Qwen2.5VL**의 이미지 의존성을 평가하기 위한 실험 레포입니다.  
의료 멀티모달 VQA 모델이 실제 시각 정보를 활용하는지, 아니면 질문 텍스트의 prior에 크게 의존하는지를 다양한 이미지 perturbation 조건에서 비교합니다.

---

## 1. 프로젝트 개요

의료 VQA 모델은 정답을 맞추더라도, 실제로는 이미지를 충분히 활용하지 않고 질문의 언어적 패턴만으로 답하는 경우가 있을 수 있습니다.  
이 프로젝트는 동일한 질문에 대해 여러 이미지 조건을 적용한 뒤 성능 변화를 측정하여, 모델의 **이미지 의존성(image dependence)** 을 정량적으로 평가하는 것을 목표로 합니다.

대상 모델:
- `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL`

대상 데이터셋:
- `SLAKE`
- English subset 사용

---

## 2. 연구 질문

이 실험은 다음 질문에 답하는 것을 목표로 합니다.

1. 모델은 실제 이미지를 보고 답하는가?
2. 이미지가 훼손되거나 제거되어도 성능이 유지되는가?
3. `closed` 질문과 `open` 질문에서 이미지 의존성 양상이 다르게 나타나는가?
4. 특정 perturbation에만 민감한가, 아니면 전반적으로 시각 정보 활용이 약한가?

---

## 3. 실험 조건

### 데이터셋
- SLAKE English subset
- 평가 split: `train` / `val` / `test` 중 선택 가능
- 일반적으로 `test` split 사용 권장

### 디코딩 설정
- greedy decoding
- `do_sample=False`
- `temperature=0.0`
- `max_new_tokens=16`

### 프롬프트
- 짧은 단답형 응답 유도
- yes/no 질문은 가능하면 `yes` 또는 `no`로 응답하도록 설계

### 재현성
- seed 고정
- deterministic patch shuffle 지원

---

## 4. 평가 설정

### Closed / Open 분리 평가
SLAKE의 질문 유형을 다음과 같이 분리하여 평가합니다.

- **closed**
  - 주로 yes/no 형태
  - robust yes/no parsing 후 accuracy 계산

- **open**
  - 일반 open-ended short answer
  - normalized exact match 기반 accuracy 계산
  - 필요 시 substring match 확장 가능

---

## 5. 이미지 조건

다음 5가지 이미지 조건에서 각각 성능을 측정합니다.

1. **original**
   - 원본 이미지 사용

2. **lpf**
   - Gaussian blur 적용
   - 저주파 성분 중심 정보만 남도록 유도

3. **hpf**
   - high-pass filtering 적용
   - 고주파 성분 중심 정보만 강조

4. **black**
   - 전체 이미지를 0으로 채운 검은 이미지 사용
   - 사실상 이미지 정보 제거

5. **patch_shuffle**
   - patch 단위 spatial shuffle
   - 지역 시각 정보는 일부 남지만 전역 구조 파괴

### Perturbation 기본값
- `lpf_sigma = 3.0`
- `hpf_sigma = 3.0`
- `patch_size = 16`

---

## 6. 설치 방법

### 환경
- Python 3.10+
- CUDA 사용 권장

### 설치
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt