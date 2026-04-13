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
- annotation: `BoKelvin/SLAKE` (Hugging Face)
- image files: original SLAKE image archive
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
- 평가 split: `train` / `validation` / `test` 중 선택 가능
- 일반적으로 `test` split 사용 권장
- Hugging Face의 SLAKE `test` split은 총 2094 samples이지만, 본 코드는 `q_lang == "en"`만 사용하므로 실제 평가 수는 더 적을 수 있음

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
```

---

## 7. 데이터 준비

이 레포는 annotation과 image를 분리해서 사용합니다.

- annotation: Hugging Face `BoKelvin/SLAKE`
- image files: original SLAKE image archive

### Hugging Face annotation 확인

공개 데이터셋이므로 일반적으로 HF token 없이 사용 가능합니다.

```bash
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("BoKelvin/SLAKE", split="test")
print(ds)
print(ds[0])
PY
```

### 원본 SLAKE 이미지 다운로드

원격 서버에서 직접 받는 것을 권장합니다.

```bash
pip install -U gdown
mkdir -p /workspace/datasets
cd /workspace/datasets
gdown 1EZ0WpO5Z6BJUqC3iPBQJJS1INWSMsh7U -O SLAKE_raw.zip
unzip SLAKE_raw.zip -d SLAKE_raw
```

### 이미지 경로

실험에서는 아래 경로를 사용합니다.

```text
/workspace/datasets/SLAKE_raw/Slake1.0
```

예시 이미지:

```text
/workspace/datasets/SLAKE_raw/Slake1.0/imgs/xmlab102/source.jpg
```

---

## 8. 실행 방법

### 소규모 디버그 실행

먼저 3개 정도만 테스트하는 것을 권장합니다.

```bash
python -m src.run_eval \
  --use_hf \
  --slake_root /workspace/datasets/SLAKE_raw/Slake1.0 \
  --split test \
  --condition original \
  --output_dir outputs/original_debug \
  --max_samples 3
```

### 단일 condition 실행

예: `black` condition

```bash
python -m src.run_eval \
  --use_hf \
  --slake_root /workspace/datasets/SLAKE_raw/Slake1.0 \
  --split test \
  --condition black \
  --output_dir outputs/black_debug \
  --max_samples 20
```

예: `hpf` condition

```bash
python -m src.run_eval \
  --use_hf \
  --slake_root /workspace/datasets/SLAKE_raw/Slake1.0 \
  --split test \
  --condition hpf \
  --output_dir outputs/hpf_debug \
  --max_samples 20
```

### 전체 condition 실행

```bash
bash scripts/run_all_conditions.sh /workspace/datasets/SLAKE_raw/Slake1.0 test 2094 outputs_full
```

주의:
- `2094`는 test split 전체 요청 개수입니다.
- 실제 평가 수는 English subset 필터링 후 줄어들 수 있습니다.

### SSH 세션 종료 후에도 계속 실행

```bash
nohup bash scripts/run_all_conditions.sh /workspace/datasets/SLAKE_raw/Slake1.0 test 2094 outputs_full > outputs_full_run.log 2>&1 &
```

로그 확인:

```bash
tail -f outputs_full_run.log
```

---

## 9. 결과 비교

모든 condition 실행 후 결과를 한 표로 정리할 수 있습니다.

```bash
python -m src.compare_results --base_out outputs_full --save_dir outputs_full/compare
```

주요 결과 파일:
- `outputs_full/compare/overall_results.csv`
- `outputs_full/compare/overall_results_pivot.csv`
- `outputs_full/compare/answer_type_results_pivot.csv`
- `outputs_full/compare/all_results_long.csv`

확인 예시:

```bash
cat outputs_full/compare/overall_results_pivot.csv
cat outputs_full/compare/answer_type_results_pivot.csv
```

---

## 10. 출력 파일 설명

각 condition 폴더에는 아래 파일이 저장됩니다.

- `predictions.jsonl`
  - sample-level prediction
- `predictions.csv`
  - sample-level prediction in csv format
- `summary.json`
  - overall / answer_type / q_type summary
- `summary.csv`
  - flattened summary table

예:

```text
outputs_full/original/
outputs_full/black/
outputs_full/lpf/
outputs_full/hpf/
outputs_full/patch_shuffle/
```

---

## 11. 결과 해석 예시

- `original` 대비 `black` 성능 하락이 작으면  
  → 모델이 이미지 없이도 텍스트 prior로 답할 가능성

- `patch_shuffle`에서 성능이 크게 하락하면  
  → 모델이 전역 spatial structure를 활용할 가능성

- `lpf`와 `hpf` 차이가 크면  
  → 모델이 저주파/고주파 정보에 서로 다르게 의존할 가능성

- `closed`는 유지되는데 `open`이 크게 하락하면  
  → yes/no 수준의 coarse 판단은 가능하지만 정밀 시각 추론은 약할 가능성

---

## 12. 한계

- open-ended exact match는 매우 보수적인 평가 방식임
- semantic equivalence를 충분히 반영하지 못할 수 있음
- prompt wording에 민감할 수 있음
- 현재 HPF 구현은 정보 제거보다는 edge enhancement에 가까워 보일 수 있음
- 결과 해석 시 질문 유형 분포(`closed/open`, `q_type`)를 함께 확인해야 함
