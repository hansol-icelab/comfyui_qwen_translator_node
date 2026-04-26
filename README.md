# ComfyUI Translator (Qwen KO→EN)

한국어 텍스트를 AI 이미지 생성에 적합한 영어 프롬프트로 번역하는 ComfyUI 커스텀 노드입니다.  
Qwen3 모델을 로컬에서 실행하며 인터넷 연결 없이 동작합니다.

---

## 모델 다운로드 (Qwen3-8B)

### 방법 1 — huggingface-cli (권장)

```bash
# huggingface_hub 설치 (없다면)
pip install huggingface_hub

# 모델 다운로드
hf download Qwen/Qwen3-8B \
    --local-dir models/text_encoders/Qwen3-8B \
```

### 방법 2 — Python 스크립트

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    local_dir="models/text_encoders/Qwen3-8B",
    local_dir_use_symlinks=False,
)
```

> **다운로드 위치:** ComfyUI 루트의 `models/text_encoders/Qwen3-8B/`  
> 다운로드 후 해당 폴더 안에 `config.json`이 있어야 노드에서 인식됩니다.

---

## ComfyUI 실행

```bash
python main.py
```

GPU VRAM이 부족하면 양자화 옵션을 사용하세요.

| VRAM | 권장 설정 |
|------|-----------|
| 24GB 이상 | dtype: `bfloat16`, quantization: `none` |
| 12GB 이상 | dtype: `bfloat16`, quantization: `8bit` |
| 8GB 이상  | dtype: `bfloat16`, quantization: `4bit` |

---

## 노드 구성

1. **Load Qwen Model** — 모델 디렉터리·dtype·양자화·디바이스를 선택해 모델을 로드합니다.
2. **Qwen Translator (KO→EN)** — 시스템 프롬프트와 한국어 텍스트를 입력받아 영어 프롬프트를 출력합니다.
3. **Show Text** — 번역 결과를 UI에 표시합니다.

---

## 의존 패키지

```bash
pip install transformers accelerate bitsandbytes
```
