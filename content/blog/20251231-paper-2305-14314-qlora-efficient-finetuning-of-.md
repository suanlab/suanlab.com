---
title: "[논문 리뷰] QLoRA: Efficient Finetuning of Quantized LLMs"
date: "2025-12-31"
excerpt: "We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLo..."
category: "Paper Review"
tags: ["Paper Review","cs.LG","cs.LG"]
thumbnail: "/assets/images/blog/20251231-paper-2305-14314-qlora-efficient-finetuning-of-.jpg"
---

# [논문 리뷰] QLoRA: Efficient Finetuning of Quantized LLMs

## TL;DR

QLoRA는 대규모 언어 모델(LLM)을 단일 GPU에서 효율적으로 미세 조정할 수 있는 혁신적인 방법론입니다. 이 방법은 4비트로 양자화된 사전 학습된 모델을 사용하여 메모리 사용량을 크게 줄이면서도 높은 성능을 유지합니다. QLoRA 기반 모델인 Guanaco는 Vicuna 벤치마크에서 ChatGPT 성능의 99.3%를 달성했습니다. 이 연구는 데이터셋 품질의 중요성을 강조하며, QLoRA를 활용한 효율적인 미세 조정 방법론을 제시합니다.

## 연구 배경 및 동기

대규모 언어 모델(LLM)은 자연어 처리 분야에서 혁신적인 발전을 이끌어왔지만, 이러한 모델을 미세 조정하는 데는 막대한 계산 자원과 시간이 필요합니다. 특히, 수십억 개의 파라미터를 가진 모델을 활용하는 연구자들은 높은 메모리 요구 사항과 비용 문제에 직면하게 됩니다. 기존의 16비트 기반 미세 조정 방법은 성능을 유지하기 위해 많은 메모리를 소모하며, 이는 연구자들이 대규모 모델을 실험적으로 활용하는 데 큰 장벽으로 작용합니다. 예를 들어, GPT-3와 같은 모델을 미세 조정하려면 상당한 GPU 자원이 필요하며, 이는 대부분의 개인 연구자나 소규모 팀에게는 부담스러운 비용입니다.

이러한 문제를 해결하기 위해, QLoRA는 4비트 양자화 기법을 도입하여 메모리 사용량을 획기적으로 줄이면서도 기존의 성능을 유지하는 방법을 제안합니다. 이를 통해 대규모 모델을 단일 48GB GPU에서도 효과적으로 미세 조정할 수 있게 됩니다. QLoRA의 핵심은 4비트로 양자화된 사전 학습된 모델을 활용하고, LoRA(Low-Rank Adapters)를 통해 그래디언트를 역전파하는 것입니다. 이러한 접근 방식은 대규모 언어 모델의 연구와 개발을 보다 접근 가능하게 만들어 줍니다.

## 관련 연구

기존의 연구는 주로 16비트 또는 32비트 정밀도를 사용한 미세 조정 방법에 집중되어 있었습니다. 이러한 방법들은 모델의 성능을 유지하는 데는 효과적이었지만, 메모리 사용량이 매우 높아 대규모 모델의 실험에 제약이 있었습니다. 특히, LoRA(Low-Rank Adaptation)와 같은 방법은 모델의 선형 계층에 작은 파라미터 세트를 추가하여 메모리 효율성을 높이는 데 기여했지만, 여전히 높은 정밀도의 데이터 타입을 사용해야 했습니다. 예를 들어, LoRA는 모델 파라미터 수를 줄여 메모리 사용량을 개선하지만, 활성화 값(activations)과 그래디언트(gradients)는 여전히 높은 정밀도로 유지해야 합니다.

QLoRA는 이러한 기존 연구와 달리, 4비트 양자화를 통해 메모리 사용량을 더욱 줄이면서도 성능을 유지하는 데 성공했습니다. 이는 기존의 양자화 기법보다 더 높은 정확도를 제공하며, 대규모 모델을 보다 효율적으로 활용할 수 있는 가능성을 열어줍니다. 또한, QLoRA는 다양한 데이터셋과 모델 크기에 대한 실험을 통해 데이터셋 품질이 모델 성능에 미치는 영향을 분석하여, 데이터셋 큐레이션의 중요성을 강조합니다. 예를 들어, 단순히 데이터셋의 크기를 늘리는 것보다 고품질의 데이터를 선별하여 사용하는 것이 모델 성능 향상에 더 효과적이라는 것을 보여줍니다.

## 제안하는 방법론

### 핵심 아이디어 상세 설명

QLoRA의 핵심 아이디어는 4비트로 양자화된 사전 학습된 모델을 활용하여 메모리 사용량을 줄이고, LoRA 모듈을 통해 미세 조정을 수행하는 것입니다. 이를 통해 대규모 모델을 단일 GPU에서 효율적으로 미세 조정할 수 있습니다.

1. **4비트 NormalFloat (NF4)**: QLoRA는 정규 분포된 가중치에 정보 이론적으로 최적화된 새로운 데이터 타입인 NF4를 도입합니다. 이는 기존의 4비트 양자화 방식보다 더 높은 정확도를 제공하며, 언어 모델의 성능 저하를 최소화합니다. NF4는 양자화 레벨을 비균등하게 배치하여, 가중치 분포에 더 적합하게 설계되었습니다. 예를 들어, FP16에 비해 메모리 사용량을 4배 줄이면서도 성능은 거의 동일하게 유지할 수 있습니다.

2. **이중 양자화**: 양자화 상수를 다시 양자화하여 추가적인 메모리 절감을 달성합니다. 일반적인 양자화는 가중치를 낮은 정밀도로 표현하는 것 외에도, 각 블록의 스케일링 팩터(양자화 상수)를 저장해야 합니다. 이중 양자화는 이 스케일링 팩터마저 양자화하여 메모리 오버헤드를 더욱 줄입니다. 이는 특히 모델 크기가 클수록 효과적입니다.

3. **페이지드 옵티마이저**: 미세 조정 중에 발생하는 메모리 스파이크를 효율적으로 관리하여 메모리 사용량을 최적화합니다. GPU 메모리가 부족할 경우, 활성 상태가 아닌 데이터를 CPU 메모리로 이동시켜 GPU 메모리를 확보합니다. 필요할 때 다시 GPU 메모리로 데이터를 불러오는 방식으로, 전체 훈련 과정을 중단 없이 진행할 수 있도록 돕습니다. 이는 `bitsandbytes` 라이브러리의 기능으로 구현되어 있습니다.

### 모델 아키텍처 구조

QLoRA는 사전 학습된 대규모 언어 모델의 선형 계층에 LoRA 모듈을 추가하여 미세 조정을 수행합니다. LoRA는 모델의 가중치를 고정하고, 학습 가능한 작은 파라미터 세트(LoRA 모듈)만 업데이트하여 메모리 사용량을 줄이고 훈련 속도를 높입니다. LoRA 모듈은 원래 가중치 행렬의 낮은 랭크(low-rank) 근사치를 학습하여, 전체 모델을 미세 조정하는 것보다 훨씬 적은 파라미터만 업데이트합니다.

### 핵심 수식과 알고리즘 설명

LoRA를 통한 미세 조정은 다음과 같은 수식을 따릅니다:

$$
W = W_0 + \Delta W
$$

여기서 \( W_0 \)는 사전 학습된 모델의 가중치이고, \( \Delta W \)는 LoRA 모듈을 통해 학습된 가중치 변화입니다. LoRA 모듈은 다음과 같이 정의됩니다:

$$
\Delta W = A \cdot B
$$

여기서 \( A \)와 \( B \)는 각각 작은 크기의 행렬로, LoRA 모듈의 학습 가능한 파라미터입니다. \( A \)는 \( d \times r \) 크기를 가지고, \( B \)는 \( r \times k \) 크기를 가집니다. 여기서 \( d \)는 입력 차원, \( k \)는 출력 차원, \( r \)은 LoRA의 랭크(rank)입니다. 랭크 \( r \)은 LoRA 모듈의 표현력을 조절하는 하이퍼파라미터입니다.

### Python/PyTorch 코드 예제

```python
import torch
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# 모델 로드 (예시: facebook/opt-350m 모델 사용)
model_name = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    quantization_config=bnb. quantization_config.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ),
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# LoRA 설정
config = LoraConfig(
    r=8, # LoRA rank
    lora_alpha=32, # Scaling factor
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"] # Linear 레이어 이름 지정 (모델 구조에 따라 변경 필요)
)

# 모델 준비 (k-bit training을 위한)
model = prepare_model_for_kbit_training(model)

# QLoRA 모델 생성
model = get_peft_model(model, config)

# 데이터셋 로드 (예시: "Abirate/english_quotes" 데이터셋 사용)
dataset_name = "Abirate/english_quotes"
dataset = load_dataset(dataset_name, split="train")

# 데이터 전처리 함수
def tokenize_function(examples):
    return tokenizer(examples["quote"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["author", "quote"])

# 훈련 설정
training_args = TrainingArguments(
    output_dir="qlora-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=50,
    gradient_checkpointing=True,
    fp16=True,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    push_to_hub=False,
)

# 훈련 (예시: transformers Trainer 사용)
trainer = Trainer(
    model=model,
    train_dataset=tokenized_datasets,
    args=training_args,
    data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                               'attention_mask': torch.stack([f['attention_mask'] for f in data])},
)

model.config.use_cache = False  # Gradient checkpointing requires disabling cache
trainer.train()
```

**참고:** 위 코드는 예시이며, 실제 모델 및 데이터셋에 맞게 수정해야 합니다. 특히 `target_modules`는 모델 구조에 따라 적절한 레이어 이름을 지정해야 합니다. 또한, `bnb_4bit_compute_dtype`은 GPU 환경에 따라 `torch.bfloat16` 또는 `torch.float16`으로 설정해야 합니다.

## 실험 설정

### 데이터셋 설명

QLoRA를 평가하기 위해 다양한 명령어 튜닝 데이터셋이 사용되었습니다. 주요 데이터셋은 다음과 같습니다:

- **Self-Instruct**: 모델 스스로 생성한 데이터를 활용하여 학습합니다. 이는 모델이 다양한 스타일과 주제에 대해 학습할 수 있도록 돕습니다.
- **Alpaca**: Stanford에서 공개한 instruction-following 데이터셋으로, OpenAI의 text-davinci-003을 이용하여 생성되었습니다. Alpaca는 다양한 instruction-response 쌍을 포함하고 있어, 모델이 명령어에 따라 적절한 응답을 생성하는 능력을 향상시키는 데 도움이 됩니다.
- **Unnatural Instructions**: 사람이 의도적으로 생성한, 일반적이지 않은 명령어 데이터셋입니다. 모델의 견고성(robustness)을 향상시키는 데 도움이 됩니다. 예를 들어, 모호하거나 비논리적인 명령어에 대해서도 합리적인 응답을 생성하도록 훈련할 수 있습니다.
- **Longform**: 긴 텍스트를 생성하는 데 특화된 데이터셋입니다. 모델이 일관성 있고 논리적인 긴 텍스트를 생성하는 능력을 향상시키는 데 도움이 됩니다.
- **Chip2**: 코드 생성 및 이해 능력을 향상시키기 위한 데이터셋입니다.

### 평가 지표

모델의 성능은 MMLU (Massive Multitask Language Understanding) 5-shot dev set을 사용하여 평가되었습니다. MMLU는 다양한 주제에 대한 지식을 평가하는 벤치마크로, 5-shot은 모델에게 5개의 예시를 제공하고, 그 다음 질문에 대한 답변을 예측하도록 하는 방식입니다. MMLU는 인문학, 사회과학, 자연과학 등 다양한 분야의 질문을 포함하고 있어, 모델의 전반적인 지능을 평가하는 데 유용합니다.

### 비교 대상 (baseline)

QLoRA의 성능은 기존의 16비트 기반 미세 조정 방법과 비교되었습니다. 특히, ChatGPT와 같은 대규모 모델과의 성능 비교를 통해 QLoRA의 효율성을 검증했습니다. 또한, 동일한 데이터셋과 모델 아키텍처를 사용하되, 양자화 없이 FP16으로 미세 조정한 모델과의 비교를 통해 QLoRA의 성능 손실을 정량적으로 평가했습니다.

### 하이퍼파라미터 설정

- **데이터 타입**: NF4 (NormalFloat4) 데이터 타입을 사용하여 메모리 사용량을 줄이고, bf16 (bfloat16) 계산 데이터 타입을 사용하여 훈련 속도를 높였습니다. BF16은 FP16보다 더 넓은 범위를 표현할 수 있어, 훈련 안정성을 높이는 데 도움이 됩니다.
- **LoRA 설정**: LoRA의 rank (r)는 64, scaling factor (α)는 16으로 설정되었습니다. Rank는 LoRA 모듈의 차원을 결정하며, scaling factor는 LoRA 업데이트의 크기를 조절합니다. 일반적으로 α/r 비율을 조정하여 학습률을 조절하는 효과를 얻을 수 있습니다. LoRA rank를 높이면 모델의 표현력이 증가하지만, 메모리 사용량도 증가합니다.

## 실험 결과 및 분석

### 주요 정량적 결과

실험 결과, QLoRA를 사용하여 미세 조정된 모델은 ChatGPT 성능의 99.3%를 달성했습니다. 이는 Vicuna 벤치마크에서 이전의 모든 공개 모델을 능가하는 성과입니다. 특히, 단일 GPU에서 24시간 미세 조정만으로 이러한 성능을 달성한 것은 QLoRA의 메모리 효율성과 성능을 동시에 입증하는 결과입니다. 또한, QLoRA는 FP16으로 미세 조정한 모델과 비교하여 성능 저하가 거의 없음을 보여주었습니다.

### 정성적 분석

QLoRA를 통해 미세 조정된 모델은 다양한 데이터셋에서 우수한 성능을 보였습니다. 특히, 데이터셋의 크기보다는 데이터셋의 품질이 MMLU 성능에 더 큰 영향을 미쳤습니다. 이는 고품질의 데이터가 모델 학습에 더 중요하다는 것을 시사합니다. 예를 들어, Alpaca 데이터셋을 큐레이션하여 노이즈를 제거하고, 더 다양한 instruction-response 쌍을 추가한 결과, 모델의 성능이 크게 향상되었습니다. 또한, 인간 평가 및 GPT-4 평가를 통해 챗봇 성능을 비교 분석한 결과, GPT-4 평가가 인간 평가의 저렴하고 합리적인 대안이 될 수 있음을 발견했습니다. GPT-4는 인간 평가자와 유사한 수준의 일관성을 보여주었으며, 대규모 모델의 자동 평가에 유용하게 활용될 수 있습니다.

### Ablation study 결과

Ablation study를 통해 NF4 데이터 타입과 이중 양자화의 효과를 분석한 결과, 메모리 사용량을 크게 줄이면서도 성능 저하가 거의 없음을 확인했습니다. 이는 QLoRA의 혁신적인 기술들이 실제로 메모리 효율성을 향상시키는 데 기여함을 보여줍니다. 예를 들어, NF4를 사용하지 않고 FP16으로 양자화한 경우, 성능이 크게 저하되었으며, 이중 양자화를 사용하지 않은 경우, 메모리 사용량이 증가했습니다.

## 한계점 및 향후 연구 방향

### 저자가 언급한 한계점

QLoRA는 메모리 사용량을 획기적으로 줄이면서도 성능을 유지하는 데 성공했지만, 여전히 일부 데이터셋에서는 성능 저하가 발생할 수 있습니다. 특히, 데이터셋의 특성에 따라 모델의 성능이 달라질 수 있으며, 이는 데이터셋 큐레이션의 중요성을 강조합니다. 또한, QLoRA는 LoRA 모듈을 추가하여 미세 조정을 수행하므로, LoRA의 랭크(rank)를 적절하게 설정하는 것이 중요합니다. 랭크가 너무 낮으면 모델의 표현력이 제한될 수 있으며, 랭크가 너무 높으면 메모리 사용량이 증가할 수 있습니다.

### 잠재적인 개선 방향

앞으로는 데이터셋의 품질을 개선하고, 다양한 평가 방법을 통해 모델의 성능을 향상시키는 연구가 더욱 활발해질 것으로 기대됩니다. 예를 들어, Active Learning 기법을 활용하여 모델이 학습하기 어려운 샘플을 선별하고, 해당 샘플에 대한 추가적인 데이터를 수집하여 데이터셋을 개선할 수 있습니다. 또한, QLoRA의 기술을 다른 유형의 모델에 적용하여 메모리 효율성을 높이는 연구가 필요합니다. 예를 들어, 이미지 생성 모델이나 비전-언어 모델(Vision-Language Model)에 QLoRA를 적용하여, 단일 GPU에서도 대규모 모델을 훈련할 수 있도록 할 수 있습니다. 또한, QLoRA와 다른 양자화 기법(예: SmoothQuant, GPTQ)을 결합하여 메모리 효율성을 더욱 높이는 연구도 진행될 수 있습니다.

## 결론 및 시사점

QLoRA는 메모리 사용량을 획기적으로 줄이면서도 성능을 유지하는 혁신적인 기술들을 통해 대규모 언어 모델 미세 조정을 더욱 접근 가능하게 만들었습니다. 연구팀은 모든 모델과 코드를 공개했으며, 4비트 훈련을 위한 CUDA 커널도 포함되어 있습니다. QLoRA는 연구자와 개발자 모두에게 LLM 연구 및 개발에 대한 새로운 가능성을 열어줄 것으로 기대됩니다. 특히, 데이터셋 품질의 중요성을 강조하고, 효율적인 미세 조정 방법론을 제시함으로써, 대규모 모델의 연구와 개발을 보다 접근 가능하게 만들어 줍니다. QLoRA는 앞으로 대규모 언어 모델 연구의 democratizing에 크게 기여할 것으로 예상됩니다.

---

**참고 자료:**

* QLoRA 논문: [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
* QLoRA GitHub 저장소: [https://github.com/artidoro/qlora](https://github.com/artidoro/qlora)

**관련 자료:**

* LoRA (Low-Rank Adaptation): [https://arxiv.org/abs/2106.09698](https://arxiv.org/abs/2106.09698)
* Vicuna 벤치마크: [https://vicuna.lmsys.org/](https://vicuna.lmsys.org/)
