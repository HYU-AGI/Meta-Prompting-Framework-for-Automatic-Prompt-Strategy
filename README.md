# AGI 발현을 위한 메타인지 프레임워크 핵심기술 개발 및 실증
## AGI 발현을 위한 Planner에 대한 연구 개발
### 입력 프롬프트를 최적으로 처리하기 위한 프롬프트 라이브러리 운용을 AI가 결정하는 메타프롬프팅 기법
### 💡 예시
![image](./image/example.png)

## ⚙️ Requirements
To install requirements:
```
pip install -r requirements.txt
```

## 💻 Usage Guide
### 1. Dataset 준비
- 자세한 내용은 [README.md](data/README.md)를 참고해주세요.

### 2. Meta-prompt 생성
```
python src/main.py --model_name "model_name" --dataset_name "dataset_name" --delta_gain_coef 0.07 --alpha_neg_cap 1.0
```
meta-prompt는 다음과 같은 연산을 거쳐 생성됩니다:
- Step 1: Self-Perplexity 계산
- Step 2: prompt module 선택 -> 선택된 module을 input prompt에 맞게 adaption -> adapted module을 사용해 reasoning 과정을 단계적으로 생성하여 meta-prompt 생성 완료

## 🧠 작동 원리
**1️⃣ Self-Perplexity(SPP) 값 측정** \
주어진 입력(input)에 대해 Perplexity를 계산하여, \
모델이 해당 입력을 얼마나 확신(Certain) 또는 불확신(Uncertain) 하는지를 측정합니다.
- SPP 값이 작을수록: 모델이 입력을 충분히 이해하고 있음 → 내부 지식만으로 해결 가능
- SPP 값이 클수록: 모델이 입력을 해석하기 어렵거나 자신감이 낮음 → 외부 정보 필요

따라서 SPP가 큰 경우, 모델은 자체 지식만으로는 답변이 어려운 상황으로 간주되며, \
총 40개의 Prompt Module Library 중 RAG 관련 모듈을 선택하도록 유도합니다. \
이 과정을 통해 외부 지식(Retrieval Augmented Generation) 기반 보강을 수행합니다.

**2️⃣**

### Reference
[Self-Discover: Large Language Models Self-Compose Reasoning Structures](https://proceedings.neurips.cc/paper_files/paper/2024/file/e41efb03e20ca3c231940a3c6917ef6f-Paper-Conference.pdf)
```
@inproceedings{NEURIPS2024_e41efb03,
 author = {Zhou, Pei and Pujara, Jay and Ren, Xiang and Chen, Xinyun and Cheng, Heng-Tze and Le, Quoc V. and H., Ed and Zhou, Denny and Mishra, Swaroop and Zheng, Huaixiu Steven},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {126032--126058},
 publisher = {Curran Associates, Inc.},
 title = {SELF-DISCOVER: Large Language Models Self-Compose Reasoning Structures},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/e41efb03e20ca3c231940a3c6917ef6f-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```
