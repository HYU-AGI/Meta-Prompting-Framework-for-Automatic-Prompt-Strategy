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

## 💻 실행 방법
### 1. dataset 준비
- 자세한 내용은 [README.md](data/README.md)를 참고해주세요.

### 2. meta-prompt 생성
```
python src/main.py --model_name "model_name" --dataset_name "dataset_name" --delta_gain_coef 0.07 --alpha_neg_cap 1.0
```
meta-prompt는 다음과 같은 연산을 거쳐 생성됩니다:
- Step 1: Self-Perplexity 계산
- Step 2: prompt module 선택 -> 선택된 module을 input prompt에 맞게 adaption -> adapted module을 사용해 reasoning 과정을 단계적으로 생성하여 meta-prompt 생성 완료


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
