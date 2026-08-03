# kt-aivle-big-proj-model-rgb

KT AIVLE 빅프로젝트 16조 — RGB 원통 셀 결함 탐지 모델 학습

## 이 레포의 역할

RGB 이미지 결함 탐지 모델(YOLOv11-seg)을 **학습**합니다. 서빙하지 않습니다.

산출물인 가중치는 S3 `models/` 프리픽스로 올라가고, 서빙은 `kt-aivle-big-proj-ai-infer` 가 기동 시 그것을 내려받아 씁니다.

## 레포 명명 규칙

| 접두어 | 성격 | 예 |
| --- | --- | --- |
| `model-` | 학습·실험. 컨테이너가 되지 않습니다 | `model-ct` · `model-rgb` · `model-quality` |
| `data-` | 데이터 생성·전처리 | `data-preprocessing` · `data-augmentation` |
| 그 외 | 서빙 컨테이너 | `ai-infer` · `vlm` · `backend` · `frontend` |

## 계약 제약

정본은 노션 「시스템 계약서 Core」입니다. 아래는 이 레포에 직접 걸리는 항목만 옮긴 것이며, 어긋나면 Core 가 이깁니다.

| 항목 | 값 |
| --- | --- |
| SAHI | **RGB 는 OFF** 입니다. plain F1 0.98 이고 SAHI 는 오탐만 늘렸습니다 (CT 만 ON, slice 1280 고정) |
| 학습 환경 | RunPod A40 |
| 서비스 추론 | EC2 |
| 결함 유형 | Core §6.5 의 4종이 전부입니다. 새 값을 임의로 만들지 않습니다 |
| bbox 형식 | `{ x, y, width, height }` (Float) |
| 판정 값 | 이 모델이 내는 것은 `PASS` · `REJECT` 입니다. `FAIL` 은 촬영 품질 분류기가 냅니다 |

## 서빙으로 넘길 때 필요한 것

학습 노트북만으로는 `ai-infer` 에 실을 수 없습니다. 아래 3개가 있어야 합니다.

1. 가중치 파일 (S3 `models/` 업로드)
2. 전처리 로직 — 학습 노트북에서 떼어내 재사용 가능한 함수로
3. 추론 인터페이스 — 이미지 1장 → `{ label, confidence, defects[] }`

## 관련 레포

- `kt-aivle-big-proj-model-ct` — CT 탐지 모델 학습
- `kt-aivle-big-proj-model-quality` — 촬영 품질 분류기 (YOLO 앞단)
- `kt-aivle-big-proj-ai-infer` — 서빙 컨테이너
- `kt-aivle-big-proj-data-augmentation` — 합성 결함 데이터 생성
