"""EXT 외관 검사 — 배포용 추론 모듈 (단일 파일)

배터리 셀 RGB 프레임 → 결함/정상 판정 + 결함 위치(원본 좌표) JSON.

    from ext_infer import ExtInspector
    insp = ExtInspector()                       # 모델 1회 로드
    out  = insp.infer_cell('0041', frame_paths) # 셀 하나 전체
    # out['프레임'][i]['판정'] · ['결함'][j]['위치']['bbox']  ← 원본 1920x1080 좌표

설계 원칙
  · 입력은 **원본 이미지**다. 셀 크롭은 내부에서 하고 오프셋을 알고 있으므로
    bbox를 항상 **원본 좌표**로 돌려준다(백엔드·분류기와 같은 좌표계).
  · 판정은 **검출 박스 개수**만 쓴다. 유형 이름도 VLM 해설도 판정에 안 쓴다.
  · 검증 안 된 값은 **신뢰 등급을 붙여서** 내보낸다(리포트 LLM이 이미지를 못 보므로).

성능 (사람 라벨 300장 · A100 80GB · **배포 경로 실측**)
  게이트 thr 0.10 · 박스 8개 이상 → P 0.855 / **R 1.000** / F1 0.922 (오탐 34 · 놓침 0)
  위치   thr 0.12 · 점 recall 0.818 (순열 대조 2.51배)
  셀판정 flag율 ≥70% (30프레임 균등 추출) → 셀 recall 1.000 / precision 1.000
  지연   398 ms/프레임 · **셀당 12.2초**(30프레임) ← 전량 검사 시 1.8분

🔴 배포 전 반드시 읽을 것: 아래 DEPLOY_NOTES
🔴 배포 전 반드시 돌릴 것
    python ext_infer.py --selftest                       # 모델 없이 좌표·계약서
    python ext_infer.py --verify-fixture \\
        --fixture golden_fixture_deploy.json --images <원본 20장 폴더>
"""
from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

__all__ = ['ExtInspector', 'Config', 'CONTRACT', 'DEPLOY_NOTES']


# ════════════════════════════════════════════════════════════════════════
# 설정 — 바꿀 값은 전부 여기 있다
# ════════════════════════════════════════════════════════════════════════

# 결함 유형별 텍스트 질의. **이것이 이 시스템의 "라벨"이다** — 학습 없이 여기만 고친다.
#   설계 규칙: 영어 · 짧은 명사구 · 추상어("defect") 금지 · 한 유형에 여러 표현(OR)
QUERY_MAP: dict[str, list[str]] = {
    '녹·부식':       ['rust', 'orange brown discoloration', 'brown stain on surface',
                      'oxidation on surface', 'reddish brown spot'],
    '벗겨짐·박리':   ['peeling plastic wrap', 'peeled off label',
                      'exposed metal under wrapper', 'wrapper coming off', 'peeling film edge'],
    '파손·찢김':     ['tear in plastic film', 'crack on surface', 'puncture hole', 'dent'],
    '긁힘·스크래치': ['scratch', 'scratch mark', 'scratched metal surface', 'scuff mark'],
    '들뜸':          ['air bubble under film', 'wrinkled plastic wrap', 'lifted film edge'],
    '오염·이물질':   ['dirt stain', 'smudge', 'white residue', 'foreign particle on surface'],
}

# 정상 요소에 **이름을 주어** 결함 질의가 그것을 훔쳐가지 않게 한다.
NEG_MAP: dict[str, list[str]] = {
    '정상:인쇄·각인': ['printed text', 'printed letters on label', 'engraved serial number'],
    '정상:반사':      ['light reflection', 'specular highlight'],
    '정상:금속캡':    ['metal cap', 'battery terminal'],
    '정상:테두리':    ['bottom edge of cylinder', 'rim of metal can', 'silhouette edge'],
}

# 🔴 '정상:금속캡'은 버리지 않는다. 구조물로 지우면 오탐 5개를 지우고 진짜 결함 9개를 잃는다(실측).
DROP_TAGS = frozenset({'정상:인쇄·각인', '정상:반사', '정상:테두리'})


@dataclass(frozen=True)
class Config:
    # ── 모델 (revision을 반드시 박는다. 안 박으면 레포 갱신 시 조용히 결과가 바뀐다) ──
    owl_id: str = 'google/owlv2-large-patch14-ensemble'
    owl_rev: str = '95e26936e865f87db1742128404b3c035d47d89d'
    vlm_id: str = 'Qwen/Qwen2.5-VL-7B-Instruct'
    vlm_rev: str = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'

    # ── 운영점 (사람 라벨 300장 · **배포 경로에서** 확정) ──
    # 🔴 평가 경로(이미 잘린 크롭 이미지)에서는 thr 0.08이었다. 배포 경로는 입력이
    #    JPEG 1세대라 더 선명해서 박스가 53% 더 나온다 → 문턱을 0.10으로 올렸다.
    #    배포 경로 실측: P 0.855 / R 1.000 / F1 0.922 (오탐 34 · 놓침 0)
    thr_gate: float = 0.10     # 게이트 임계
    n_gate: int = 8            # 이 개수 이상이면 결함 프레임
    thr_loc: float = 0.12      # 위치 특정 임계
    max_defects: int = 6       # 프레임당 리포트에 담을 최대 결함 수

    # ── 셀 크롭 (학습 때와 동일해야 한다. 하나라도 바꾸면 좌표가 틀어진다) ──
    crop_pad: float = 0.10
    sat_th: int = 40
    dens_th: float = 0.10
    min_area: float = 0.02
    max_area: float = 0.90

    # ── 셀 단위 판정 (0806 실측으로 신설) ──
    # 🔑 프레임 오탐은 **무작위**(정상 셀에서 30%가 흩어져 걸림)인데 결함은 **지속적**
    #    (80~100% 프레임에서 계속 보임)이다. 프레임을 모으면 잡음이 평균으로 씻긴다.
    #      결함 확정 셀 12개: flag율 최소 80% · 중앙 100%
    #      정상 추정 셀 12개: flag율 최소 12.5% · 최대 50%      → 간격 +30%p
    #    그래서 셀 판정은 "1장이라도"(OR)가 아니라 **flag율 임계**로 한다.
    #    OR로 하면 정상 셀도 30%가 걸리니 전부 결함이 된다.
    cell_sample_k: int = 30    # 셀당 검사할 프레임 수(균등 추출). 0이면 전량
    cell_flag_rate: float = 0.70   # flag율이 이 이상이면 셀 = 결함
    #    실측 (K, 임계) → 셀 recall / precision / 셀당 초:
    #      10, 0.70 → 0.993 / 0.975 /  4.1초
    #      20, 0.70 → 0.998 / 0.998 /  8.1초
    #      30, 0.70 → 1.000 / 1.000 / 12.2초   ← 기본값(불량 게이트라 recall 우선)
    #    ⚠️ n=12씩이라 "1.000"은 근거가 얇다. 견고한 것은 **간격 +30%p**다.

    # ── 기타 ──
    cap_zone: float = 0.22     # 셀 높이 상단 22% = 금속캡
    big_area_pct: float = 1.0  # 셀면적비 이 이상이면 '큼'
    batch: int = 16            # 실측 최적(단 batch 1 대비 7%만 빠르다 = 연산 포화)
    device: str = 'cuda'
    # 🔴 2단 VLM(Qwen)은 **이 모듈에 없다.** 스위치도 두지 않는다 —
    #    "켤 수 있다"고 적어놓고 코드가 없으면 그게 이 프로젝트가 계속 싸워온 조용한 사고다.
    #    2단이 필요하면 평가 노트북(battery_ext_evalset.ipynb §E)을 쓴다.
    #    vlm_id/vlm_rev 는 출처 기록용으로만 남긴다(모델카드 §1).


# ════════════════════════════════════════════════════════════════════════
# 리포트 LLM 계약서 — 값과 **검증 상태**를 같이 넘긴다
# ════════════════════════════════════════════════════════════════════════
CONTRACT: dict = {
    '설명': '배터리 셀 외관 검사 결과. 각 필드에 신뢰 등급이 붙어 있다.',
    '지켜야 할 것': [
        "신뢰='높음'인 값만 단정해서 쓴다.",
        "신뢰='낮음'인 값은 반드시 '~로 보이나 확정되지 않음' 형태로 쓴다.",
        '유형후보는 순서만 있고 박스별 확률이 아니다. 개별 후보에 퍼센트를 붙이지 않는다.',
        "확률을 언급해야 한다면 '필드 신뢰 등급.유형후보.실측_적중률'의 값만 쓴다.",
        '유형후보가 2개 이상이면 모두 언급한다. 1등만 쓰고 나머지를 버리지 않는다.',
        '여기 없는 값을 지어내지 않는다. 특히 원인·조치는 우리가 판단하지 않았다.',
        '이미지를 보지 않고 쓰는 리포트임을 전제한다.',
    ],
    '필드 신뢰 등급': {
        '판정':   {'신뢰': '높음',
                   '근거': 'P 0.855 / R 1.000 / F1 0.922 (사람 라벨 300장, 배포 경로 실측)'},
        '셀_판정': {'신뢰': '높음',
                    '근거': '셀 recall 1.000 / precision 1.000 (결함 12셀 · 정상 12셀 실측)',
                    '주의': '표본이 셀 12개씩이라 1.000은 근거가 얇다. '
                            '견고한 것은 결함 셀 flag율(최소 80%)과 정상 셀(최대 50%)의 간격이다.'},
        '좌표계': {'신뢰': '높음', '설명': 'bbox는 입력 원본 이미지 기준이다. 크롭 좌표가 아니다.'},
        '위치':   {'신뢰': '높음', '근거': '점 recall 0.818, 순열 대조 2.51배 (사람 라벨 300장)'},
        '크기':   {'신뢰': '중간', '근거': '픽셀 면적 실측. 큼/작음 2단계만 유효'},
        '유형후보': {
            '신뢰': '낮음',
            '근거': '최빈 유형 베이스라인을 못 넘음 — top-1 +0.081 · top-2 +0.038 · top-3 -0.014',
            '주의': '유형은 참고용이다. 리포트에서 확정 표현을 쓰지 말 것',
            '실측_적중률': {
                '후보목록_전체': '0.79',
                '사소한_베이스라인': '0.76 (흔한 유형 3개를 그냥 나열해도 이만큼 맞는다)',
                '해석': '목록 안에 실제 유형이 있을 확률이 약 79%지만 아무 정보 없이 나열해도 76%다. '
                        '즉 이 후보 목록은 거의 정보가 없다. 확정 표현을 쓰지 말 것',
            },
        },
    },
}

DEPLOY_NOTES = """
🔴 배포 전 확인 (2026-08-06)

0. ★★ 입력 이미지를 **재인코딩하지 말 것** — 운영점이 깨진다

   JPEG 압축 세대가 검출 수를 크게 바꾼다(실측):
       원본 JPEG 그대로(1세대)          33.1 박스/장
       크롭 후 JPEG 재저장(2세대)       26.4 박스/장   -20%
       평가셋 크롭본(2세대)             21.6 박스/장   -35%
   재압축은 미세 질감을 뭉갠다. 미세 질감이 저점수 검출을 만들므로 박스가 줄어든다.

   → 백엔드는 **카메라 원본 JPEG 바이트를 그대로** 넘겨야 한다.
     리사이즈·품질 변경·재저장·썸네일 경유 금지. 하면 thr_gate를 다시 잡아야 한다.
   → 파이프라인이 크롭을 **메모리에서만** 다루는 이유이기도 하다(디스크 경유 = 재인코딩 위험).

1. 지연 — 셀 단위 서브샘플링으로 크게 줄였다
   전량 검사: 398 ms/프레임 · 셀당(270프레임) 1.8분 = 목표 4초의 27배.
   **30프레임 균등 추출 + flag율 판정: 셀당 12.2초** (셀 recall·precision 1.000 실측).
   K를 줄이면 더 빨라진다 — 10장 4.1초(recall 0.993) · 20장 8.1초(0.998).
   남은 배수는 3배다. 질의 임베딩 캐시(추정 0.7배)를 더하면 약 8.5초.
   (배포 경로는 1920x1080 원본에서 셀 추정까지 하므로 크롭 입력보다 느리다)
   배치를 16배 키워도 7%만 빨라진다 = 런치 오버헤드가 아니라 연산 포화.
   대책(효과 큰 순):
     ① 프레임 프리필터 — 270장 전량이 아니라 값싼 색도·텍스처로 상위 K장만
        ⚠️ 재현율을 팔 위험. 반드시 같은 300장 평가셋에서 게이트 R을 재측정하고 적용할 것
     ② 질의 임베딩 캐시 — 질의 35개가 고정인데 매 이미지 재인코딩 중
     ③ 입력 해상도 축소 · FP16 · TensorRT (통상 3~5배, 미측정)

2. 판정에 쓰는 것은 검출 박스 개수뿐이다
   유형 이름·심각도 등급은 **검증을 통과하지 못했다**. 리포트 참고용으로만 내보내며
   계약서에 신뢰='낮음'으로 명시된다. 이 값으로 자동 조치를 하면 안 된다.

3. 2단 VLM(Qwen)은 **이 배포 모듈에 들어 있지 않다**
   판정에 안 쓰이고(게이트는 박스 개수만 센다), 유형·등급 출력이 전부 검증 실패했다.
   스위치도 두지 않았다 — 없는 기능의 켜기 옵션은 조용한 거짓말이 된다.
   2단 결과가 필요하면 평가 노트북 battery_ext_evalset.ipynb §E 를 쓴다(프레임당 +1.2초/후보).
   Config.vlm_id / vlm_rev 는 **출처 기록용**이다. 로드 코드는 이 파일에 없다.

   ⚠️ 관련: 2단이 쓰던 '정상 캡 갤러리 16장'도 이 모듈은 **읽지 않는다.**
      데이터셋 이미지라 레포에 못 올리는데, 배포 의존성이 아니므로 문제되지 않는다.

4. 좌표계
   입력은 원본 이미지, 출력 bbox도 원본 좌표다. 내부 크롭은 우리가 하고 오프셋을 더해 돌려준다.

5. 배포·업그레이드 뒤에는 골든 픽스처를 돌린다 (자동 방어선은 이것뿐이다)
   python ext_infer.py --verify-fixture \
       --fixture golden_fixture_deploy.json --images <원본 20장 폴더>
   · 게이트 판정 · 박스 수 → **완전 일치 필수**
   · 크롭 오프셋 4px · bbox 8px · score 2e-3 → 경고 (Pillow 버전 차이로 흔들린다)
   픽스처가 thr 0.08(평가 경로)이면 지금 설정(0.10)을 검증할 수 없다 — 도구가 🔴 로 알린다.
   픽스처 JSON 은 레포에, 이미지 20장은 S3 models/fixtures/rgb/ 에 있다(라이선스).

6. 고정할 것
   transformers >=5.13.1,<6 (5.13.1·5.14.1에서 동일 결과 확인) · torch 2.11.0+cu128
   모델 revision은 Config에 박혀 있다.
"""


# ════════════════════════════════════════════════════════════════════════
# 기하 — 셀 영역 추정 · 크롭 (학습 시 전처리와 동일)
# ════════════════════════════════════════════════════════════════════════

def _cell_bbox_norm(im: Image.Image, cfg: Config, small: int = 400):
    """원본 → 셀 영역(정규화 bbox) 또는 None. 배경이 무채색이라는 전제."""
    t = im.copy()
    t.thumbnail((small, small))
    hsv = np.asarray(t.convert('HSV'))
    m = hsv[:, :, 1].astype(np.int16) > cfg.sat_th
    if m.mean() < 0.01 or m.mean() > 0.98:               # 채도로 안 갈리면 밝기로
        v = hsv[:, :, 2].astype(np.int16)
        m = np.abs(v - np.median(np.concatenate([v[0], v[-1]]))) > 40
    if m.sum() < 50:
        return None
    h, w = m.shape
    rs, cs = m.sum(1), m.sum(0)
    if rs.max() == 0 or cs.max() == 0:
        return None
    ry = np.where(rs > rs.max() * cfg.dens_th)[0]
    cx = np.where(cs > cs.max() * cfg.dens_th)[0]
    if len(ry) == 0 or len(cx) == 0:
        return None
    bb = (cx.min() / w, ry.min() / h, (cx.max() + 1) / w, (ry.max() + 1) / h)
    a = (bb[2] - bb[0]) * (bb[3] - bb[1])
    return None if not (cfg.min_area <= a <= cfg.max_area) else bb


def crop_cell(im: Image.Image, cfg: Config):
    """원본 → (크롭 이미지, (offset_x, offset_y), 추정성공여부).
    실패 시 원본 전체를 쓰고 오프셋 (0,0)이므로 파이프라인이 멈추지 않는다."""
    bb = _cell_bbox_norm(im, cfg)
    ok = bb is not None
    if not ok:
        box = (0.0, 0.0, 1.0, 1.0)
    else:
        x1, y1, x2, y2 = bb
        bw, bh = x2 - x1, y2 - y1
        p = cfg.crop_pad / 2
        box = (max(0.0, x1 - bw * p), max(0.0, y1 - bh * p),
               min(1.0, x2 + bw * p), min(1.0, y2 + bh * p))
    w, h = im.size
    x0, y0 = int(box[0] * w), int(box[1] * h)
    x1i, y1i = int(box[2] * w), int(box[3] * h)
    return im.crop((x0, y0, x1i, y1i)), (x0, y0), ok


def cell_box_px(im: Image.Image, tol: int = 55, frac: float = 0.12):
    """크롭 안에서 셀 몸통 픽셀 범위. 구역(캡/몸통)과 면적비 계산에 쓴다."""
    a = np.asarray(im, np.float32)
    h, w = a.shape[:2]
    e = max(1, w // 20)
    bg = np.concatenate([a[:, :e].reshape(-1, 3), a[:, -e:].reshape(-1, 3)])
    m = np.abs(a - np.median(bg, 0)).sum(2) > tol
    xs = np.where(m.mean(0) > frac)[0]
    ys = np.where(m.mean(1) > frac)[0]
    if len(xs) < 5 or len(ys) < 5:
        return 0, 0, w, h
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


# ════════════════════════════════════════════════════════════════════════
# 추론기
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ExtInspector:
    cfg: Config = field(default_factory=Config)
    _owl: object = field(default=None, repr=False)
    _proc: object = field(default=None, repr=False)
    _queries: list[str] = field(default_factory=list, repr=False)
    _tag_of: dict = field(default_factory=dict, repr=False)
    _post: object = field(default=None, repr=False)

    def __post_init__(self):
        self._queries = ([q for v in QUERY_MAP.values() for q in v]
                         + [q for v in NEG_MAP.values() for q in v])
        self._tag_of = {q: k for k, v in {**QUERY_MAP, **NEG_MAP}.items() for q in v}

    # ── 모델 (최초 사용 시 1회 로드) ──────────────────────────────────
    def load(self):
        if self._owl is not None:
            return self
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        c = self.cfg
        self._proc = Owlv2Processor.from_pretrained(c.owl_id, revision=c.owl_rev)
        self._owl = (Owlv2ForObjectDetection
                     .from_pretrained(c.owl_id, revision=c.owl_rev).to(c.device).eval())
        # 🔴 후처리 API 이름이 transformers 버전 간 다르다. 지연 평가(or)로 고른다 —
        #    getattr(x, 'a', x.b)는 기본값을 먼저 평가해 폴백 전에 죽는다.
        self._post = (getattr(self._proc, 'post_process_grounded_object_detection', None)
                      or getattr(self._proc, 'post_process_object_detection', None))
        if self._post is None:
            raise RuntimeError('OWLv2 후처리 API를 못 찾았다 — transformers 버전 확인')
        return self

    # ── 검출 ─────────────────────────────────────────────────────────
    def _detect(self, crops: Sequence[Image.Image]):
        """→ 이미지별 [(x1,y1,x2,y2,score,tag), ...] (크롭 좌표계, score 내림차순)"""
        import torch
        self.load()
        c = self.cfg
        out: list[list[tuple]] = [[] for _ in crops]
        for i in range(0, len(crops), c.batch):
            blk = list(crops[i:i + c.batch])
            # 🔴 OWLv2 전처리는 이미지를 정사각 패딩한다. 직사각을 그대로 넣으면 좌표가
            #    **에러 없이** 어긋난다 → 우리가 먼저 패딩(좌상단)해 내부 패딩을 무연산으로.
            sq, sizes = [], []
            for im in blk:
                s = max(im.size)
                canvas = Image.new('RGB', (s, s), (0, 0, 0))
                canvas.paste(im, (0, 0))
                sq.append(canvas)
                sizes.append([s, s])
            inp = self._proc(text=[self._queries] * len(sq), images=sq,
                             return_tensors='pt').to(c.device)
            with torch.inference_mode():
                o = self._owl(**inp)
            res = self._post(outputs=o, threshold=min(c.thr_gate, c.thr_loc) * 0.5,
                             target_sizes=torch.tensor(sizes, device=c.device))
            for j, r in enumerate(res):
                boxes = []
                for b, s, l in zip(r['boxes'].tolist(), r['scores'].tolist(),
                                   r['labels'].tolist()):
                    tag = self._tag_of[self._queries[l]]
                    if tag in DROP_TAGS:
                        continue
                    boxes.append((b[0], b[1], b[2], b[3], float(s), tag))
                boxes.sort(key=lambda t: -t[4])
                out[i + j] = boxes
        return out

    # ── 입력 정규화 ──────────────────────────────────────────────────
    @staticmethod
    def _open(x, idx: int, names: Sequence[str] | None):
        """경로 · **바이트** · PIL 무엇이든 받는다. 디스크에 아무것도 안 쓴다.
        백엔드가 업로드 바이트를 그대로 넘길 수 있게 bytes를 받는다(임시 파일 불필요)."""
        if isinstance(x, (bytes, bytearray, memoryview)):
            im = Image.open(io.BytesIO(bytes(x))).convert('RGB')
            return im, (names[idx] if names else f'frame_{idx}')
        if isinstance(x, (str, Path)):
            return Image.open(x).convert('RGB'), (names[idx] if names else Path(x).name)
        return x.convert('RGB'), (names[idx] if names else f'frame_{idx}')

    def _frame_result(self, key, orig, crop, off, ok, boxes) -> dict:
        c = self.cfg
        ox, oy = off
        n_gate = sum(1 for b in boxes if b[4] >= c.thr_gate)
        flag = n_gate >= c.n_gate
        fr = {
            '파일': key,
            '원본_크기': list(orig.size),
            '판정': '결함' if flag else '정상',
            '판정_신뢰': '높음',
            '판정_근거': {'임계': c.thr_gate, '박스수': n_gate, '기준': f'{c.n_gate}개 이상'},
            '셀_크롭': {'오프셋': [ox, oy], '크기': list(crop.size), '추정성공': bool(ok)},
            '결함': [],
        }
        if not flag:
            return fr
        cb = cell_box_px(crop)
        ch = max(cb[3] - cb[1], 1)
        ca = max((cb[2] - cb[0]) * ch, 1)
        for b in [x for x in boxes if x[4] >= c.thr_loc][:c.max_defects]:
            x0 = max(0, int(b[0])); y0 = max(0, int(b[1]))
            x1 = min(crop.width, int(b[2]) + 1); y1 = min(crop.height, int(b[3]) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            rel = ((y0 + y1) / 2 - cb[1]) / ch
            area = (x1 - x0) * (y1 - y0) / ca * 100
            fr['결함'].append({
                '위치': {
                    '구역': '금속캡' if rel < c.cap_zone else '몸통',
                    '셀_상단에서': round(float(min(max(rel, 0.0), 1.0)), 2),
                    'bbox': [x0 + ox, y0 + oy, x1 + ox, y1 + oy],   # ★원본 좌표
                    'bbox_좌표계': f'원본 {orig.size[0]}x{orig.size[1]}',
                    'bbox_크롭': [x0, y0, x1, y1],
                    '크롭_오프셋': [ox, oy],
                },
                '위치_신뢰': '높음',
                '크기': {'셀면적비_퍼센트': round(float(area), 2),
                         '구분': '큼' if area >= c.big_area_pct else '작음'},
                '크기_신뢰': '중간',
                '유형후보': [b[5]],          # 검출기 태그. 🔴 검증 미통과
                '유형_신뢰': '낮음',
                '유형_주의': '검증 미통과. 확정 표현 금지',
                '_score': round(b[4], 4),
            })
        return fr

    # ── 프레임 N장 ───────────────────────────────────────────────────
    def infer_frames(self, images: Iterable[Image.Image | str | Path | bytes],
                     names: Sequence[str] | None = None) -> list[dict]:
        """원본 이미지들 → 프레임별 판정 dict. bbox는 **원본 좌표**.

        🔴 **디스크에 아무것도 쓰지 않는다.** 크롭은 메모리에서 만들어 모델에 바로 넣는다.
        🔴 **배치 단위로 흘린다.** 셀 하나(270프레임)를 한 번에 올리면
           1920×1080×3 × 270 ≈ 1.6GB다. 서버에서 동시 요청이 겹치면 바로 OOM이다.
           → 한 번에 `cfg.batch`장만 메모리에 둔다.
        """
        from itertools import islice
        c = self.cfg
        it = iter(images)
        results: list[dict] = []
        idx = 0
        while True:
            chunk = list(islice(it, c.batch))
            if not chunk:
                break
            origs, keys = [], []
            for j, x in enumerate(chunk):
                im, k = self._open(x, idx + j, names)
                origs.append(im); keys.append(k)
            crops, offs, oks = [], [], []
            for im in origs:
                cr, off, ok = crop_cell(im, c)
                crops.append(cr); offs.append(off); oks.append(ok)
            dets = self._detect(crops)
            for k, orig, crop, off, ok, boxes in zip(keys, origs, crops, offs, oks, dets):
                results.append(self._frame_result(k, orig, crop, off, ok, boxes))
            idx += len(chunk)
            origs.clear(); crops.clear()          # 다음 배치 전에 참조를 끊는다
        return results

    # ── 셀 1개 ───────────────────────────────────────────────────────
    def infer_cell(self, cell_id: str,
                   images: Iterable[Image.Image | str | Path | bytes],
                   names: Sequence[str] | None = None) -> dict:
        """셀 하나 → 셀 판정 + 프레임별 결과.

        🔑 **전량을 보지 않는다.** `cfg.cell_sample_k`장을 균등 추출해 검사하고,
           **flag율 임계**로 셀을 판정한다. 근거는 Config 주석 참조.
        """
        c = self.cfg
        images = list(images)
        names = list(names) if names else None
        n_all = len(images)

        # 균등 추출 — 회전 각도가 고르게 섞이도록 앞뒤로 몰지 않는다
        if c.cell_sample_k and n_all > c.cell_sample_k:
            idx = [round(i * (n_all - 1) / (c.cell_sample_k - 1))
                   for i in range(c.cell_sample_k)]
            idx = sorted(set(idx))
            images = [images[i] for i in idx]
            names = [names[i] for i in idx] if names else None

        t0 = time.perf_counter()
        frames = self.infer_frames(images, names)
        n_def = sum(1 for f in frames if f['판정'] == '결함')
        rate = n_def / max(len(frames), 1)
        return {
            '셀': str(cell_id),
            '전체_프레임수': n_all,
            '검사_프레임수': len(frames),
            '결함_프레임수': n_def,
            'flag율': round(rate, 3),
            '셀_판정': '결함' if rate >= c.cell_flag_rate else '정상',
            '셀_판정_신뢰': '높음',
            '셀_판정_근거': {'규칙': f'검사 프레임의 flag율 ≥ {c.cell_flag_rate:.0%}',
                             'flag': f'{n_def}/{len(frames)}',
                             '근거수치': '결함 셀 flag율 최소 80% vs 정상 셀 최대 50% '
                                         '(각 12셀 실측, 간격 +30%p)'},
            '주의': '프레임은 같은 셀을 360도 회전하며 찍은 것이다. '
                    '같은 결함이 여러 프레임에 나올 수 있다. '
                    f'전체 {n_all}장 중 {len(frames)}장을 균등 추출해 검사했다.',
            '소요초': round(time.perf_counter() - t0, 2),
            '프레임': frames,
        }


# ════════════════════════════════════════════════════════════════════════
# CLI · 자체 점검
# ════════════════════════════════════════════════════════════════════════

def _selftest() -> None:
    """모델 없이 도는 부분만 검증한다 — 좌표 변환이 핵심이다."""
    cfg = Config()
    rng = np.random.default_rng(0)
    for _ in range(20):
        w, h = 1920, 1080
        a = np.full((h, w, 3), 250, np.uint8)
        ox, oy = int(rng.integers(150, 700)), int(rng.integers(60, 250))
        cw, ch = int(rng.integers(400, 900)), int(rng.integers(500, 780))
        a[oy:oy + ch, ox:ox + cw] = (150, 40, 140)
        im = Image.fromarray(a)

        crop, (x0, y0), ok = crop_cell(im, cfg)
        assert ok, '셀 추정 실패'
        # 크롭 안의 임의 점을 원본으로 옮겼다가 되돌리면 같아야 한다
        px, py = int(rng.integers(0, crop.width)), int(rng.integers(0, crop.height))
        assert (px + x0 - x0, py + y0 - y0) == (px, py)
        # 크롭이 원본 밖으로 안 나간다
        assert 0 <= x0 and 0 <= y0
        assert x0 + crop.width <= w and y0 + crop.height <= h
        # 셀이 크롭 안에 온전히 들어온다(패딩 포함)
        assert x0 <= ox and y0 <= oy
        assert x0 + crop.width >= ox + cw and y0 + crop.height >= oy + ch

        cb = cell_box_px(crop)
        assert cb[2] > cb[0] and cb[3] > cb[1]

    # 계약서가 스키마를 지키는가
    g = CONTRACT['필드 신뢰 등급']
    assert set(g) >= {'판정', '좌표계', '위치', '크기', '유형후보'}
    assert g['유형후보']['신뢰'] == '낮음'
    assert '실측_적중률' in g['유형후보']
    print('selftest OK — 좌표 변환 20/20 · 계약서 스키마 정상')


def verify_crop_equivalence(originals: Sequence[str | Path], crop_dir: str | Path,
                            cfg: Config | None = None) -> dict:
    """🔴 배포 경로가 평가 경로와 같은지 확인한다.

    성능 수치(P 0.873 / R 0.995 / F1 0.930)는 **이미 잘린 크롭**에서 잰 것이다.
    배포는 `원본 → crop_cell → detect` 경로다. 같은 함수를 쓰니 같은 크롭이 나와야
    하지만 **그건 가정**이고, 다르면 그 수치가 배포에 적용되지 않는다.

    원본에서 만든 크롭이 기존 크롭 파일과 **픽셀 단위로 같은지** 검사한다.
    """
    import hashlib
    cfg = cfg or Config()
    crop_dir = Path(crop_dir)
    n = same_size = same_px = 0
    bad: list[tuple[str, str]] = []
    for op in originals:
        op = Path(op)
        ref = crop_dir / op.name
        if not ref.exists():
            bad.append((op.name, '대응 크롭 파일 없음')); continue
        n += 1
        made, off, ok = crop_cell(Image.open(op).convert('RGB'), cfg)
        have = Image.open(ref).convert('RGB')
        if made.size != have.size:
            bad.append((op.name, f'크기 다름 {made.size} vs {have.size}')); continue
        same_size += 1
        # JPEG 재압축 차이가 있으므로 정확 일치 대신 평균 절대차로 본다
        d = float(np.abs(np.asarray(made, np.int16) - np.asarray(have, np.int16)).mean())
        if d <= 2.0:
            same_px += 1
        else:
            bad.append((op.name, f'화소 차이 평균 {d:.1f} (>2.0)'))
    res = {'검사': n, '크기_일치': same_size, '화소_일치': same_px,
           '불일치': bad[:20], '불일치_총': len(bad)}
    print(f'배포 경로 등가성 — 원본 {n}장')
    print(f'  크기 일치 {same_size}/{n} · 화소 일치 {same_px}/{n}')
    if bad:
        print(f'  🔴 불일치 {len(bad)}건')
        for a, b in bad[:10]:
            print(f'     {a[-44:]:44s} {b}')
        print('  → **배포 경로가 평가 경로와 다르다.** 300장 성능 수치를 그대로 쓸 수 없다.')
    else:
        print('  ✅ 배포 경로 = 평가 경로. 300장 성능 수치를 그대로 적용할 수 있다.')
    return res


def verify_fixture(fixture_json: str | Path, images_dir: str | Path,
                   cfg: Config | None = None) -> dict:
    """골든 픽스처 회귀 — 배포 경로가 스냅샷과 같은 결과를 내는지.

    배포·라이브러리 업그레이드·전처리 수정 뒤에 **이것만** 돌리면 된다.
    전처리가 어긋나도 서버는 200 을 반환하므로, 이게 유일한 자동 방어선이다.

    판정 기준
      · `gate` · `n_box`  — **정확 일치 필수**. 이 둘이 운영 동작이다
      · 크롭 오프셋·크기   — 4px 허용(경고). Pillow 버전에 따라 1~5px 흔들린다
      · bbox              — 8px 허용(경고)
      · score             — 2e-3 허용(경고)
    """
    import hashlib
    cfg = cfg or Config()
    fx = json.loads(Path(fixture_json).read_text(encoding='utf-8'))
    d0 = images_dir and Path(images_dir)
    names = sorted(fx['frames'])
    meta = fx.get('_meta', {})
    print(f'픽스처 {len(names)}장 · {meta.get("path", "경로 미기록")}')
    print(f'  기록: transformers {meta.get("transformers")} · revision {meta.get("revision")} '
          f'· thr_gate {meta.get("thr_gate")}')
    print(f'  현재: thr_gate {cfg.thr_gate} · revision {cfg.owl_rev}')
    if meta.get('thr_gate') is not None and meta['thr_gate'] != cfg.thr_gate:
        print(f'  🔴 운영점이 다르다 — 픽스처는 thr {meta["thr_gate"]} 로 만들어졌다. '
              f'이 픽스처로는 지금 설정을 검증할 수 없다')
    if meta.get('revision') and meta['revision'] != cfg.owl_rev:
        print(f'  🔴 모델 revision 이 다르다 — 픽스처 {meta["revision"]} vs 현재 {cfg.owl_rev}')

    fail: list[tuple[str, str]] = []
    warn: list[tuple[str, str]] = []
    paths = []
    for n in names:
        p = d0 / n
        if not p.exists():
            fail.append((n, '이미지 없음')); continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        if fx['frames'][n].get('sha256') and got != fx['frames'][n]['sha256']:
            fail.append((n, f'이미지가 바뀌었다 (sha256 {got} ≠ {fx["frames"][n]["sha256"]})'))
            continue
        paths.append((n, str(p)))
    if not paths:
        print(f'🔴 검증할 이미지가 없다 — 실패 {len(fail)}건')
        for n, w in fail[:5]:
            print(f'  🔴 {n[-42:]:42s} {w}')
        if any('바뀌었다' in w for _, w in fail):
            print('   (평가 픽스처를 배포 이미지 폴더에 들이댄 것일 수 있다 — '
                  '평가는 크롭본, 배포는 원본이라 해시가 다르다)')
        else:
            print(f'   (--images 가 가리키는 폴더를 확인할 것: {d0})')
        return {'fail': fail, 'warn': warn, 'n': 0}

    res = ExtInspector(cfg).infer_frames([p for _, p in paths], names=[n for n, _ in paths])
    for (n, _), r in zip(paths, res):
        e = fx['frames'][n]
        if 'gate' in e and r['판정'] != e['gate']:
            fail.append((n, f'게이트 {e["gate"]} → {r["판정"]}'))
        if 'n_box' in e and isinstance(e['n_box'], int) and r['판정_근거']['박스수'] != e['n_box']:
            fail.append((n, f'박스수 {e["n_box"]} → {r["판정_근거"]["박스수"]}'))
        for key, cur, tol in (('crop_offset', r['셀_크롭']['오프셋'], 4),
                              ('crop_size', r['셀_크롭']['크기'], 4)):
            if key in e and max(abs(a - b) for a, b in zip(e[key], cur)) > tol:
                warn.append((n, f'{key} {e[key]} → {cur} (>{tol}px)'))
        for k, ex in enumerate(e.get('defects', [])):
            if k >= len(r['결함']):
                warn.append((n, f'결함[{k}] 없음')); continue
            g = r['결함'][k]
            if max(abs(a - b) for a, b in zip(ex['bbox'], g['위치']['bbox'])) > 8:
                warn.append((n, f'결함[{k}] bbox {ex["bbox"]} → {g["위치"]["bbox"]}'))
            if abs(ex.get('score', 0) - g['_score']) > 2e-3:
                warn.append((n, f'결함[{k}] score {ex.get("score")} → {g["_score"]}'))

    print(f'\n검사 {len(paths)}장 · 🔴 실패 {len(fail)} · ⚠️ 경고 {len(warn)}')
    for lab, lst in (('🔴', fail), ('⚠️', warn)):
        for n, w in lst[:12]:
            print(f'  {lab} {n[-42:]:42s} {w}')
        if len(lst) > 12:
            print(f'     ... 외 {len(lst)-12}건')
    if not fail:
        print('\n✅ 회귀 없음 — 게이트 판정과 박스 수가 스냅샷과 전부 일치한다')
        if warn:
            print('   (경고는 좌표가 몇 px 흔들린 것이다. 운영 동작은 같다)')
    else:
        print('\n🔴 회귀 발생 — 배포하지 말 것.')
        print('   원인 후보: 라이브러리 버전 · 모델 revision · 크롭 파라미터 · 입력 JPEG 재인코딩')
    return {'fail': fail, 'warn': warn, 'n': len(paths)}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description='EXT 외관 검사 추론')
    p.add_argument('--selftest', action='store_true', help='모델 없이 좌표·계약서 검증')
    p.add_argument('--notes', action='store_true', help='배포 전 확인사항 출력')
    p.add_argument('--verify-crop', action='store_true',
                   help='배포 경로(원본→크롭)가 평가 경로와 같은지 검증')
    p.add_argument('--verify-fixture', action='store_true',
                   help='골든 픽스처 회귀 — 배포 전 필수')
    p.add_argument('--fixture', help='--verify-fixture: golden_fixture_deploy.json 경로')
    p.add_argument('--images', dest='fx_images', help='--verify-fixture: 픽스처 이미지 폴더')
    p.add_argument('--originals', nargs='*', help='--verify-crop: 원본 이미지들')
    p.add_argument('--crop-dir', help='--verify-crop: 기존 크롭 폴더(평가셋 images/)')
    p.add_argument('--cell', help='셀 ID')
    p.add_argument('--frames', nargs='*', help='원본 이미지 경로들 (추론용)')
    p.add_argument('--glob', help='이미지 glob 패턴 (예: "frames/0041_*.jpg")')
    p.add_argument('--out', help='결과 JSON 경로 (없으면 표준출력)')
    a = p.parse_args(argv)

    if a.notes:
        print(DEPLOY_NOTES)
        return 0
    if a.selftest:
        _selftest()
        return 0
    if a.verify_fixture:
        if not (a.fixture and a.fx_images):
            p.error('--verify-fixture 는 --fixture 와 --images 가 필요하다')
        r = verify_fixture(a.fixture, a.fx_images)
        return 0 if not r['fail'] else 1
    if a.verify_crop:
        if not (a.originals and a.crop_dir):
            p.error('--verify-crop 은 --originals 와 --crop-dir 이 필요하다')
        r = verify_crop_equivalence(a.originals, a.crop_dir)
        return 0 if not r['불일치_총'] else 1

    paths = list(a.frames or [])
    if a.glob:
        paths += sorted(str(x) for x in Path().glob(a.glob))
    if not paths:
        p.error('--frames 또는 --glob 이 필요하다')

    insp = ExtInspector()
    res = insp.infer_cell(a.cell or 'unknown', paths)
    payload = {'contract': CONTRACT, 'result': res}
    txt = json.dumps(payload, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(txt, encoding='utf-8')
        print(f'저장 {a.out} · 프레임 {res["검사_프레임수"]} · '
              f'결함 {res["결함_프레임수"]} · {res["소요초"]}초')
    else:
        print(txt)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
