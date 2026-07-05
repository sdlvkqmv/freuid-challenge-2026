# Session 5 (2026-07-05) — EDA 진단 + 다음 세션 실행 계획

> 이 문서는 2026-07-05 RESEARCH 세션의 발견/제안만 담는다. SUMMARY.md 등 다른 문서는 수정하지 않았다.
> 다음 세션은 이 문서를 읽고 §4 우선순위대로 진행할 것.

## 1. 컨텍스트 (세션 시작 시점)

- 우리 최고 기록: **attempt 06 = LB 0.15185** (effb3 SRM + recapture aug).
- LB 상위권 (2026-07-05 조회): 1위 0.00039, 4위 0.00063, 20위권도 0.0025.
  → 상위 팀은 public test를 사실상 만점 처리. 우리와 2 orders of magnitude 격차
  = 점진적 모델 개선 문제가 아니라 **구조적 문제**.
- 마감 2026-07-14. 남은 ~9일 × 5제출/일.
- 세션 시작 시 attempt 26/27 (effb3 seed123/456 reseed 앙상블), 28 (convnext_large
  srm+recap) 학습 중이었음 — docs 기존 결론상 기대값 낮음 (diverse-backbone/앙상블
  계열 이미 실패).

## 2. 이번 세션에서 배제한 가설 (제출 0회 소모, 전부 로컬 EDA)

| # | 가설 | 방법 | 결과 |
|---|---|---|---|
| 1 | 메타데이터 leak (train) | label별 파일크기/해상도/JPEG quant table 비교 (1500장×2) | **없음.** quant table 전 이미지 동일(일괄 재인코딩), 크기/해상도 분포 라벨 간 동일 |
| 2 | train↔test 문서 중복 | 8×8 dhash 전수 → 80% "일치"했으나 가짜 양성(템플릿 충돌). 1024-bit dhash + GPU Hamming NN 전수 재검 | **없음.** 최근접 쌍(d=11/1024)도 육안 확인 결과 다른 인물/데이터. 같은 템플릿이 해시를 지배할 뿐 |
| 3 | fraud-원본 편집 쌍 구조 | train 내부 / test 내부 NN 거리 분석 | **없음.** fraud 29,347 중 190개만 d<20 파트너 보유, 파트너 라벨 혼재(0=113, 1=77). test 내부 tight pair 0.05% |

부수 관찰: test는 train과 다른 quant table로 재인코딩됨. test 내부 소수(~2%) 별도
quant table 그룹 + 비표준 해상도(840×530 등 ~1.6%) 존재 — 미추적, 낮은 우선순위.

## 3. 핵심 발견 — 타입 간 점수 정렬 붕괴 (this session's main result)

06은 test에서 극단적 확신: 7,821장 중 96%가 score<0.01 또는 >0.99, 중간대 거의 없음.
dhash32 NN으로 test 각 이미지에 문서타입 부여(템플릿 지배라 사실상 정확) 후 분해:

| 타입 | test n (NN 추정) | 06 예측 fraud율 | train 실제 fraud율 |
|---|---|---|---|
| MAURITIUS/ID | 1,724 | **21%** | 40% |
| EGYPT/DL | 991 | 50% | 50% |
| GUINEA/DL | 1,652 | 53% | 40% |
| BENIN/DL | 1,724 | **70%** | 40% |
| MOZAMBIQUE/DL | 1,730 | **70%** | 40% |

test가 train처럼 타입별 40~50% 균형이라 가정하면:
- MAURITIUS: fraud 절반가량 놓침 (APCER↑)
- BENIN/MOZ: bona-fide ~30%p를 fraud로 오인 (BPCER↑)

→ **타입별 점수 스케일이 서로 어긋나 pooled DET 곡선이 붕괴.** 타입 내 랭킹이
멀쩡해도 FREUID(AuDET + APCER@1%BPCER)는 무너진다. 이것이 LB 0.15의 유력한
구조적 원인.

추가 정합 단서: train의 실물 recapture 20장(is_digital=False) 중 **13장이
MAURITIUS fraud** — test MAURITIUS fraud가 recapture 위주라서 06이 놓친다는
해석과 일치.

주의: per-type 정규화는 *per-group* 변환이라 monotonic no-op가 아님 — DET를
실제로 움직일 수 있는 유일한 후처리 계열 (SUMMARY.md 기존 노트와 일치, 미실행
상태였음).

## 4. 다음 세션 실행 순서 (우선순위)

### P1 — per-type rank-norm 제출 (파일 준비 완료, 제출 1회)

`experiments/sub_06_pertype_ranknorm.csv` 생성해둠: 06 점수를 타입 내
rank → (rank−0.5)/n 으로 균일화. public 7,821 전부 매칭, 나머지 135k 행은 06
placeholder 유지.

```bash
kaggle competitions submit the-freuid-challenge-2026-ijcai-ecai \
  -f experiments/sub_06_pertype_ranknorm.csv \
  -m "06 + per-type rank normalization (dhash32-NN type assignment)"
```

해석 가이드:
- **크게 개선** (예: 0.15 → <0.05): 타입 내 랭킹은 이미 양호. 이후 모든 제출에
  같은 후처리 적용. 다음 병목 = 타입 내 랭킹 품질 → 타입별 specialist 모델,
  MAURITIUS 집중(P3).
- **소폭 개선/무변화**: 타입 내 랭킹 자체가 나쁨 (특히 MAURITIUS). P2/P3로.
- 변형 실험 여지: 순수 rank-flatten 대신 train prior(40~50%)로 타입별 중심 이동,
  또는 global score와 blend. 먼저 순수 버전으로 방향 확인.

### P2 — 실물 recapture 외부 데이터 투입 ("new DATA" 실행)

Session 4까지의 결론: 합성 recapture aug(17/18)는 합성 도메인에 과적합, 진짜
recapture 데이터 필요. 후보 공개 데이터셋:
- **MIDV-2020 / MIDV-500 / MIDV-Holo** — 인쇄된 ID 문서의 실제 스마트폰 촬영
- **DLC-2021** — 문서 캡처 도메인
- **KID34K** — 한국 ID 카드 촬영 (진짜/인쇄재촬영/화면재촬영 구분 라벨)

사용법: 06 레시피에 recapture 도메인 실데이터 혼합 (fine-tune 마지막 epoch들,
또는 domain-adversarial/consistency 브랜치). 라이선스 확인 필수 (최종 제출물에
OSI 라이선스 코드 + 기술보고서 요구됨 — 외부 데이터 허용 여부 규정도 재확인).

### P3 — MAURITIUS 집중 (저비용)

- train recapture 13장(MAURITIUS fraud) 대폭 oversample 후 06 fine-tune.
- 06이 낮은 점수 준 test MAURITIUS 이미지 육안 검수 — 뭘 놓치는지 직접 확인.
  (타입 부여 결과: `experiments/session5_artifacts/test_type_score.csv`,
  컬럼 id/type/nn_dist/score)

### P4 — 진행 중이던 26/27/28

reseed 앙상블(26/27)만 소폭 기대(~0.005), 28은 기존 결론상 낮음. P1~P3에 자원
우선. 결과 나오면 SUMMARY.md 스코어보드에만 기록.

## 5. 아티팩트 (재현용)

`experiments/session5_artifacts/` 에 보존:
- `test_type_score.csv` — test 7,821장의 NN 기반 타입 + 06 score
- `dhash32.npz` — train/test 전체 1024-bit dhash (packbits) + 파일명
- `nn_result.npz` — test→train 최근접 Hamming 거리/인덱스
- `internal_nn.npz` — train/test 내부 최근접 구조
- `nn_hash.py`, `pair_struct.py` — 위 산출물 생성 스크립트

제출 파일: `experiments/sub_06_pertype_ranknorm.csv` (P1).

방법 요약: dhash32 = grayscale 33×32 resize 후 인접 픽셀 비교 1024-bit,
NN 탐색은 ±1 벡터 GPU matmul (hamming = (1024−dot)/2).
