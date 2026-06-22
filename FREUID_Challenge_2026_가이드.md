# The FREUID Challenge 2026 — IJCAI-ECAI 참가 가이드

> 1st Competition on Identity Documents Fraud Detection
> 신분증(ID 문서) 위변조 탐지 대회 — Microblink Fraud Lab 주최

- **Kaggle**: https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai
- **공식 사이트**: https://freuid2026.microblink.com/
- **IJCAI-ECAI 2026**: https://2026.ijcai.org/ (독일 Bremen)
- **문의**: freuid-challenge-2026@microblink.com / Discord: https://discord.gg/8hKbNEKnT
- **문서 작성일**: 2026-06-22

---

## 1. 대회 개요

차세대 신분증 위변조를 탐지하는 대회. 기존 벤치마크가 놓치는 현실적 공격 표면 3가지 커버:

1. **Physical manipulations** — 실제 문서 원판 위 물리적 변조
2. **GenAI-driven multimodal edits** — 생성형 AI 기반 디지털 편집 (일반 사기범도 접근 가능)
3. **Print-and-capture forgeries** — 인쇄 후 재촬영으로 디지털 노이즈 흔적 제거 (기존 탐지기 회피)

핵심 난이도: 디지털 위조물을 **인쇄→재촬영**해 디지털 흔적 지움. 물리적 위조는 인쇄된 문서 위에 수행. → 도메인 경계 흐림.

| 항목 | 값 |
|---|---|
| 카테고리 | Community |
| 상금 | $6,000 USD (Microblink 후원) |
| 마감 (deadline) | **2026-07-14 11:59 (UTC)** |
| 팀 병합 마감 | 2026-07-14 11:59 |
| 시작일 | 2026-05-15 |
| 현재 참가팀 | 105팀 (예상 ~20팀) |
| 평가지표 | FREUID score (= AuDET 기반, **낮을수록 좋음**) |

---

## 2. 중요 규정 (반드시 확인)

### 팀 / 제출 제한
- **최대 팀원 5명**. 1인 1팀만 가능 (중복 팀 불가).
- **하루 제출 5회** (max daily submissions = 5).
- 학계·산업계 전 세계 팀 참가 가능.
- **Microblink 직원 및 직속 연구그룹은 상금 대상 제외**.

### 데이터 / 코드 의무사항
- 코드 제출 필수: **소스코드 + 설정파일 + 학습/추론 스크립트 + 재현 지침**.
- 코드는 **OSI 승인 오픈소스 라이선스**로 공개해야 함.
- **외부 데이터/사전학습 모델 허용** — 단, 적절한 라이선스 + 출처 인용 필수.
- **비공개/유료 데이터 사용 금지** (proprietary/non-freely-accessible 금지).
- 데이터는 **비상업적 연구 목적**으로만 사용 (대회 데이터셋도 동일).

### 최종 평가 제출물 (mandatory deliverables)
1. Kaggle 테스트셋 예측 파일
2. **Technical report** (기술 보고서) 동반 제출
3. 본인 인프라 사용 (GPU 쿼터 없음)

---

## 3. 평가 (Evaluation)

- **주 지표: AuDET** — Area under Detection Error Trade-off curve (DET 곡선 아래 면적). **낮을수록 우수**.
- 운영점(operating point): **BPCER 1% 고정 시점의 APCER**
  - APCER = Attack Presentation Classification Error Rate (위조를 진짜로 오판)
  - BPCER = Bona-Fide Presentation Classification Error Rate (진짜를 위조로 오판)
- **Public 검증 리더보드** + **Private 비공개 테스트셋**으로 최종 순위.
- 참고: 현재 리더보드 상위 점수 ~0.0006 수준 (낮을수록 상위).

---

## 4. 데이터셋 & 제출 형식

### 데이터셋 (FREUID)
- Microblink Fraud Lab 제공 독점 컬렉션. 진본(bona-fide) + 위조(fraudulent) 문서.
- **7개 문서 타입** (아시아·아프리카권), 다국어 스크립트 (라틴, 아랍).
- 합성 데이터 + 인쇄/촬영된 플라스틱 카드 포함.
- **전체 데이터셋 공개: 6월 초** (현재 일부 샘플 제공).
- Kaggle 파일 구조: `public_test/public_test/<hash>.jpeg` (테스트 이미지 다수) 등.

### 제출 형식 (`sample_submission.csv`)
```csv
id,label
0009cd06cf8c426789aa58b2b05c60d2,0
000c9f83f98c4babac3f030df5300a8e,0
...
```
- `id` = 이미지 파일 해시 (확장자 제외)
- `label` = 위조 점수/확률 (DET 곡선 계산용 연속값 권장; 1=fraud, 0=bona-fide 방향)

---

## 5. 일정 (Timeline)

| 이벤트 | 일자 |
|---|---|
| 대회 시작 | 2026-05-15 |
| 전체 데이터셋 공개 | 2026년 6월 초 |
| **Kaggle 제출 마감** | **2026-07-14 11:59 UTC** |
| 워크샵/튜토리얼 | 2026-08-15 ~ 08-17 |
| Live Showdown (본 컨퍼런스) | 2026-08-18 ~ 08-21 |
| 장소 | Bremen Exhibition Center, 독일 |

---

## 6. 상금

| 순위 | 상금 |
|---|---|
| 1위 | $3,000 |
| 2위 | $2,000 |
| 3위 | $1,000 |

- 상위 3팀: Live Showdown(8/18~21) 참석 **강력 권장**, 컨퍼런스 입장권 제공(확정 조건부).

---

## 7. 참가 절차

1. Kaggle 대회 페이지 등록 (이미 참가 완료 — `userHasEntered=True`).
2. 학습 데이터 샘플 확보 → 6월 전체 데이터셋 다운로드.
3. 모델 학습 → 예측 파일 생성 → Kaggle 제출 (하루 5회).
4. Public 리더보드 모니터링.
5. 최종: **코드(OSI 라이선스) + Technical report** 제출.

---

## 8. CLI 빠른 참조 (conda env: `fraud`)

```bash
conda activate fraud

# 대회 파일 목록
kaggle competitions files the-freuid-challenge-2026-ijcai-ecai

# 전체 데이터 다운로드
kaggle competitions download the-freuid-challenge-2026-ijcai-ecai

# 단일 파일 (예: 제출 샘플)
kaggle competitions download the-freuid-challenge-2026-ijcai-ecai -f sample_submission.csv

# 제출
kaggle competitions submit the-freuid-challenge-2026-ijcai-ecai -f submission.csv -m "메시지"

# 내 제출 이력 / 리더보드
kaggle competitions submissions the-freuid-challenge-2026-ijcai-ecai
kaggle competitions leaderboard the-freuid-challenge-2026-ijcai-ecai -s
```

---

## 9. 주최진

- **산업**: Microblink (데이터·상금·운영 주도) — Ivan Relić, Vincenzo D'Elia, Stefano Bortoli, Radu Tudoran 외
- **학계**: UniZG FER 자그레브대 (Marin Oršić), Politecnico di Torino (Lorenzo Vaiani, Paolo Garza)

---

## 10. 핵심 체크리스트

- [ ] 팀원 ≤5명, 1인 1팀 확인
- [ ] 하루 제출 5회 한도 관리
- [ ] 외부 데이터/모델 사용 시 라이선스+인용 기록
- [ ] 코드 OSI 오픈소스 라이선스 준비
- [ ] Technical report 작성
- [ ] 제출 형식 `id,label` 준수
- [ ] 마감 2026-07-14 11:59 UTC 엄수
- [ ] AuDET 낮추기 = BPCER 1%에서 APCER 최소화 목표
