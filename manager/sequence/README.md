# manager/sequence

**적재 시퀀스 파일** — Packing Model 의 입력 (`input.*`) 과 추론 결과 (`moves.*`).

## 표준 파일

| 파일 | 역할 | 생성 주체 |
|---|---|---|
| `input.json` | **입력 시퀀스** — 이번에 적재할 상자 목록 (id, 지점, 종류) | 사용자 (또는 상위 스케줄러) |
| `moves.json` | **추론 결과** — 각 상자의 적재 순서 + goal 위치 | GUI 1단계 "적재 알고리즘 추론" 또는 수동 `plan_moves.py` |

GUI 흐름:
```
input.json → [적재 알고리즘 추론 창] → moves.json → [런처 창] → main.py → 로봇 실행
```

## 포맷 — `input.json`

```json
{
  "boxes": [
    { "id": "P001", "stop": 5, "type": 3 },
    { "id": "P002", "stop": 1, "type": 5 },
    ...
  ]
}
```
- `id` — 상자 식별자 (문자열)
- `stop` — 하차 지점 번호 (1 이 가장 먼저 내리는 곳)
- `type` — 상자 종류 (`modules/arm/stacking.py:BOXES` / `modules/packing_algorithm/pack_env.BIG_TYPES` 참조)

## 포맷 — `moves.json`

```json
{
  "moves": [
    {
      "kind": "place",
      "box_id": "P001",
      "to": {
        "size_xyz_m": [0.10, 0.16, 0.16],
        "center_m":   [0.488, -0.158, -0.579]
      }
    },
    ...
  ]
}
```
`center_m` 은 Robot Coords (RC). `main.py:load_manifest` 가 MW / CF 로 변환해 `BoxTable` 에 기록.

## 추론 (CLI 직접 돌릴 때)

```bash
cd modules/packing_algorithm
python plan_moves.py --mode big \
    --manifest ../../manager/sequence/input.json \
    --out ../../manager/sequence/moves.json \
    --plans 16 --workers 16
# --motion 추가하면 Piper reach 검증 포함 (느리지만 안전)
```

## 예시 파일

- `moves_big.json`, `moves_stable.json` — 레거시 예시 (80 개 big 배치, stable 배치)
- `manifest_big.json`, `manifest_stable.json` — 그에 대응하는 입력 예시

**현재 `input.json` = `manifest_big.json` 복사본**. 실제 사용 시 수정하거나 교체.
