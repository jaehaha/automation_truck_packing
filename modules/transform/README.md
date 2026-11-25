# transform: 통합 좌표계 변환

모든 측정·명령·좌표를 **Container Frame (CF)** — 바깥 프레임 왼쪽-아래 모서리 원점 — 로 통일하는 유틸.

| 파일 | 내용 |
|---|---|
| `transform.py` | 좌표계 변환 함수 + 사용자 보정 상수 |

## 다루는 좌표계

| 약자 | 뜻 |
|---|---|
| CF | Container Frame — 모든 좌표의 단일 기준 |
| MW | MuJoCo world |
| RC | Robot Coords (팔 베이스 기준) |
| RM | Rail mm (Arduino 0~1000) |
| CC | Camera coords (OpenCV) |

## 변환 함수

- `mw_to_cf`, `cf_to_mw`
- `rc_to_cf`, `cf_to_rc`
- `rail_mm_to_cf_x`, `cf_x_to_rail_mm`
- `cam_to_cf(xyz, cam_extr_in_cf)`
- `eef_to_cf(xyz, eef_pose_in_cf)`

## 쓰는 쪽

| 쪽 | 쓰는 함수 |
|---|---|
| [`manager/main.py`](../../manager/main.py) | `mw_to_cf()` — Packing goal 을 CF 로 기록 |
| [`modules/camera/camera.py`](../camera/camera.py) | `cam_to_cf()`, `OVERVIEW_CAM_IN_CF` — ArUco 검출 결과 CF 로 |
| [`modules/rail/rail.py`](../rail/rail.py) | (추후) `rail_mm_to_cf_x()` — 아두이노 레일 mm → CF x |
| [`modules/arm/arm.py`](../arm/arm.py) | (추후) `eef_to_cf()` — Piper EEF → CF |

## 사용자가 채워야 할 상수

`transform.py` 상단의 `===== 사용자 보정 필요 =====` 블록. 자세한 내용은 [`pipeline/pipeline.drawio`](../../pipeline/pipeline.drawio) 의 "편집 지점" 섹션 참고.
