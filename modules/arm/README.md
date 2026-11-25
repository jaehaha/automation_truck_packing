# arm: 팔 (Piper 6자유도) + 모션 플래닝

팔 장치 wrapper 와 **모션 플래닝 로직** (옛 `modules/mp/` 가 통합됨) 을 함께 보관.

## 파일

| 파일 | 내용 |
|---|---|
| `arm.py` | `Arm` (추상), `SimArm` (MuJoCo `ctrl[1:7]`), `RealArm` (Piper SDK, ❌), `DualArm` |
| `piper_keyboard.py` | 키보드로 Piper 를 직접 움직이는 테스트. `RealArm` 구현 참고 |
| `sequence.py` | 4단계 플래너 `Sequencer`. 빈 장면에서 틀을 계산해 저장하고 실행 전에 지금 장면에서 검사 |
| `stacking.py` | `Stacker`. IK + 충돌 검사 + MuJoCo 씬 관리 |
| `rules.py` | **편집하는 곳.** `KEEPOUT_X`, `PLACE_CANDIDATES`, `EXTRA_BOXES` |
| `model/manibox_modified.xml` | MuJoCo 모델 (프레임 + 레일 + Piper 팔 + 공구 + 카메라) |
| `assets/` | 메시 (프레임, Piper, 공구) |

## 쓰는 쪽

- [`manager/robot.py`](../../manager/robot.py) — `SimArm/RealArm/DualArm` 사용
- [`manager/main.py`](../../manager/main.py) — `Sequencer` 호출
- [`modules/rail/rail.py`](../rail/rail.py), [`modules/vacuum/suction.py`](../vacuum/suction.py), [`modules/camera/camera.py`](../camera/camera.py) — 여기 `Stacker` 가 로드한 MuJoCo `model/data` 를 공유
