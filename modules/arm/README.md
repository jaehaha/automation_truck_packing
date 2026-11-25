# arm: 팔 (Piper 6자유도)

`Arm` 인터페이스: `read() -> q`, `command(q)`.

| 파일 | 내용 |
|---|---|
| `arm.py` | `Arm` (추상), `SimArm` (MuJoCo `ctrl[1:7]`), `RealArm` (Piper SDK, 미구현) |
| `piper_keyboard.py` | 키보드로 Piper 를 직접 움직이는 테스트 스크립트. `RealArm` 구현 참고용 |

쓰는 쪽: [`manager/robot.py`](../../manager/robot.py). 설계: [`../../pipeline/details_by_modules/unified.md`](../../pipeline/details_by_modules/unified.md).
