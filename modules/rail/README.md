# rail: 레일 (x 방향 선형 액추에이터)

`Rail` 인터페이스: `read() -> m`, `command(m)`, `home()`.

| 파일 | 내용 |
|---|---|
| `rail.py` | `Rail` (추상), `SimRail` (MuJoCo `ctrl[0]`), `RealRail` (아두이노 시리얼, 미구현) |

쓰는 쪽: [`manager/robot.py`](../../manager/robot.py). 설계: [`../../pipeline/details_by_modules/linear.md`](../../pipeline/details_by_modules/linear.md).
