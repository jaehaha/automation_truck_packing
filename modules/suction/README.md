# suction: 흡착 공구 (벨로즈 흡착컵 + 밸브)

`Suction` 인터페이스: `on()`, `off()`, `holding()`.

| 파일 | 내용 |
|---|---|
| `suction.py` | `Suction` (추상), `SimSuction` (상태만 추적), `RealSuction` (아두이노 시리얼 `ON` / `OFF`, `modules/rail_suction.ino`), `DualSuction` (실물 + 시뮬) |

쓰는 쪽: [`manager/robot.py`](../../manager/robot.py). 설계: [`../../pipeline/details_by_modules/suction.md`](../../pipeline/details_by_modules/suction.md).
