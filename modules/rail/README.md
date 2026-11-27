# rail: 레일 (x 방향 선형 액추에이터)

`Rail` 인터페이스: `read() -> m`, `command(m)`, `home()`, `mv(x_cf)`.

| 파일 | 내용 |
|---|---|
| `rail.py` | `Rail` (추상), `SimRail` (MuJoCo `ctrl[0]`), `RealRail` (아두이노 시리얼, `modules/rail_suction.ino`), `DualRail` (실물 + 시뮬 동시) |

## Python API

| 메서드 | 하는 일 | 좌표 |
|---|---|---|
| `read()` | 현재 레일 위치 (실물: `POS`) | RC m, 0 ~ 1.0 |
| `command(m)` | 목표 위치로 이동 (실물: `MOVE <mm>`) | RC m, 0 ~ 1.0 |
| `home()` | 0 으로 복귀 (실물: `HOME`) | — |
| `mv(x_cf)` | 상자 CF x 에 맞춰 이동 — 팔 베이스가 `x_cf - APPROACH_X_OFFSET(0.015)` 에 서도록. `x_cf == 0` 이면 `home()` | CF m |

- 범위 밖이면 `ValueError`.
- `RealRail(arduino)` 는 생성 시 `INITIALIZE` 를 보낸다 (`auto_init=True`) → 그 순간 위치가 0 mm.
- `RealRail` 은 이동 중 `ERROR: Motor busy` 를 조용히 무시한다 (타이밍 A).

## 아두이노 시리얼 명령 (`rail_suction.ino`, 115200 baud, 대소문자 무관)

| 명령 | 설명 | 조건 |
|---|---|---|
| `ON` / `OFF` | 흡착 릴레이 ON/OFF | 언제나 |
| `STOP` / `S` | 감속 정지 | 언제나 |
| `POS` / `STATUS` / `HELP` | 위치 / 상태 / 도움말 출력 | 언제나 |
| `INITIALIZE` / `INIT` | 현재 위치를 0 mm 로 설정 | 정지 중 |
| `CAL <mm>` | 2000 펄스당 거리 (기본 150 mm). 이후 다시 INIT/MEASURE 필요 | 정지 중 |
| `MEASURE <mm>` | 실측 위치로 현재 위치 보정 | 정지 중 |
| `RATE <steps/s>` | 펄스 속도 (10 ~ 5400) | 정지 중 |
| `SPEED <mm/s>` | 속도 (최대 400, 기본 400 mm/s) | 정지 중 |
| `ACCEL <mm/s²>` | 가속도 (최대 1500, 기본 1500 mm/s²) | 정지 중 |
| `MOVE <mm>` | 절대 위치 이동 (0 ~ 1000, 0 = 모터 반대쪽 끝, 1000 = 모터쪽) | 초기화 후 |
| `REL <mm>` | 상대 이동 | 초기화 후 |
| `HOME` / `H` | 0 mm 복귀 | 초기화 후 |
| `START` | 랜덤 이동 테스트 100회 후 홈 복귀 | 초기화 후 |

이동 중 다른 명령은 `ERROR: Motor busy`. 이동이 끝나면 `DONE` + 위치 출력.

GUI (`gui.py` → 개발자 모드 → Rail) 에서 INIT / MOVE / H / SPEED / ACCEL 을 직접 보낼 수 있다.

쓰는 쪽: [`manager/robot.py`](../../manager/robot.py). 설계: [`../../pipeline/details_by_modules/linear.md`](../../pipeline/details_by_modules/linear.md).
