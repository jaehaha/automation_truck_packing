# 적재 파이프라인

Packing Model 이 정한 순서대로 상자를 하나씩 집어 적재하고, 끝날 때마다 상자 표를 갱신한다.
도식: [`pipeline.drawio`](pipeline.drawio).

## 상태 표기

- 🟢 **완료** — sim 모드에서 지금 바로 돌아감
- 🟡 **부분** — sim 은 OK, real 은 추가 작업 필요 (또는 더미값)
- 🔴 **미구현** — 뼈대만 있음

## 사전 (한 번만)

| 단계 | 하는 일 | 호출 지점 | 상태 |
|---|---|---|---|
| 0 | 상자 시퀀스 입력 | `manager/main.py` `load_manifest()` | 🟢 |
| 1 | Packing Model — 각 상자의 goal 결정 | `modules/packing/plan_moves.py` (사전 생성된 `moves_big.json` 로드) | 🟢 |
| — | BoxTable 초기화 (`id → {size, current, goal, state}`) | `manager/boxes.py` | 🟢 |

## LOOP (상자 하나씩, `main.py` for)

| 단계 | 하는 일 | 호출 지점 | 하드웨어 | 상태 |
|---|---|---|---|---|
| 2 | 사람이 투입 (수동) 위치 (85, 240), ArUco 마커 위 | 코드 X — `PICK_XY` 가 참조 좌표 | 사람 | 🟡 |
| 3-a | 손목 카메라로 오차 측정 → BoxTable.current 보정 | `robot.measure_box()` (현재 더미) | 손목 카메라 | 🔴 |
| 3-b | Motion Planning — `Plan(approach, carry, retreat)` | `Sequencer.plan()` (`modules/mp/sequence.py`) | — | 🟢 |
| 4 | 접근 — 투입 위치 위로 (빈손) | `robot.follow(plan.approach)` | 레일 🟢 + 팔 🔴 | 🟡 |
| 5 | 흡착 ON — 상자 집기 | `robot.suction_on()` → Arduino `ON` | 솔레노이드 | 🟢 |
| 6 | 반송 (carry) — MV_POS 거쳐 적재 자리로 | `robot.follow(plan.carry)` | 레일 🟢 + 팔 🔴 | 🟡 |
| 7 | 흡착 OFF — 상자 분리 | `robot.suction_off()` → Arduino `OFF` | 솔레노이드 | 🟢 |
| 8 | 후퇴 + 레일 0 복귀 | `robot.follow(plan.retreat)` | 레일 🟢 + 팔 🔴 | 🟡 |
| 9 | BoxTable 갱신 → 다음 상자 | `table.done(id)` | — | 🟢 |

4·6·8 은 레일 쪽은 끝났고 **`RealArm` 만 채우면 전부 초록** 으로 바뀜.

## 장치 레이어 (`modules/<device>/`)

| 장치 | Sim | Real | 파일 |
|---|---|---|---|
| Arm | 🟢 MuJoCo ctrl[1:7] | 🔴 Piper CAN | `arm/arm.py` |
| Rail | 🟢 MuJoCo ctrl[0] | 🟢 Arduino M/P/H | `rail/rail.py` |
| Suction | 🟢 상태 추적 | 🟢 Arduino ON/OFF | `vacuum/suction.py` |
| Camera | 🟢 MuJoCo 렌더 | 🔴 OpenCV + ArUco | `camera/camera.py` |

- **공유 Arduino**: `manager/arduino.py` — 레일·밸브가 한 보드에 물려 있음. 프로토콜: M/P/H, ON/OFF, POS, STOP (115200 ASCII)
- **Robot**: `manager/robot.py` — 네 장치 묶어 `.sim / .real / .dual` 팩토리와 `follow / suction_on/off / state / stop` 노출

## 모드별 실행

```bash
cd manager
mjpython main.py --mode sim                     # 시뮬 + 뷰어
python  main.py --mode sim --no-view            # 시뮬 계획만
mjpython main.py --mode dual --port /dev/tty.usbmodemXXXX   # 실물+시뮬 (디지털 트윈)
python  main.py --mode real --port /dev/tty.usbmodemXXXX    # 실물만
```

| 모드 | 명령 전송 | read | 뷰어 |
|---|---|---|---|
| sim | Sim* → MuJoCo | MuJoCo | mjpython |
| real | Real* → CAN + Arduino | 하드웨어 | 없음 |
| dual | Dual* = 실물 + 시뮬 | **실물** | mjpython (시뮬=거울) |

## 참고

**타이밍 A (dual)**: 같은 명령을 양쪽에 동시 전송. 시뮬은 즉시 반영, 실물은 모터 속도로 따라옴. 실물이 예외를 던지면 시뮬도 멈춤. `rail.command(m)` 중 MOVE 가 Arduino 쪽에서 `ERROR busy` 로 거부되면 Python 이 조용히 무시하고 다음 호출에 최신 목표로 추종.

**좌표계 통일 (후속)**: 지금은 Sequencer 가 MuJoCo 좌표, `RealRail(zero_offset_m=-0.49)` 로 Arduino 0~1000 mm 와 연결. 추후 "컨테이너 좌표계" 로 모든 상수(`ORIGIN`, `PICK_XY`, `KEEPOUT_X`) 정비 예정.

**기준 결과**: `packing/moves_big.json` (80개) — sim 모드 **성공 57 / 실패 23** (투입 구역 20 + 실제 실패 3).
