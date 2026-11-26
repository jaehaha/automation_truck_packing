# 적재 파이프라인

Packing Model 이 정한 순서대로 상자를 하나씩 집어 적재하고, 끝날 때마다 상자 표를 갱신한다.
도식: [`pipeline.drawio`](pipeline.drawio).

## 사전 (한 번만)

| 단계 | 하는 일 |
|---|---|
| 0 | 상자 시퀀스 입력 |
| 1 | Packing Model → 각 상자의 goal 결정 (순서대로) |
| — | BoxTable 초기화 (`id → {size, current, goal, state}`) |

## LOOP (상자 하나씩) - [파일 호출 흐름](./file_call_flow.md)

**manager** 는 사이클 하나를 호출만 한다. 각 단계는 세 가지 중 하나다:
- **arm 동작** — [`modules/arm/skills.py`](../modules/arm/skills.py) 의 함수
- **흡착** — [`modules/suction/suction.py`](../modules/suction/suction.py) 의 `Suction.grip()/un_grip()` 메서드
- **레일** — [`modules/rail/rail.py`](../modules/rail/rail.py) 의 `Rail.mv()` 메서드

`//demo` = 사용자가 궤적/자세를 녹화해서 저장할 함수. 괄호 안 인자는 상황별 분기.

### Packing  — 호출: [`actions.packing(robot, box, table)`](../manager/actions.py#L19)

0. 시작: 파이퍼 = mv_pos, rail(0) 이 규칙 (초기 자세)
1. 사람이 정해진 위치에 대충 중심 맞춰서 상자 투입 (수동)
2. `skills.approach_to_buffer()` //demo — 박스 위로 이동 ([skills.py](../modules/arm/skills.py))
3. `skills.alignment(pack)` — wrist 캠 ArUco 로 호버→접촉까지 IK ([skills.py](../modules/arm/skills.py))
4. `suction.grip(pack)` — 흡착 ON ([suction.py](../modules/suction/suction.py))
5. `skills.mv_pos()` //demo — 레일 이동 자세 ([skills.py](../modules/arm/skills.py))
6. `rail.mv(x)` — 상자 CF x 를 레일 위치로 변환해 이동 ([rail.py](../modules/rail/rail.py))
7. `skills.approach_to_box(pack)` — yz 로 자세 맞춘 뒤 x 방향으로 놓기 ([skills.py](../modules/arm/skills.py))
8. `suction.un_grip()` — 흡착 OFF ([suction.py](../modules/suction/suction.py))
9. `skills.mv_pos()` //demo — 레일 이동 자세 ([skills.py](../modules/arm/skills.py))
10. `rail.mv(0)` — 홈 복귀 ([rail.py](../modules/rail/rail.py))
11. `table.done(id)` — BoxTable 갱신: current ← goal, state = placed ([boxes.py](../manager/boxes.py))

다시 2번으로 이동.

### UnPacking  — 호출: [`actions.unpacking(robot, box, table)`](../manager/actions.py#L47)

0. 시작: 파이퍼 = mv_pos, rail(0) 이 규칙
1. `rail.mv(x)` — BoxTable 의 current x 로 레일 이동 ([rail.py](../modules/rail/rail.py))
2. `skills.approach_to_box(unpack)` — yz 로 자세 맞춘 뒤 정해진 x 로 이동 ([skills.py](../modules/arm/skills.py))
3. `skills.alignment(unpack)` — wrist 캠으로 상자 입구 면에 호버→접촉까지 IK ([skills.py](../modules/arm/skills.py))
4. `suction.grip(unpack)` — 흡착 ON ([suction.py](../modules/suction/suction.py))
5. `skills.mv_pos()` //demo — 레일 이동 자세 ([skills.py](../modules/arm/skills.py))
6. `rail.mv(0)` — 홈 복귀 ([rail.py](../modules/rail/rail.py))
7. `skills.approach_to_buffer()` //demo — 투입 공간 위로 이동 ([skills.py](../modules/arm/skills.py))
8. `skills.put_to_buffer()` — BoxTable 의 상자 size 참고해 z offset 맞춤 ([skills.py](../modules/arm/skills.py))
9. `suction.un_grip()` — 흡착 OFF ([suction.py](../modules/suction/suction.py))
10. `skills.mv_pos()` //demo — 레일 이동 자세 ([skills.py](../modules/arm/skills.py))
11. `table.remove(id)` — BoxTable 에서 제거 ([boxes.py](../manager/boxes.py))

다시 1번 반복 (버퍼에 둔 상자는 사람이 바로 뺀다고 가정).

## 사용자가 완성해야 할 것 — 단계별

### A. 데모 녹화 ([manager/demos/](../manager/demos/))
`//demo` 가 붙은 함수들은 사용자 녹화 궤적/자세를 쓴다. 각각 `.npz` 파일로 저장. 포맷: `np.savez(path, waypoints=np.ndarray[N, 7])` — 각 행이 `[rail, j1..j6]`.

- [ ] `manager/demos/mv_pos.npz` — 레일 이동 자세 (1자세 또는 짧은 궤적)
- [ ] `manager/demos/approach_to_buffer.npz` — 아무 데서나 투입 공간 위까지의 궤적

### B. offset 상수 ([modules/arm/skills.py](../modules/arm/skills.py))
alignment / approach / put 의 미세 오프셋. 실측으로 조정.

- [ ] `ALIGN_OFFSET_PACK_Z`
- [ ] `ALIGN_OFFSET_UNPACK_X`
- [ ] `GRIP_APPROACH_PACK_Z`
- [ ] `GRIP_APPROACH_UNPACK_X`
- [ ] `APPROACH_X_OFFSET`
- [ ] `PUT_BUFFER_Z_OFFSET`

### C. IK·이동 구현 (현재 NotImplementedError)
RealArm 이 들어오면 `Stacker.ik()` 를 호출해서 CF 목표 → 관절 각으로 풀고 `robot.arm.command()` 로 보내야 함.

- [ ] `skills.alignment()` — 호버 + 접촉 2단계 IK 호출
- [ ] `skills.approach_to_box()` — yz→x 순차 이동
- [ ] `skills.put_to_buffer()` — z 이동 (상자 size 참조)

### D. 좌표계 상수 ([modules/transform/transform.py](../modules/transform/transform.py))
`rail.mv(x)` 가 CF x → 레일 명령값을 올바로 뽑으려면 먼저 이 보정값들이 필요.

- [ ] `RAIL_ZERO_CF_X`, `RAIL_DIR` — 레일 0 mm 의 CF 위치
- [ ] `CF_ORIGIN_IN_MW` — sim 쪽 CF↔MW 평행이동 (SimRail._cf_x_to_rail_m)
- [ ] `WRIST_CAM_IN_EEF`, `ARUCO_BOX_SIZE_M` — alignment 에 필수
- [ ] 자세한 체크리스트: [`pipeline.drawio`](pipeline.drawio) 의 "편집 지점" 섹션

### E. BoxTable ([manager/boxes.py](../manager/boxes.py))
- [x] `BoxTable.remove(id)` — 추가 완료 (UnPacking 11단계)
- [ ] 각 `BoxEntry` 에 `goal_cf`, `current` 를 CF 좌표로 유지 (이미 `main.py:load_manifest` 가 `goal_cf` 기록)

## 장치 추상화

| 장치 | 역할 | 저수준 API | 고수준 API (actions 가 호출) |
|---|---|---|---|
| Arm | 팔 (Piper 6자유도) 관절 제어 | `read()`, `command(q)` | (skills.py 의 함수들) |
| Rail | x 방향 레일 1축 이동 | `read()`, `command(m)`, `home()` | `mv(x_cf)` |
| Suction | 흡착 밸브 ON/OFF | `on()`, `off()`, `holding()` | `grip(mode)`, `un_grip()` |
| Camera | 손목·전체뷰 ArUco 기반 측정 | `frame()`, `markers()` | — |

네 장치를 `Robot` 추상이 묶어 `follow(waypoints)`, `suction_on/off()`, `measure_box()`, `state()`, `stop()` 로 노출.
sim(MuJoCo) / real(하드웨어) / dual(둘 동시) 세 백엔드가 같은 API 뒤에 있음.

## 좌표계

모든 측정·명령은 **Container Frame (CF)** — 바깥 프레임 왼쪽-아래 모서리 원점 — 로 통일.
변환은 [`modules/transform/transform.py`](../modules/transform/transform.py) 가 담당 (CF ↔ MW/RC/RailMM/Cam/EEF).

## 참고

- 구현 상태·파일 매핑·사용자 보정 체크리스트는 [`pipeline.drawio`](pipeline.drawio)
- 장치별 상세: [`details_by_modules/`](details_by_modules/)
- 프레임 치수·좌표계 그림: [`settings/`](settings/)
