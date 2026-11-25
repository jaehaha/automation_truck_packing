# 적재 파이프라인

Packing Model 이 정한 순서대로 상자를 하나씩 집어 적재하고, 끝날 때마다 상자 표를 갱신한다.
도식: [`pipeline.drawio`](pipeline.drawio).

## 사전 (한 번만)

| 단계 | 하는 일 |
|---|---|
| 0 | 상자 시퀀스 입력 |
| 1 | Packing Model → 각 상자의 goal 결정 (순서대로) |
| — | BoxTable 초기화 (`id → {size, current, goal, state}`) |

## LOOP (상자 하나씩)

각 줄 오른쪽 괄호는 해당 함수의 구현 위치. `//demo` = 사용자가 궤적을 녹화해서 저장할 함수. 괄호 안 인자는 상황별 분기.

### Packing  — 전체 사이클: [`pack_one()`](../manager/skills.py#L149)

0. 시작: 파이퍼 = mv_pos, rail(0) 이 규칙 (초기 자세)
1. 사람이 정해진 위치에 대충 중심 맞춰서 상자 투입 (수동)
2. `approach_to_buffer()` //demo 호출 — 박스 위로 이동 ([manager/skills.py:66](../manager/skills.py#L66))
3. `alignment(pack)` — wrist 캠 ArUco 로 ALIGN 위치로 IK 미세 조정 ([manager/skills.py:73](../manager/skills.py#L73))
4. `grip(pack)` — 정한 z offset 만큼 내려가서 흡착 ON ([manager/skills.py:96](../manager/skills.py#L96))
5. `mv_pos()` //demo 호출 — 레일 이동 자세 ([manager/skills.py:60](../manager/skills.py#L60))
6. `rail(x)` — 상자 x 를 규칙으로 변환해 아두이노 레일 이동 ([manager/skills.py:115](../manager/skills.py#L115))
7. `approach_to_box(pack)` — yz 로 자세 맞춘 뒤 x 방향으로 놓기 ([manager/skills.py:129](../manager/skills.py#L129))
8. `un_grip()` — 그리퍼 끄기 ([manager/skills.py:110](../manager/skills.py#L110))
9. `mv_pos()` //demo 호출 — 레일 이동 자세 ([manager/skills.py:60](../manager/skills.py#L60))
10. `rail(0)` — 다음 상자 적재 준비 — `robot.rail.command(0.0)`

다시 2번으로 이동.

### UnPacking  — 전체 사이클: [`unpack_one()`](../manager/skills.py#L163)

0. 시작: 파이퍼 = mv_pos, rail(0) 이 규칙
1. `rail(x)` — BoxTable 에 저장된 x 로 레일 이동 ([manager/skills.py:115](../manager/skills.py#L115))
2. `approach_to_box(unpack)` — yz 로 자세 맞춘 뒤 정해진 x 로 이동 ([manager/skills.py:129](../manager/skills.py#L129))
3. `alignment(unpack)` — wrist 캠으로 출구 보이는 상자면에 정해진 x offset 으로 IK ([manager/skills.py:73](../manager/skills.py#L73))
4. `grip(unpack)` — 정한 x offset 만큼 이동해서 흡착 ON ([manager/skills.py:96](../manager/skills.py#L96))
5. `mv_pos()` //demo 호출 — 레일 이동 자세 ([manager/skills.py:60](../manager/skills.py#L60))
6. `rail(0)` — `robot.rail.command(0.0)`
7. `approach_to_buffer()` //demo 호출 — 투입 공간 위로 이동 ([manager/skills.py:66](../manager/skills.py#L66))
8. `put_to_buffer()` — BoxTable 의 상자 size 참고해 z offset 맞춤 ([manager/skills.py:141](../manager/skills.py#L141))
9. `un_grip()` — 그리퍼 끄기 ([manager/skills.py:110](../manager/skills.py#L110))
10. `mv_pos()` //demo 호출 — 레일 이동 자세 ([manager/skills.py:60](../manager/skills.py#L60))

다시 1번 반복 (버퍼에 둔 상자는 사람이 바로 뺀다고 가정).

## 사용자가 완성해야 할 것 — 단계별

### A. 데모 녹화 ([manager/demos/](../manager/demos/))
`//demo` 가 붙은 함수들은 사용자 녹화 궤적/자세를 쓴다. 각각 `.npz` 파일로 저장. 포맷: `np.savez(path, waypoints=np.ndarray[N, 7])` — 각 행이 `[rail, j1..j6]`.

- [ ] `manager/demos/mv_pos.npz` — 레일 이동 자세 (1자세 또는 짧은 궤적)
- [ ] `manager/demos/approach_to_buffer.npz` — 아무 데서나 투입 공간 위까지의 궤적

### B. offset 상수 ([manager/skills.py:33-44](../manager/skills.py#L33-L44))
alignment / grip / approach / put 의 미세 오프셋. 실측으로 조정.

- [ ] `ALIGN_OFFSET_PACK_Z` — 집기 직전 상자 윗면 위에 떠 있을 거리 ([L33](../manager/skills.py#L33))
- [ ] `ALIGN_OFFSET_UNPACK_X` — 꺼내기 직전 상자 옆면 앞에 떠 있을 거리 ([L34](../manager/skills.py#L34))
- [ ] `GRIP_APPROACH_PACK_Z` — 흡착 ON 전에 z 로 내려가는 양 ([L37](../manager/skills.py#L37))
- [ ] `GRIP_APPROACH_UNPACK_X` — 흡착 ON 전에 x 로 밀어 넣는 양 ([L38](../manager/skills.py#L38))
- [ ] `APPROACH_X_OFFSET` — 자리 x 에서 뒤로 빠져 있는 거리 ([L41](../manager/skills.py#L41))
- [ ] `PUT_BUFFER_Z_OFFSET` — 버퍼 바닥 위 띄우는 양 ([L44](../manager/skills.py#L44))

### C. IK·이동 구현 (현재 NotImplementedError)
RealArm 이 들어오면 `Stacker.ik()` 를 호출해서 CF 목표 → 관절 각으로 풀고 `robot.arm.command()` 로 보내야 함.

- [ ] `alignment()` 안의 IK 호출 ([manager/skills.py:93](../manager/skills.py#L93))
- [ ] `grip()` 안의 상대 이동 + `suction_on` ([manager/skills.py:96](../manager/skills.py#L96))
- [ ] `approach_to_box()` 안의 yz→x 순차 이동 ([manager/skills.py:129](../manager/skills.py#L129))
- [ ] `put_to_buffer()` 안의 z 이동 ([manager/skills.py:141](../manager/skills.py#L141))

### D. 좌표계 상수 ([modules/transform/transform.py](../modules/transform/transform.py))
`rail(x)` 가 CF x → Arduino mm 를 올바로 뽑으려면 먼저 이 보정값들이 필요.

- [ ] `RAIL_ZERO_CF_X`, `RAIL_DIR` — 레일 0 mm 의 CF 위치
- [ ] `WRIST_CAM_IN_EEF`, `ARUCO_BOX_SIZE_M` — alignment 에 필수
- [ ] 자세한 체크리스트: [`pipeline.drawio`](pipeline.drawio) 의 "편집 지점" 섹션

### E. BoxTable 확장 ([manager/boxes.py](../manager/boxes.py))
- [ ] `BoxTable.remove(id)` — UnPacking 10단계에서 사용 (버퍼로 뺀 상자 제거)
- [ ] 각 `BoxEntry` 에 `goal_cf`, `current` 를 CF 좌표로 유지 (이미 `main.py:load_manifest` 가 `goal_cf` 기록)

## 장치 추상화

| 장치 | 역할 |
|---|---|
| Arm | 팔 (Piper 6자유도) 관절 제어 |
| Rail | x 방향 레일 1축 이동 |
| Suction | 흡착 밸브 ON/OFF |
| Camera | 손목·전체뷰 ArUco 기반 측정 |

네 장치를 `Robot` 추상이 묶어 `follow(waypoints)`, `suction_on/off()`, `measure_box()`, `state()`, `stop()` 로 노출.
sim(MuJoCo) / real(하드웨어) / dual(둘 동시) 세 백엔드가 같은 API 뒤에 있음.

## 좌표계

모든 측정·명령은 **Container Frame (CF)** — 바깥 프레임 왼쪽-아래 모서리 원점 — 로 통일.
변환은 [`modules/transform/transform.py`](../modules/transform/transform.py) 가 담당 (CF ↔ MW/RC/RailMM/Cam/EEF).

## 참고

- 구현 상태·파일 매핑·사용자 보정 체크리스트는 [`pipeline.drawio`](pipeline.drawio)
- 장치별 상세: [`details_by_modules/`](details_by_modules/)
- 프레임 치수·좌표계 그림: [`settings/`](settings/)
