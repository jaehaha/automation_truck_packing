# TODO — 완성해야 하는 것들

지금 상태: `python manager/main.py --mode sim --no-view` → **46/80 성공**. 옛 `pick_and_place`+`Sequencer` 경로로 돌아감.
새 `packing/unpacking`(skills + ArUco) 경로로 전환하고 레거시를 **완전히 걷어내려면** 아래를 채워야 함.

도식 + 자세한 매핑: [pipeline/pipeline.drawio](pipeline/pipeline.drawio), 단계별 흐름: [pipeline/pipeline.md](pipeline/pipeline.md), 호출 트리: [pipeline/file_call_flow.md](pipeline/file_call_flow.md).

---

## 1. 데모 녹화 — sim 뷰어에서 가능

사용자가 손으로 궤적·자세를 녹화해 `.npz` 로 저장. 포맷: `np.savez(path, waypoints=np.ndarray[N, 7])` — 각 행 `[rail, j1..j6]` (m / rad).
녹화 프로그램: [`manager/demos/record.py`](manager/demos/record.py) (조작·포맷 상세: [manager/demos/README.md](manager/demos/README.md)).

### 녹화해야 하는 2개

| 파일 | 호출 위치 | 무엇을 담나 |
|---|---|---|
| [`manager/demos/mv_pos.npz`](manager/demos/) | [`skills.mv_pos`](modules/arm/skills.py) | **레일 이동 중 유지할 안전 자세** — 상자를 든 채 레일만 움직일 때, 팔이 프레임·벽에 안 걸리는 자세 하나(또는 짧은 궤적) |
| [`manager/demos/approach_to_buffer.npz`](manager/demos/) | [`skills.approach_to_buffer`](modules/arm/skills.py) | **mv_pos 자세 → 투입 공간 바로 위까지** 의 궤적. 2 ~ 5 개 keyframe 정도, 프레임 가로대·벽에 안 부딪히게 경유 |

### 녹화 명령

```bash
# Mac (뷰어 필수)
mjpython manager/demos/record.py --name mv_pos
mjpython manager/demos/record.py --name approach_to_buffer

# Linux
python manager/demos/record.py --name mv_pos
python manager/demos/record.py --name approach_to_buffer
```

조작: `0`=레일 / `1~6`=관절, `←→` 이동, `↑↓` step, `SPACE` keyframe 추가, `BACKSPACE` 삭제, `r` home 리셋, `s` 저장 후 종료, `q`/`ESC` 취소.

### 체크리스트

- [ ] `mv_pos.npz` 녹화 — 1 자세로 충분. 레일 이동 중 팔이 안 흔들리는 "운반 자세"
- [ ] `approach_to_buffer.npz` 녹화 — 처음 keyframe 을 `mv_pos` 와 같은 자세로 시작하면 두 함수 연결이 매끄럽다. 마지막 keyframe 은 [`PICK_XY=(0.29, 0.28)`](modules/arm/stacking.py#L37) 바로 위 (상자 높이보다 조금 위), 흡착면이 아래(-z)를 향하도록

---

## 2. offset 상수 — 실측·시뮬 튜닝

[modules/arm/skills.py](modules/arm/skills.py). 일단 sim 기준 그럴듯한 값 넣고 실물에서 재측정.

- [ ] `ALIGN_OFFSET_PACK_Z` — 집기 직전 상자 윗면 위에 떠 있을 거리 (호버 지점)
- [ ] `ALIGN_OFFSET_UNPACK_X` — 꺼내기 직전 상자 옆면 앞 (호버 지점)
- [ ] `GRIP_APPROACH_PACK_Z` — 호버 → 접촉까지 내려가는 양
- [ ] `GRIP_APPROACH_UNPACK_X` — 호버 → 접촉까지 밀어 넣는 양
- [ ] `APPROACH_X_OFFSET` ([skills.py](modules/arm/skills.py) + [Rail.APPROACH_X_OFFSET](modules/rail/rail.py)) — 자리 x 에서 팔 베이스 뒤로 빠져 있는 거리
- [ ] `PUT_BUFFER_Z_OFFSET` — 버퍼 바닥 위 띄우는 양

---

## 3. IK · ArUco 구현 — `NotImplementedError` 3곳 + SimCamera

### skills 뼈대 채우기 ([modules/arm/skills.py](modules/arm/skills.py))

`Stacker.ik()` ([modules/arm/stacking.py:202](modules/arm/stacking.py#L202)) 를 호출해 CF 목표 → 관절 각으로 풀고 `robot.arm.command()` 로 보낸다. 여러 단계인 경우 중간 waypoint 는 `robot.follow()` 로 연결.

- [ ] `skills.alignment(robot, mode)` — 2단계 IK (호버 → 접촉). `robot.camera.markers()` 로 상자 중심 CF 좌표 얻음
- [ ] `skills.approach_to_box(robot, box, mode)` — box.goal_cf / box.current 의 yz 로 자세 맞춘 뒤 x 로 미세 이동
- [ ] `skills.put_to_buffer(robot, box)` — 상자 size 로 z 내리기 (버퍼 바닥 위 `PUT_BUFFER_Z_OFFSET`)

### SimCamera 가짜 ArUco ([modules/camera/camera.py:43](modules/camera/camera.py#L43))

지금 `SimCamera.markers()` 가 빈 리스트 → sim 에서 `skills.alignment` 바로 실패. ground-truth 를 돌려주도록 구현:

- [ ] MuJoCo `d.mocap_pos` 에서 지금 "집어야 할" 상자 중심 + yaw 를 꺼내 `[(id, xyz_cf, yaw_cf)]` 형식으로 반환
- [ ] 마커 id 는 `BoxTable` 의 box_id 해시 등으로 매핑 (어떤 상자를 가리키는지 알게)

### 분리된 책임 (이미 됨)

흡착 ON/OFF 는 `Suction.grip(mode)` / `un_grip()` ([modules/suction/suction.py](modules/suction/suction.py)).
레일 이동은 `Rail.mv(x_cf)` ([modules/rail/rail.py](modules/rail/rail.py)) — `_cf_x_to_rail_m()` 캘리브 필요 (§4).

---

## 4. 통합 좌표계 보정값 — 실측

[modules/transform/transform.py](modules/transform/transform.py) 상단 상수들.

### 레일
- [ ] `RAIL_ZERO_CF_X` — Arduino 레일 0 mm 를 CF x 어디로 둘지 (레일을 물리적 0 끝에 두고 측정)
- [ ] `RAIL_DIR` — Arduino mm 증가 방향 (+1 이면 CF +x)

### 카메라
- [ ] `WRIST_CAM_IN_EEF` — 손목 카메라 → TCP 4×4 변환 (CAD 수치 or hand-eye calibration)
- [ ] `OVERVIEW_CAM_IN_CF` — 전체뷰 카메라 → CF 4×4 (ArUco fiducial 로 자동 보정 가능)
- [ ] 손목·전체뷰 각각 **내부 파라미터 `K`, `dist`** — OpenCV 체커보드 스크립트 (별도 수행)

### ArUco
- [ ] `ARUCO_BOX_SIZE_M` — 상자 윗면 마커 한 변 크기
- [ ] `ARUCO_FIDUCIAL_IDS` — 프레임 고정 마커 id → CF 위치 (최소 3개)

### (선택)
- [ ] `CF_ORIGIN_IN_MW` — MuJoCo world 와 CF 원점 어긋나면 입력. 모델 shift 로 대신 맞출 수도 있음

---

## 5. 실물 하드웨어 백엔드 (❌ → ✅)

### RealArm — Piper CAN
- [ ] [modules/arm/arm.py](modules/arm/arm.py) `RealArm.__init__ / read / command` 구현
- [ ] [modules/arm/piper_keyboard.py](modules/arm/piper_keyboard.py) 를 참고해 CAN 연결
- [ ] EEF pose 조회 메서드 추가 (modules/transform `eef_to_cf()` 에 넘김)

### RealCamera — OpenCV + ArUco
- [ ] [modules/camera/camera.py](modules/camera/camera.py) `RealCamera` 는 뼈대 있음. 실제 캡처·마커 pose 추정은 `detectMarkers + estimatePoseSingleMarkers` 로 이미 작성됨
- [ ] 실측 `K`, `dist`, `extrinsics` 주입만 하면 됨 → 4번 좌표계 섹션 완료 후

---

## 6. 물리적 설치 (한 번만)

- [ ] 프레임 각 코너·중앙에 **ArUco fiducial 3개 이상 부착**
- [ ] 각 마커의 CF 위치 실측 → `transform.ARUCO_FIDUCIAL_IDS` 에 등록
- [ ] **상자 윗면에 ArUco 마커 부착** (크기 통일, `ARUCO_BOX_SIZE_M` 과 일치)
- [ ] 레일을 물리적 0 mm 끝 (모터 반대쪽) 에 둔 상태로 Python 시작 — `INITIALIZE` 자동 전송

---

## 7. main.py 경로 전환 + 레거시 제거

### 전환 (단계 ①)
위 §1 ~ §3 이 sim 에서 동작하는 걸 확인한 뒤:

- [ ] [manager/main.py:138-142](manager/main.py#L138-L142) 재생 루프를 다음으로 교체
  ```python
  for e in plans_entries:
      box = table[e['box_id']]
      actions.packing(robot, box, table)
  ```
- [ ] [manager/main.py](manager/main.py) 의 `load_manifest` 를 **CF 좌표로 반환**하도록 정비 (현재 MW/RC 를 섞어 넘김). Sequencer 가 필요 없어지면 ORIGIN·mw_to_cf 변환도 걷어낸다

### 레거시 제거 (단계 ② — 새 경로가 안정된 뒤)
"과도기" 로 남겨 둔 Sequencer-기반 코드 완전 삭제:

- [ ] [manager/actions.py:76-95](manager/actions.py#L76-L95) `pick_and_place(robot, plan, s=None)` 함수 삭제
- [ ] [manager/main.py:31](manager/main.py#L31) `from sequence import BOXES, Sequencer` — Sequencer 임포트 제거. `BOXES` 는 `modules/arm/stacking.py` 에서 직접 받도록
- [ ] [manager/main.py:36](manager/main.py#L36) `from actions import pick_and_place` 삭제 (전환 후 미사용)
- [ ] [manager/main.py:90](manager/main.py#L90) `seq = Sequencer(types)` 삭제
- [ ] [manager/main.py:114-128](manager/main.py#L114-L128) Sequencer 기반 **플래닝 단계**(seq.plan / seq.commit / seq.save) 삭제 → 플래닝은 `actions.packing` 안에서 매 상자마다 실행
- [ ] [manager/main.py:94](manager/main.py#L94) `viewer_ctx = mujoco.viewer.launch_passive(seq.s.m, seq.s.d)` 의 `seq.s` 의존 제거 — MuJoCo 모델을 `Robot.sim()` 안에서 자체 로드하도록
- [ ] [modules/arm/sequence.py](modules/arm/sequence.py) 전체 삭제 (Sequencer 는 레거시 전용)
- [ ] [modules/arm/cache/templates.pkl](modules/arm/cache/) 삭제 — Sequencer 용 템플릿 캐시
- [ ] [modules/arm/rules.py](modules/arm/rules.py) 는 `BOXES`·`EXTRA_BOXES`·`KEEPOUT_X`·`place_candidates` 중 `KEEPOUT_X` 만 `job.py` 로 옮기고 나머지는 삭제 검토

### Stacker 는 유지할지 검토
`Stacker.ik()`·`Stacker.path()`·`Stacker.free()` 는 skills 가 호출할 **범용 IK/충돌 유틸**이므로 **유지**.
반면 `Stacker.plan()`·`Stacker.depart()`·`Stacker.commit()` 같은 플래닝 메서드는 레거시와 함께 걷어낼 수 있음.

- [ ] [modules/arm/stacking.py](modules/arm/stacking.py) 에서 IK/충돌 유틸만 남기고 플래닝 전용 메서드 (`plan`, `depart`, `_roll`, `_line`, `_connect`, `_lift`, `_pick_tpl` 등) 분리/삭제 검토
- [ ] 또는 `Stacker` 를 **`ArmKinematics`** 같은 이름으로 리네임해 역할 명확화

---

## 8. BoxTable

- [x] `BoxTable.remove(id)` ([manager/boxes.py](manager/boxes.py)) — UnPacking 11단계에서 사용 ✅
- [ ] `BoxEntry.current` 를 **CF 좌표로 유지** (레거시 때문에 지금 MW 혼용). 레거시 걷어낼 때 함께

---

## 작업 순서 추천

1. **sim 에서 새 경로 검증** — §1 데모 녹화 + §2 offset 대충 + §3 (SimCamera ground-truth + IK 3곳) → `actions.packing` 이 sim 에서 완주하는지 확인
2. **단계 ①: main.py 전환** — §7.① 재생 루프 교체 후 sim 46/80 수준으로 동작 확인
3. **단계 ②: 레거시 제거** — §7.② Sequencer / sequence.py / rules.py / templates.pkl 삭제. Stacker 는 IK 유틸로 축소
4. **실물 조립 중** — §6 (마커 부착) + §4 (내부 파라미터 캘리브레이션)
5. **실물 가동 시작** — §5 (RealArm·RealCamera) + §4 보정값 실측
6. **통합** — §2·§4 보정값 실물 기준 재조정, dual 모드로 디지털 트윈 확인

---

## 지금 돌아가는 것 ✅

- sim 모드 전체 파이프라인 (옛 경로): `python manager/main.py --mode sim --no-view` → 46/80
- 뷰어 재생: `python manager/main.py --mode sim` (Linux) / `mjpython manager/main.py --mode sim` (Mac)
- 아두이노 공유 시리얼: `manager/arduino.py` (레일 + 밸브)
- `RealRail`, `RealSuction` (실물 백엔드 — 보정값만 채우면)
- `Dual*` 래퍼 (실물 + 시뮬 동시, 타이밍 A)
- 통합 좌표계 변환기 `transform.py` (보정값만 채우면 됨)
- `Suction.grip(mode)` / `un_grip()`, `Rail.mv(x_cf)` — 새 API 대기 중
- `BoxTable.remove(id)` 추가됨
