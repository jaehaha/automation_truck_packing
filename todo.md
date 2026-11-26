# TODO — 완성해야 하는 것들

지금 상태: `python manager/main.py --mode sim --no-view` → 57/80 성공. 옛 `pick_and_place`+`Sequencer` 경로로 돌아감.
새 `packing/unpacking` 경로로 전환하고 실물까지 올리려면 아래를 채워야 함.

도식 + 자세한 매핑: [pipeline/pipeline.drawio](pipeline/pipeline.drawio), 단계별 흐름: [pipeline/pipeline.md](pipeline/pipeline.md).

---

## 1. 데모 녹화 — sim 에서도 가능

사용자가 손으로 궤적·자세를 녹화해 `.npz` 로 저장. 포맷: `np.savez(path, waypoints=np.ndarray[N, 7])` (각 행 `[rail, j1..j6]`).

- [ ] [manager/demos/mv_pos.npz](manager/demos/) — 레일 이동 자세 (1자세 또는 짧은 궤적)
- [ ] [manager/demos/approach_to_buffer.npz](manager/demos/) — 아무 데서나 투입 공간 위까지의 궤적

> sim 에서 뷰어 열고 자세 세팅 → qpos 저장 스크립트로 가능. 필요하면 녹화 스크립트 요청.

---

## 2. offset 상수 — 실측·시뮬 튜닝

[modules/arm/skills.py](modules/arm/skills.py). 일단 sim 기준 그럴듯한 값 넣고 실물에서 재측정.

- [ ] `ALIGN_OFFSET_PACK_Z` — 집기 직전 상자 윗면 위에 떠 있을 거리 (호버 지점)
- [ ] `ALIGN_OFFSET_UNPACK_X` — 꺼내기 직전 상자 옆면 앞 (호버 지점)
- [ ] `GRIP_APPROACH_PACK_Z` — 호버 → 접촉까지 내려가는 양
- [ ] `GRIP_APPROACH_UNPACK_X` — 호버 → 접촉까지 밀어 넣는 양
- [ ] `APPROACH_X_OFFSET` — 자리 x 에서 뒤로 빠져 있는 거리 (rail.mv 에서도 참조)
- [ ] `PUT_BUFFER_Z_OFFSET` — 버퍼 바닥 위 띄우는 양

---

## 3. IK·이동 구현 — `NotImplementedError` 3곳

[modules/arm/skills.py](modules/arm/skills.py) 의 뼈대만 있는 함수들. `Stacker.ik()` (modules/arm/stacking.py) 호출로 CF 목표 → 관절 각으로 풀고 `robot.arm.command()` 로 보낸다.

- [ ] `skills.alignment()` 안의 2단계 IK 호출 (호버 → 접촉)
- [ ] `skills.approach_to_box()` 안의 yz→x 순차 이동
- [ ] `skills.put_to_buffer()` 안의 z 이동

흡착 ON/OFF 는 이제 `Suction.grip(mode)` / `un_grip()` 로 분리 ([modules/suction/suction.py](modules/suction/suction.py)).
레일 이동은 `Rail.mv(x_cf)` 로 분리 ([modules/rail/rail.py](modules/rail/rail.py)) — `_cf_x_to_rail_m()` 캘리브레이션 필요.

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

## 5. BoxTable 확장

[manager/boxes.py](manager/boxes.py).

- [ ] `BoxTable.remove(id)` — UnPacking 10단계에서 사용 (버퍼로 뺀 상자 제거)
- [ ] `BoxEntry.current` 를 CF 좌표로 유지 (현재 `goal_cf` 는 이미 기록됨)

---

## 6. 실물 하드웨어 백엔드 (❌ → ✅)

### RealArm — Piper CAN
- [ ] [modules/arm/arm.py](modules/arm/arm.py) `RealArm.__init__ / read / command` 구현
- [ ] [modules/arm/piper_keyboard.py](modules/arm/piper_keyboard.py) 를 참고해 CAN 연결
- [ ] EEF pose 조회 메서드 추가 (modules/transform `eef_to_cf()` 에 넘김)

### RealCamera — OpenCV + ArUco
- [ ] [modules/camera/camera.py](modules/camera/camera.py) `RealCamera` 는 뼈대 있음. 실제 캡처·마커 pose 추정은 `detectMarkers + estimatePoseSingleMarkers` 로 이미 작성됨
- [ ] 실측 `K`, `dist`, `extrinsics` 주입만 하면 됨 → 4번 좌표계 섹션 완료 후

---

## 7. 물리적 설치 (한 번만)

- [ ] 프레임 각 코너·중앙에 **ArUco fiducial 3개 이상 부착**
- [ ] 각 마커의 CF 위치 실측 → `transform.ARUCO_FIDUCIAL_IDS` 에 등록
- [ ] **상자 윗면에 ArUco 마커 부착** (크기 통일, `ARUCO_BOX_SIZE_M` 과 일치)
- [ ] 레일을 물리적 0 mm 끝 (모터 반대쪽) 에 둔 상태로 Python 시작 — `INITIALIZE` 자동 전송

---

## 8. main.py 경로 전환

지금은 옛 `pick_and_place` + `Sequencer` 로 돈다. 새 경로로 바꾸려면:

- [ ] [manager/main.py:114](manager/main.py#L114) 의 `pick_and_place(robot, p, s=seq.s)` 를 `actions.packing(robot, box, table)` 로 교체
- [ ] 또는 `--algo {legacy, new}` 플래그 추가해서 둘 다 유지 선택

---

## 작업 순서 추천

1. **지금 당장 (sim)** — 1번(데모 녹화) + 2번(offset 대충) + 3번(IK 구현 — `Stacker.ik` 가져다 쓰기) → 새 `packing/unpacking` 경로가 sim 에서 동작하는지 확인
2. **실물 조립 중** — 7번(마커 부착) + 4번(내부 파라미터 캘리브레이션)
3. **실물 가동 시작** — 6번(RealArm·RealCamera) + 4번 보정값 실측
4. **통합** — 8번(main.py 경로 전환) + 2·4번 보정값 실물 기준 재조정

---

## 지금 돌아가는 것 ✅

- sim 모드 전체 파이프라인 (옛 경로): `python manager/main.py --mode sim --no-view`
- 뷰어 재생: `mjpython manager/main.py --mode sim`
- 아두이노 공유 시리얼: `manager/arduino.py` (레일 + 밸브)
- `RealRail`, `RealSuction` (실물 백엔드)
- `Dual*` 래퍼 (실물 + 시뮬 동시)
- 통합 좌표계 변환기 `transform.py` (보정값만 채우면 됨)
