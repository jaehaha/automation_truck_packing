# TODO — 완성해야 하는 것들

**지금 상태**
- `python gui.py` → Welcome / 실행 / 개발자 모드 흐름 정상 (다중 화면)
- **"실행 → 적재 실행" 은 dual 모드 (sim+실물 동기화) 로 작동하는데 `RealArm` 미구현** 때문에 바로 종료됨
- sim 만 돌리려면 CLI: `python manager/main.py --mode sim` → **37/80** (내부 가벽 포함 영향)
- 레일 좌표 재정렬: **rail=0 = 투입 위치쪽 끝**, 범위 [0.0, 1.0]. ORIGIN=(0.210, 0.280, 0.672)
- MuJoCo 뷰어: 좌측 패널 숨김, **우측 Control 패널은 표시** (슬라이더로 액추에이터 직접 조작)

도식·흐름: [pipeline/pipeline.drawio](pipeline/pipeline.drawio), [pipeline/pipeline.md](pipeline/pipeline.md), [pipeline/file_call_flow.md](pipeline/file_call_flow.md).

---

## 1. 데모 녹화  [GUI: 개발자 모드 → arm → 녹화]

**녹화 대상 1 개로 축소** — `skills.mv_pos` 와 `skills.reset` 은 더 이상 데모를 쓰지 않고 관절/레일 0 으로 보내는 함수로 구현됨.

| 파일 | 담는 궤적 | 호출 위치 |
|---|---|---|
| `manager/demos/approach_to_buffer.npz` | 안전 자세 → 투입 공간 바로 위 (2~5 keyframe) | [`skills.approach_to_buffer`](modules/arm/skills.py) |

실물 Piper 가 준비되어야 녹화 가능. sim 백엔드도 지원:
```bash
python manager/demos/record.py --name approach_to_buffer --backend sim
```

---

## 2. skills 뼈대 구현 — `NotImplementedError` 3 곳  [GUI: 개발자 모드 → arm → 함수 호출 테스트]

[`modules/arm/skills.py`](modules/arm/skills.py) 에서 `Stacker.ik()` + `robot.follow()` 조합으로 채운다.

- [x] `skills.reset(robot)` — 관절 + 레일 전부 0 ✅
- [x] `skills.mv_pos(robot)` — 관절만 0, 레일 유지 ✅
- [ ] `skills.alignment(robot, mode)` — 2 단계 IK (호버 → 접촉)
- [ ] `skills.approach_to_box(robot, box, mode)` — yz → x 순차 이동
- [ ] `skills.put_to_buffer(robot, box)` — 상자 size 로 z 내리기

**연관**: [`modules/camera/camera.py`](modules/camera/camera.py) `SimCamera.markers()` 를 MuJoCo mocap_pos 로 ground-truth 반환하도록 구현 (지금 빈 리스트).

---

## 3. 보정값 실측

### [`modules/arm/skills.py`](modules/arm/skills.py) offset 상수 (6 개)
`ALIGN_OFFSET_PACK_Z` / `ALIGN_OFFSET_UNPACK_X` / `GRIP_APPROACH_PACK_Z` / `GRIP_APPROACH_UNPACK_X` / `APPROACH_X_OFFSET` ([Rail.APPROACH_X_OFFSET](modules/rail/rail.py) 과 동기화) / `PUT_BUFFER_Z_OFFSET`

### [`modules/transform/transform.py`](modules/transform/transform.py) 상단
- 레일: `RAIL_ZERO_CF_X`, `RAIL_DIR` (주의: 레일 0 = 투입 위치쪽 끝으로 재정렬됨)
- 카메라: `WRIST_CAM_IN_EEF`, `OVERVIEW_CAM_IN_CF`, 내부 파라미터 K/dist (OpenCV 체커보드)
- ArUco: `ARUCO_BOX_SIZE_M`, `ARUCO_FIDUCIAL_IDS` (≥ 3 개)
- (선택) `CF_ORIGIN_IN_MW`

---

## 4. 실물 하드웨어 백엔드  [GUI: 개발자 모드 → arm/suction/rail/camera]

### RealArm — Piper CAN (**실행 플로우의 필수 선행 작업**)
- [ ] [`modules/arm/arm.py`](modules/arm/arm.py) `RealArm.__init__ / read / command` 구현
- [x] `RealArm.set_speed_percent(pct)` 뼈대 추가됨 — `command()` 안에서 `MotionCtrl_2(0x01, MOVE_J, self._speed_percent, 0)` 로 전달하면 됨
- [ ] [`modules/arm/piper_keyboard.py`](modules/arm/piper_keyboard.py) 패턴 참고
- [ ] EEF pose 조회 메서드 (transform `eef_to_cf()` 로 넘김)

### RealCamera — OpenCV + ArUco
- [x] 뼈대 작성됨 — `detectMarkers` + `estimatePoseSingleMarkers`
- [ ] 실측 `K`, `dist`, `extrinsics` 주입 (§3 완료 후)

### RealRail / RealSuction
- [x] 아두이노 통해 명령 전송 작동 — `test.py rail`, `test.py suction` 으로 확인 가능 (또는 GUI 개발자 모드)
- [x] `RealRail.zero_offset_m` 기본값 0.0 (Arduino mm ↔ RC m 1:1 대응)

---

## 5. 물리적 설치 (한 번만)

- [ ] 프레임 코너·중앙에 ArUco fiducial ≥ 3 개 부착, CF 위치 실측
- [ ] 상자 윗면에 ArUco 마커 (크기 통일)
- [ ] 레일을 **물리적 투입 위치쪽 끝**에 두고 Python 시작 → Arduino `INITIALIZE` 가 그 지점을 0 mm 로 설정

---

## 6. main.py 레거시 → 새 경로 전환

### 단계 ① — `pick_and_place` → `actions.packing`
위 §1 ~ §2 가 sim 에서 완주한 뒤:
- [ ] [`manager/main.py:138-142`](manager/main.py#L138-L142) 재생 루프를 `actions.packing(robot, box, table, speed=args.speed)` 로 교체
- [ ] `load_manifest` 를 CF 좌표 통일 (현재 MW/RC 혼용)

### 단계 ② — 레거시 삭제
- [ ] [`manager/actions.py`](manager/actions.py) `pick_and_place` 함수 제거
- [ ] [`manager/main.py`](manager/main.py) 의 Sequencer 임포트·호출 제거
- [ ] [`modules/arm/sequence.py`](modules/arm/sequence.py) 파일 삭제
- [ ] [`modules/arm/cache/templates.pkl`](modules/arm/cache/) 삭제
- [ ] [`modules/arm/rules.py`](modules/arm/rules.py) `KEEPOUT_X` 만 `main.py` 로 옮기고 나머지 검토 삭제
- [ ] [`modules/arm/stacking.py`](modules/arm/stacking.py) — `ik`/`path`/`free` 는 skills 가 쓰므로 유지, 플래닝 전용 메서드 (`plan`/`depart`/`_roll`/…) 삭제 검토. `ArmKinematics` 로 리네임 고려

---

## 7. GUI 보강 (선택)

- [x] 개발자 모드 → arm → 함수 호출 테스트 에 **"실물 연동" 체크박스** 추가 (OFF=sim, ON=dual)
- [x] 개발자 모드 → arm → `reset` 버튼 추가
- [x] arm 테스트 시 MuJoCo 뷰어 자동 띄움 (`on_step` 콜백으로 sync)
- [x] MuJoCo 뷰어 우측 Control 패널 복구 → 슬라이더로 액추에이터 조작 가능
- [ ] "실행 → 적재 실행" 에 **"실물 연동 OFF (sim 만)" 토글** — RealArm 완성 전 안전 모드
- [ ] 추론 창 progress bar: plan_moves.py stdout 파싱해 **%** 표시 (현재는 indeterminate)
- [ ] arm 함수 호출 테스트 결과를 로그에 하이라이트
- [ ] camera 개발 화면 — ArUco 마커 검출 오버레이 추가 (디버깅용)

---

## 작업 순서 추천

1. **§7 "실행 → 적재 실행" 토글** — 지금 바로 "실행 → 적재 실행" 을 sim 으로 돌릴 수 있게
2. **§1 데모 녹화 (approach_to_buffer 만) + §2 skills IK 3 개 + SimCamera** → sim 에서 새 경로 완주 확인
3. **§6 단계 ①** — main.py 재생 루프를 `actions.packing` 으로 교체
4. **§6 단계 ②** — 레거시 제거
5. **§5 물리 설치 + §3 보정값**
6. **§4 RealArm 구현** → "실행 → 적재 실행" 디지털 트윈 실제 동작

---

## 지금 돌아가는 것 ✅

- `python gui.py` — Welcome / 실행 / 개발자 메뉴 + 5 개 서브 멀티화면. 창 크기 고정 1000×720, 뒤로가기 깨끗
- sim 전체 파이프라인 (레거시 경로): `python manager/main.py --mode sim` → **37/80** 성공, 뷰어 우측 Control 패널 사용 가능
- `Suction.grip()` / `un_grip()` (인자 없음), `Rail.mv(x_cf)`
- `skills.reset`, `skills.mv_pos` — 데모 없이 관절/레일 제어
- `RealRail`, `RealSuction`, `Arduino` 공유 시리얼. 레일 0 = 투입 위치쪽 끝
- `SimCamera` RGB 렌더링 / `RealCamera` ArUco pose 추정 코드
- `BoxTable.remove(id)`, actions.packing/unpacking 정의 (호출 대기)
- MuJoCo 뷰어 자동 배치 (Mac osascript / Linux wmctrl)
- 데모 녹화 (`record.py` real/sim 백엔드)

### 모델·배치 최근 변경
- 내부 가로대 2 개: x ∈ [600, 640], [1200, 1240]
- 바닥판 2 개 (흰색, 비충돌): board_1 [40, 600], board_2 [640, 1200]
- 끝쪽 [1240, 1560] 320×480 mm 빈공간 + 흰색 가벽 (wall_partition, x=1240~1245)
- 양쪽 긴 옆벽 (`wall_front`, `wall_back`) 제거 → 내부 시인성 향상
- `frame_visual` STL 메시 숨김, `frame_col` 박스들이 그대로 보임 (실제 collision 그대로)

---

## 8. 단일 실행 파일로 빌드 (나중에)

지금은 `python gui.py` 로 돌리지만, 최종 배포 단계에서는 더블클릭으로 뜨는 **독립 실행형 앱 하나**로 묶는다.

### 빌드 도구 선택
- **macOS**: [`py2app`](https://py2app.readthedocs.io) — `.app` 번들 생성. `Info.plist` 에 `CFBundleIconFile` 지정 → Dock 아이콘까지 교체 가능 (현재 `iconphoto` 로는 창 아이콘만 바뀜).
- **Linux**: [`PyInstaller`](https://pyinstaller.org) `--onefile` — 단일 ELF.
- **Windows**: `PyInstaller --onefile --windowed` — 콘솔 없는 `.exe`.
- 크로스플랫폼 한 가지로 통일하고 싶으면 PyInstaller 로도 Mac `.app` 생성 가능 (`--windowed --icon=assets/icon.icns`).

### 아이콘 파일 준비
- `assets/icon.png` (현재, 창 아이콘용) 외에 플랫폼별 포맷 추가:
  - macOS: `assets/icon.icns` (여러 해상도 포함, `iconutil -c icns icon.iconset`)
  - Windows: `assets/icon.ico` (16/32/48/256 멀티해상도)
- Linux `.desktop` 파일은 PNG 그대로 사용 가능.

### 번들에 포함해야 하는 자원
빌드 스펙 (`py2app` 의 `setup.py` / PyInstaller 의 `.spec`) 에서 `--add-data` 또는 `data_files` 로 명시:
- `assets/` (아이콘)
- `modules/arm/model/` MuJoCo XML + mesh (STL/OBJ)
- `manager/demos/*.npz` (데모 궤적)
- `modules/arm/cache/templates.pkl` (레거시 제거 후엔 불필요)
- `manager/sequence/` 샘플 (`input.json` 등 기본 경로)

### subprocess 모델 재검토 (핵심 리팩터)
현재 GUI 는 `sys.executable` 로 `plan_moves.py` / `manager/main.py` 등을 **별도 python 프로세스**로 띄운다. 번들 안에는 외부 python 이 없으므로:
- **옵션 A** — 번들에 python 인터프리터 포함 (py2app/PyInstaller 둘 다 기본 지원), subprocess 호출 시 `sys.executable` 이 번들 내부 python 을 가리킴 → 지금 코드 거의 그대로 동작. 단 `mjpython` (MuJoCo Mac 뷰어) 을 함께 번들링해야 "실행 → 적재 실행" 의 뷰어가 뜸 — 이게 가장 큰 숙제.
- **옵션 B** — subprocess 를 **스레드 호출로 교체**. `plan_moves.main()` / `manager.main.main()` 을 import 해서 함수로 직접 실행, stdout 캡처는 `contextlib.redirect_stdout` 로. 장점: 번들 가볍고 python 하나로 동작. 단점: MuJoCo 뷰어는 Mac 에서 메인 스레드 전용이라 GUI 메인 스레드와 충돌 — 뷰어 띄우는 작업은 여전히 자식 프로세스 필요.
- 현실적 절충: **알고리즘 추론은 옵션 B (스레드)**, **뷰어 띄우는 적재 실행만 옵션 A (서브프로세스 + mjpython 번들)**.

### Dock 아이콘 (macOS)
`py2app` 번들의 `Info.plist` 에:
```xml
<key>CFBundleIconFile</key><string>icon.icns</string>
<key>LSUIElement</key><false/>
```
→ 창 좌상단 + Dock + Cmd-Tab 모두 교체됨. 현재 `_apply_icon()` 은 그대로 둬도 OK (번들 안에서도 창 아이콘 보장).

### 산출물 흐름
`setup.py` (py2app) → `python setup.py py2app` → `dist/ManiBox.app` → `.dmg` 또는 그대로 배포.

**의존성 주의**: `mujoco`, `piper_sdk`, `pyserial`, `opencv-python` 모두 native 확장 모듈이라 `includes=` / `packages=` 에 명시해야 함. 특히 MuJoCo shared library (`libmujoco.dylib`) 경로 문제가 가장 흔한 실패 원인.
