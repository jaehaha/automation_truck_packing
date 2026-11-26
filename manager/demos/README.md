# manager/demos

사용자 녹화 궤적·자세. `modules/arm/skills.py` 의 `_load_demo(name)` 이 여기서 `.npz` 파일을 읽는다.

포맷: `np.savez(path, waypoints=np.ndarray[N, 7])` — 각 행 `[rail, j1..j6]` (m / rad).

## 필요한 파일

| 이름 | 용도 | 호출 위치 |
|---|---|---|
| `mv_pos.npz` | 레일 이동 자세 (1자세 또는 짧은 궤적) | `skills.mv_pos` |
| `approach_to_buffer.npz` | 투입 공간 위까지의 궤적 | `skills.approach_to_buffer` |

## 녹화하기

[`record.py`](record.py) — 두 가지 백엔드 지원.

### (기본) 실물 Piper CAN

```bash
# Piper 만
python manager/demos/record.py --name mv_pos --can can0

# Piper + Arduino 레일 (레일 값도 기록)
python manager/demos/record.py --name mv_pos --can can0 --port /dev/ttyACM0
```

사전 조건:
```bash
sudo ip link set can0 up type can bitrate 1000000
# Arduino 가 있으면 사용자가 dialout 그룹에 있어야 함
```

### (대안) 시뮬레이션 뷰어

```bash
# Mac
mjpython manager/demos/record.py --name mv_pos --backend sim
# Linux
python manager/demos/record.py --name mv_pos --backend sim
```

## 조작 키 (양쪽 공통)

| 키 | 동작 |
|---|---|
| `0` | 레일 선택 (real 에서 `--port` 지정 시 Arduino 명령 전송) |
| `1` ~ `6` | 관절 1 ~ 6 선택 |
| ← / → | 선택 축을 step 만큼 감소 / 증가 |
| ↑ / ↓ | step 크기 증가 / 감소 |
| `SPACE` | **현재 자세를 keyframe 으로 추가** |
| `BACKSPACE` | 마지막 keyframe 삭제 |
| `h` | 조인트 모두 0° 로 복귀 (real: 팔이 실제로 움직임, 주의) |
| `s` | 저장 후 종료 |
| `q` / `ESC` | 저장 없이 종료 |

## 실물 녹화 시 안전 수칙

- 팔 주변을 치울 것. `h` (home) 을 누르면 즉시 0° 로 이동함
- 초기 step 은 조인트 1°, 레일 10 mm — ↑↓ 로 조절. 처음에는 작은 step 부터
- `joint-speed` 기본 30 deg/s, `--joint-speed 10` 등으로 느리게 하면 더 안전
- 종료 시 기본은 enable 유지 (현재 자세 고정). 완전 전원 off 하려면 `--disable-on-exit` 또는 수동으로 전원 끄기

## 팁

- 첫 keyframe 은 "시작 자세" (예: READY home) 부터. `skills` 는 저장된 자세를 순서대로 `robot.follow()` 로 재생한다
- `mv_pos.npz` 는 **1개 자세**만 담아도 충분 (레일 이동 중 유지할 안전 자세)
- `approach_to_buffer.npz` 는 **2~5개 자세** 정도 — 중간 지점을 넣어 팔이 프레임에 걸리지 않게 경유
- 레일 축(0) 과 관절 축(1~6) 의 step 은 각각 독립적으로 기억

## GUI 에서 쓰기

[`gui.py`](../../gui.py) 의 "● 데모 녹화" 버튼 → **새 터미널 창** 이 열리며 record.py 가 그 안에서 실행된다. 키 입력은 터미널 창에 포커스 둔 상태로. (GUI 자체 창에서는 입력 받지 못함 — raw 터미널 I/O 때문)

## 파일 구조

```
manager/demos/
├── README.md              (이 파일)
├── record.py              녹화 프로그램 (real/sim 백엔드)
├── mv_pos.npz             (사용자 녹화)
└── approach_to_buffer.npz (사용자 녹화)
```
