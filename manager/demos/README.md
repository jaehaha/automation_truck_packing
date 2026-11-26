# manager/demos

사용자 녹화 궤적·자세. `modules/arm/skills.py` 의 `_load_demo(name)` 이 여기서 `.npz` 파일을 읽는다.

포맷: `np.savez(path, waypoints=np.ndarray[N, 7])` — 각 행 `[rail, j1..j6]` (m / rad).

## 필요한 파일

| 이름 | 용도 | 호출 위치 |
|---|---|---|
| `mv_pos.npz` | 레일 이동 자세 (1자세 또는 짧은 궤적) | `skills.mv_pos` |
| `approach_to_buffer.npz` | 투입 공간 위까지의 궤적 | `skills.approach_to_buffer` |

## 녹화하기

```bash
# Mac (뷰어 필요)
mjpython manager/demos/record.py --name mv_pos
mjpython manager/demos/record.py --name approach_to_buffer

# Linux
python manager/demos/record.py --name mv_pos
```

### 조작 키

| 키 | 동작 |
|---|---|
| `0` | 레일 선택 (축 0) |
| `1` ~ `6` | 관절 1 ~ 6 선택 |
| ← / → | 선택 축을 step 만큼 감소 / 증가 |
| ↑ / ↓ | step 크기 증가 / 감소 |
| `SPACE` | 현재 자세를 keyframe 으로 추가 |
| `BACKSPACE` | 마지막 keyframe 삭제 |
| `r` | home 자세로 리셋 |
| `s` | 저장 후 종료 |
| `q` / `ESC` | 저장 없이 종료 |

### 팁
- 첫 keyframe 은 "시작 자세" (예: READY home) 부터. `skills` 는 저장된 자세를 순서대로 `robot.follow()` 로 재생한다.
- `mv_pos.npz` 는 **1개 자세**만 담아도 충분 (레일 이동 중 유지할 안전 자세).
- `approach_to_buffer.npz` 는 **2~5개 자세** 정도 — 중간 지점을 넣어 팔이 프레임에 걸리지 않게 경유.
- 레일 축(0) 과 관절 축(1~6) 의 step 은 각각 독립적으로 기억되니, 축을 바꿀 때 step 도 자동 복원.

## 파일 구조

```
manager/demos/
├── README.md              (이 파일)
├── record.py              녹화 프로그램
├── mv_pos.npz             (사용자 녹화)
└── approach_to_buffer.npz (사용자 녹화)
```
