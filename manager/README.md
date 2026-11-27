# manager

`modules/` 의 장치·알고리즘 모듈을 묶어 **파이프라인 전체를 돌리는 엔트리포인트와 제어 계층**
자세한 파이프라인 흐름: [`../pipeline/pipeline.md`](../pipeline/pipeline.md)

## 파일

| 파일 | 역할 |
|---|---|
| `main.py` | 엔트리포인트. 상자 목록 로드 → 순차 계획·실행·표 갱신 루프 |
| `robot.py` | `Robot` 추상 — 팔·레일·흡착·카메라 네 장치를 묶어 `follow / suction_on/off / measure_box / state / stop` 노출. `.sim / .real / .dual` 팩토리 |
| `actions.py` | 사이클 호출 — `packing(box)`, `unpacking(box)`. arm 동작은 [`modules/arm/skills.py`](../modules/arm/skills.py), 흡착은 [`Suction.grip/un_grip`](../modules/suction/suction.py), 레일은 [`Rail.mv`](../modules/rail/rail.py) |
| `boxes.py` | `BoxTable` — id → `{size, current, goal, state}` 상자 표 |
| `arduino.py` | 레일·흡착이 공유하는 시리얼 연결 (`send / drain / init_shared`) |
| `demos/*.npz` | 사용자 녹화 궤적·자세 (`mv_pos`, `approach_to_buffer`) — 녹화: `demos/record.py` |
| `sequence/moves_*.json` | Packing Model 출력 (적재 순서 + 상자 goal). `main.py` 가 여기서 읽는다 |

## 실행

```bash
conda activate capstone2
cd manager

mjpython main.py --mode sim                                   # 시뮬 + 뷰어
python  main.py --mode sim --no-view                          # 시뮬 계획만 (빠름)
mjpython main.py --mode dual --port /dev/tty.usbmodemXXXX     # 실물 + 뷰어 (디지털 트윈)
python  main.py --mode real --port /dev/tty.usbmodemXXXX      # 실물만
```

| 모드 | 명령 전송 | 상태 read | 뷰어 |
|---|---|---|---|
| sim | Sim* → MuJoCo | MuJoCo | mjpython |
| real | Real* → CAN + Arduino | 하드웨어 | 없음 |
| dual | Dual* = 실물 + 시뮬 동시 | **실물** | mjpython (시뮬=거울) |

## 쓰이는 모듈

```
modules/
  arm/                팔 wrapper + 모션플래닝 + actions (세부 동작)
  rail/  suction/  camera/   장치 wrapper (sim + real 백엔드)
  transform/          통합 좌표계 변환 (CF ↔ MW/RC/RailMM/Cam/EEF)
  packing_algorithm/  어디에 놓을지 → moves.json
  rail_suction.ino    Arduino 1보드 펌웨어
```

## 참고

- 구현 상태·편집 지점: [`../pipeline/pipeline.drawio`](../pipeline/pipeline.drawio)
- 완성해야 할 것: [`../todo.md`](../todo.md)
