# manager: 통합 관리

**arm · rail · vacuum · camera 네 장치를 하나의 Robot 추상 뒤에 묶어서 관리**하는 계층.

파이프라인 전체 흐름(Packing Model → Motion Planning → Robot 실행 → BoxTable 갱신)을 돌리는 엔트리포인트와, 장치 명령·skill·상자 상태를 중앙에서 다룬다. 상세 흐름은 [`../pipeline/pipeline.drawio`](../pipeline/pipeline.drawio) 참고.

## 파일

| 파일 | 역할 |
|---|---|
| `main.py` | 엔트리포인트. 상자 목록 로드 → 순차 계획·실행·표 갱신 루프 |
| `robot.py` | `Robot` 추상. 네 장치를 묶어 `follow()`, `suction_on/off()`, `measure_box()`, `state()`, `stop()` 노출. `.sim / .real / .dual` 팩토리 |
| `skills.py` | `pick_and_place()` 등 Robot 메서드를 조합한 동작 |
| `boxes.py` | `BoxTable`. id → {size, current, goal, state} |
| `arduino.py` | 레일·흡착이 공유하는 시리얼 연결 (`send`, `drain`, `init_shared`) |

## 실행

```bash
conda activate capstone2
cd manager

mjpython main.py --mode sim                            # 시뮬 + 뷰어
python  main.py --mode sim --no-view                   # 시뮬 계획만 (빠름)
mjpython main.py --mode dual --port /dev/tty.usbmodemXXXX  # 실물 + 뷰어 (디지털 트윈)
python  main.py --mode real --port /dev/tty.usbmodemXXXX   # 실물만
```

| 모드 | 명령 전송 | 상태 read | 뷰어 |
|---|---|---|---|
| sim | Sim* → MuJoCo | MuJoCo | mjpython |
| real | Real* → CAN + 아두이노 | 하드웨어 | 없음 |
| dual | Dual* = 실물 + 시뮬 동시 | **실물** | mjpython (시뮬이 거울) |

**타이밍 A (dual)**: 같은 명령을 양쪽에 동시 전송. 시뮬은 즉시 반영, 실물은 모터 속도로 따라옴.

## 쓰이는 모듈

```
modules/
  packing/   어디에 놓을지 → moves.json
  mp/        어떻게 옮길지 → [레일, 관절1..6] 경유점
  arm/  rail/  vacuum/  camera/     장치 인터페이스 (sim + real 백엔드)
  rail_vacuum.ino                   Arduino 1보드 펌웨어
```

## 구현 상태

구현 상태별 블록 색깔은 [`../pipeline/pipeline.drawio`](../pipeline/pipeline.drawio) 참고.

- ✅ 완료: Rail, Suction, Arduino 공유 시리얼, Robot/skills/BoxTable, main.py
- ⚠️ 부분: Arm (sim ✅ / real ❌), Camera (sim ✅ / real ❌), `measure_box()` 더미
- ❌ 미구현: `RealArm` (Piper CAN), `RealCamera` (OpenCV + ArUco)

기준 결과 (`packing/moves_big.json`, 80개, sim 모드): **성공 57 / 실패 23** (투입 구역 20 + 실제 실패 3).
