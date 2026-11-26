# 실행 방법

## 0. 가장 쉬운 방법 — GUI 런처

```bash
conda activate capstone2
python gui.py
```

창 하나에서 모드·포트 설정 + 파이프라인 실행 / 데모 녹화 / 장치 테스트 / 환경 점검을 다 할 수 있습니다. Mac 은 뷰어 작업에 `mjpython` 자동 사용. 아래 §1 ~ §7 은 CLI 로 직접 쓰고 싶을 때 참고.

## 1. 환경 설치 (Mac·Linux 공통)

```bash
git clone <this-repo>
cd automation_truck_packing

conda env create -f environment.yml     # name: capstone2, python 3.12 + 의존성
conda activate capstone2
```

`environment.yml` 이 설치하는 주요 pip 패키지: `mujoco`, `numpy`, `scipy`, `gymnasium`, `imageio`, `pillow`, `torch`, `piper_sdk`, `pyserial`, `opencv-contrib-python`.

## 2. 실행 명령

| 모드 | Mac | Linux |
|---|---|---|
| 시뮬 + 뷰어 | `mjpython manager/main.py --mode sim` | `python manager/main.py --mode sim` |
| 시뮬 계획만 (빠름, 뷰어 X) | `python manager/main.py --mode sim --no-view` | `python manager/main.py --mode sim --no-view` |
| 실물 + 시뮬 디지털 트윈 | `mjpython manager/main.py --mode dual --port <PORT>` | `python manager/main.py --mode dual --port <PORT> --can can0` |
| 실물만 | `python manager/main.py --mode real --port <PORT>` | `python manager/main.py --mode real --port <PORT> --can can0` |

Mac 은 MuJoCo 뷰어를 띄울 때 **`mjpython`** 이 필수. Linux 는 그냥 `python` 으로 됩니다.

### 공통 옵션
- `--mode {sim, real, dual}`
- `--no-view` — 뷰어 끄기 (sim 에서 계획만 돌릴 때)
- `--port /dev/...` — 아두이노 시리얼 포트 (real/dual)
- `--can can0` — Piper 팔 CAN 인터페이스 (real/dual, Linux)
- `--speed <float>` — 재생 배속 (기본 4.0)

## 3. 하드웨어 — `real` / `dual` 모드에서만

### 아두이노 (레일 + 흡착 밸브)
- 보드에 [`modules/rail_suction.ino`](modules/rail_suction.ino) 업로드
- 포트 확인
  - Mac: `ls /dev/tty.usbmodem*`
  - Linux: `ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null`
- **Linux 권한**: 일반 사용자가 시리얼을 열려면 `dialout` 그룹 가입
  ```bash
  sudo usermod -aG dialout $USER     # 로그아웃 → 재로그인
  ```
- 레일을 **물리적 0 mm 끝 (모터 반대쪽)** 에 둔 상태로 Python 시작 — `INITIALIZE` 가 자동 전송됨

### Piper 팔 (CAN)
- Linux SocketCAN 올리기
  ```bash
  sudo ip link set can0 up type can bitrate 1000000
  ```
- 속도 모를 때: `sudo bash can_bitrate_scan.sh can0` (python-can 추가 필요)
- Mac 은 USB-CAN 어댑터 벤더 드라이버 설치 필요

### 카메라 (손목·전체뷰)
- 내부 파라미터 `K`, `dist` 는 OpenCV 체커보드로 별도 보정
- ArUco fiducial 설치 후 `transform.ARUCO_FIDUCIAL_IDS` 등록 — [todo.md §6](todo.md) 참고

## 4. 뷰어 환경 (Linux)

- X11 세션: 자동. SSH 원격은 `ssh -X user@host` 로
- Wayland 는 보통 XWayland 로 됨. 안 뜨면 `QT_QPA_PLATFORM=xcb python manager/main.py ...` 로 강제
- GPU 가속: `MUJOCO_GL=egl` (오프스크린) / `glfw` (창)

## 5. 빠른 검증

```bash
python manager/main.py --mode sim --no-view
# → "전체 80개 / 성공 46 / 실패 34" 비슷하게 나오면 환경 OK
```

이게 나오면 conda env + MuJoCo + numpy 가 정상.

## 6. 데모 녹화 (`skills.mv_pos`, `skills.approach_to_buffer` 용)

```bash
# Mac
mjpython manager/demos/record.py --name mv_pos
mjpython manager/demos/record.py --name approach_to_buffer

# Linux
python manager/demos/record.py --name mv_pos
```

조작·포맷 자세히: [manager/demos/README.md](manager/demos/README.md).

## 7. 처음 돌릴 때 유의사항

- `manager/demos/` 비어 있음 → 위 §6 으로 `mv_pos.npz`, `approach_to_buffer.npz` 녹화 ([todo.md §1](todo.md))
- 보정값 (`modules/transform/transform.py` 상단 상수, `modules/arm/skills.py` offset) 은 실측값으로 채워야 실물 동작 ([todo.md §2, §4](todo.md))
- 현재 sim 은 **옛 레거시 경로**(Sequencer + pick_and_place) 로 돈다. 새 ArUco 기반 `actions.packing/unpacking` 경로로의 전환 체크리스트는 [todo.md §7](todo.md)

## 참고 문서
- 파이프라인 전체 흐름: [`pipeline/pipeline.md`](pipeline/pipeline.md)
- 호출 순서: [`pipeline/file_call_flow.md`](pipeline/file_call_flow.md)
- 장치별 상세: [`pipeline/details_by_modules/`](pipeline/details_by_modules/)
- 구현 상태 도식: [`pipeline/pipeline.drawio`](pipeline/pipeline.drawio)
- 보정값·좌표계 설정: [`pipeline/settings/settings.md`](pipeline/settings/settings.md)
