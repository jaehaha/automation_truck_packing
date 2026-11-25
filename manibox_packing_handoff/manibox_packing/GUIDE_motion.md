# ManiBox 상자 적재: 모션 플래닝 전달 설명서

적재 목록(상자 id, 배송 지점, 종류)을 넣으면 **투입 순서**와 **로봇 이동 목록**이 나온다. 이동마다 시작점, 끝점, 밀어 넣기 전 대기점이 있다.
모션 플래닝은 이 점들 사이 경로만 만들면 된다. 이동 순서가 곧 실행 순서다.

## 1. 실행

```bash
uv sync                                                    # 처음 한 번 (Python 3.12, mujoco, numpy, gymnasium)
uv run python plan_moves.py --mode big --manifest examples/minbuf/manifest_big_104.json --motion --out moves.json
#   -> moves.json (전체) + moves.csv (한 줄 = 이동 하나). 약 1분
uv run python check_plan.py moves.json                     # 이 저장소 로봇 모델로 모든 이동을 처음부터 다시 검사
MUJOCO_GL=egl uv run --with "imageio[ffmpeg]" python plan_moves.py --mode big --manifest 목록.json --motion --video load.mp4
```

- `--mode big`: 큰 상자 5종(`policy_big_minbuf.npz`). `--mode stable`: pack_env 기본 상자 5종(`policy_minbuf.npz`).
- 적재 목록은 JSON `{"boxes": [{"id": "P001", "stop": 4, "type": 1}, ...]}` 또는 CSV(`id,stop,type` 또는 `id,stop,L,W,H`)다.
  - 지점 1이 먼저 내린다.
  - 목록 순서는 상관없다. 투입 순서는 프로그램이 정한다.

## 2. 공간과 좌표

| 항목 | 값 |
|---|---|
| 리그 안쪽 | 깊이 1200 × 너비 480 × 높이 568 mm |
| 버퍼 | 문 쪽 줄 2칸뿐이다: 투입 칸 `arrival`, 재배치 칸 `buffer1`. 칸 안쪽 170 × 170, 칸막이 10 mm(바닥 표시로 가정) |
| 적재 공간 | 두 칸을 뺀 바닥 전체(깊이 약 1020). 칸막이는 바닥 표시라 그 위에도 걸쳐 놓일 수 있다 |
| 칸 중심 (로봇 좌표 m) | `arrival` (-0.285, 0.090), `buffer1` (-0.285, -0.090). 바닥 z = -0.629 |
| **출력 좌표** | 로봇 좌표(m) = 월드 - `ROBOT_ORIGIN` (0.700, 0.280, 0.672). 축은 월드와 같다 |
| `ROBOT_ORIGIN` 주의 | pack_env 값이다. 팀 정의로는 "레일 0일 때 팔 베이스"다. `rig/` 모델은 레일 0에서 베이스 x = 0.2 m로 가정했으므로, 이 점은 레일 500 mm 위치에 해당한다. `ik`의 `base_x_m`은 월드 x다. 실제 값을 맞춰야 한다(5절) |
| 컨테이너 좌표 (mm, `container_mm`) | 원점 = 안쪽 벽 바닥 모서리. x = 출구 방향 = 월드 -x, y = 월드 -y, z = 위 |

## 3. 이동 하나 (`moves`의 한 항목)

| 순서 | 로봇 동작 | 필드 |
|---|---|---|
| 1 | 투입 칸 상자의 윗면을 흡착한다 | `from.grasp_m`, `from.grasp_normal` = (0, 0, 1) |
| 2 | 들어 올려, 흡착면이 출구(월드 -x)를 보게 돌린다 | `to.quat_wxyz` = 놓인 뒤 상자 자세 |
| 3 | 대기점으로 간다: 끝점에서 출구 쪽 40 mm, 끝점과 같은 높이 | `to.pre_insert_m` |
| 4 | +x로 40 mm 밀어 넣고 놓는다 | `to.grasp_m` = 흡착면 중심, `to.grasp_normal` = (-1, 0, 0) |

- `kind`
  - `place`: 컨테이너에 넣는다.
  - `stage`: 투입 칸 → `buffer1`. 위에서 집어 위에서 내려놓고, 자세는 그대로다. 104개 예시에서는 1번이다.
- `scene_before`: 이 이동 직전 장면이다. `placed` = `placements` 앞에서부터 실린 상자 수이고, `slots` = 칸마다 있는 상자다. 장애물을 만들 때 쓴다.
- `ik`: 참고 해다(이 저장소 모델 기준). `pick` / `pre` / `insert` 자세마다 레일 위치와 관절 6개(rad)가 있다. `pre`와 `insert`는 관절이 같고 레일만 40 mm 다르다.
- `kept`: 마지막 상자들은 칸에 둔 채로 싣는다. 로봇 이동이 없다.
- `box_id`: 적재 목록의 id다. `feed_order`는 투입 칸에 넣을 순서다. `not_loaded`는 싣지 못한 상자와 그 이유다.

## 4. 미리 확인한 것과 모션 담당 몫

계획 단계에서 이미 거른 것:
- 집기, 대기점, 밀어 넣기 세 자세의 IK와 충돌. 팔과 들고 가는 상자를 프레임, 바닥, 실린 상자, 칸 상자에 대해 본다.
- 실린 상자가 팔꿈치나 아래팔에 닿는 자리. 팔 공간 표(`rig/arm_front.npz`)로 뺐다.
- 상자 안정성
  - 바닥의 90% 이상이 받쳐질 때만 놓고, 놓을 때마다 물리 시뮬레이션으로 확인한다.
  - 위치 ±6 mm, 치수 ±2 mm 오차와 마찰 0.35 조건으로 다시 쌓아도 넘어짐은 0이다.

모션 담당이 할 것:
- **자세 사이 경로**: 특히 집은 뒤 돌리기다. 이 팔 모델은 제자리에서 툴을 돌리면 관절이 이어지지 않는다. 빈 공간으로 들어 옮기며 돌려야 한다.
- **실제 로봇 모델로 다시 판단하기**: `rig/` 모델과 `ik`는 참고용이다.
- **못 푸는 이동이 있을 때**
  - 실시간으로 쓰면 `p.reject()`로 다음 순위를 받는다(아래).
  - 오프라인 계획이면 이동 번호를 알려 주면 그 자리를 빼고 다시 계획한다.

```python
from buffer_planner import OnlinePlanner                     # 실시간: 상자가 들어올 때마다
from buffer_core import BufferConfig
from plan_moves import BIG_TYPES
cfg = BufferConfig(slot_clear=True, reach='rig/reach_front.npz', arm='rig/arm_front.npz', zone=True,
                   stage_order=(1,), keep_stage=True, box_types=BIG_TYPES)   # 기본 상자면 cfg 생략 + 'policy_minbuf.npz'
p = OnlinePlanner('policy_big_minbuf.npz', cfg=cfg, motion=True)
p.add([(6, 5), (6, 3)])      # 투입 칸 상자 + 보이는 다음 상자 (지점, 종류)
mv = p.next_move()           # 이동 하나 (위 형식). 경로를 못 찾으면 mv = p.reject()
p.commit()                   # 실행 끝. 새 상자는 p.add(...), 더 없으면 p.close()
```

## 5. 같이 정해야 할 가정

| 항목 | 지금 가정 | 바뀌면 |
|---|---|---|
| 레일 0 위치 | 베이스 월드 x 0.2 m (`rig/model.py` `BASE_X0`). pack_env `ROBOT_ORIGIN` 정의로는 0.7 m | 값을 고치고 표 2개를 다시 만든다: `uv run python -m rig.reach_table --tool 0.08`, `uv run python -m rig.arm_table` |
| 툴 길이 | 80 mm (`rig/model.py` `TOOL_LENGTH`) | 위와 같다 (`--tool`에 새 길이, m) |
| 칸막이 | 바닥 표시 (높이 없음) | 높은 벽이면 알려 달라 |
| 밀어 넣는 높이 | 끝점과 같은 높이라서 아래 상자 위를 40 mm 미끄러진다 | 1~2 mm 띄워 밀고 내려놓기를 권한다 (끌림은 시험하지 않았다) |
| 도착 순서 | 나중에 내릴 지점부터 들어온다 | 무작위로 오면 하차 때 다시 옮길 상자가 생긴다 |

## 6. 성능 (64개 작업, 물리 시뮬레이션과 팔 검사 켬)

| | 들어오는 대로 | 적재 목록으로 미리 계획 |
|---|---|---|
| 기본 상자 | 1200 전체 부피의 58.4% (158개) | 61.4% (165개) |
| 큰 상자 | 56.0% (104개) | 60.7% (113개) |

- 아래는 모두 0이다: 팔 걸림, 하차 순서 위반, 하차 재작업, 오차 시험 넘어짐.
- 예시 `examples/minbuf/`
  - 큰 상자 104개를 103번 이동으로 모두 싣는다. 이동은 모두 검사를 통과한다.
  - 영상은 `minbuf_big_104_load_unload.mp4`(적재 다음 하차)다.

## 7. 파일

| 파일 | 내용 |
|---|---|
| `plan_moves.py` | 계획 → `moves.json` / `.csv` |
| `check_plan.py` | 계획 파일 모션 재검사 (`--flip`이면 돌리기 자세도) |
| `buffer_planner.py` | `OnlinePlanner` (실시간) |
| `buffer_core.py`, `buffer_env.py`, `buffer_policy.py`, `pack_env.py`, `stack_twin.py` | 규칙, 시뮬레이션, 정책 실행 |
| `policy_minbuf.npz`, `policy_big_minbuf.npz` | 학습된 정책 (기본 상자 / 큰 상자) |
| `rig/` | R17 + Piper 모델(Piper URDF·메시 MIT), 모션 검사, 도달 표, 팔 공간 표 |
| `verify_stack.py`, `unload_video.py` | 실물 오차 시험, 하차 영상 |
| `examples/minbuf/` | 적재 목록 → 이동 목록 JSON·CSV, 영상 |
| `README_buffer.md` | 전체 기록 (실험, 수치, 규칙) |

이 묶음에는 기본 모드(`stable`, `big`)의 정책과 예시만 넣었다. `README_buffer.md`에 나오는 다른 모드의 정책, 예시, 학습 기록(`runs/`)은 원래 폴더에 있다.
