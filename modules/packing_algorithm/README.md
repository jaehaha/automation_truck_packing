# ManiBox 적재 위치 학습 환경

상자가 도착 순서대로 하나씩 들어올 때 **컨테이너 안 어디에, 어떤 방향으로 놓을지**를 학습하기 위한 MuJoCo 환경입니다.
로봇 팔은 없습니다. 팔의 경로는 별도의 모션 플래너가 만들고, 이 환경은 그 플래너가 요구하는 조건을 공간 규칙으로 검사합니다.
학습한 모델이 내야 하는 것은 상자별 **위치와 방향**뿐입니다.

## 설치와 실행

[uv](https://docs.astral.sh/uv/)가 필요합니다. 이 폴더에서 실행합니다.

```bash
uv sync                                        # 처음 한 번 (Python 3.12, mujoco, gymnasium, numpy)
uv run python pack_env.py --check              # Gymnasium 규격 검사
uv run python pack_env.py --episodes 5         # 기준 정책 통계
uv run python pack_env.py --render human       # 뷰어로 한 에피소드 보기
uv run python pack_env.py --save-json out.json # 배치 결과 저장
uv run python example_random.py                # 학습 루프 뼈대 (마스크를 쓴 무작위 정책)
```

화면이 없는 서버에서는 `MUJOCO_GL=egl`을 앞에 붙이면 `render_mode="rgb_array"`가 동작합니다. 학습 자체는 화면 없이 돌아갑니다.

## 적재 공간

컨테이너 좌표계, 단위 mm.

| 항목 | 값 |
|---|---|
| 원점 | 가장 안쪽 벽의 바닥 모서리 |
| x | 출구 방향 (0 = 안쪽 벽, 820 = 출구 쪽 끝) |
| y | 세로 (0 ~ 480) |
| z | 위 (0 ~ 480) |
| 격자 | 10 mm → 82 × 48칸 |

상자 5종 (가로 L × 세로 W × 높이 H, mm). 흡착면은 L × W 면입니다.

| 종류 | 치수 |
|---|---|
| 1 | 80 × 80 × 80 |
| 2 | 80 × 80 × 100 |
| 3 | 160 × 80 × 80 |
| 4 | 160 × 80 × 100 |
| 5 | 160 × 160 × 100 |

## 규칙

`action_masks()`가 아래 규칙을 지키는 자리만 알려 줍니다. 정책은 그 안에서만 고르면 됩니다.

| 규칙 | 내용 |
|---|---|
| 방향 | 흡착면(L × W)이 출구(+x)를 본다. 즉 상자 높이 H가 x 방향이다. 고를 수 있는 것은 그 면 안의 0°/90° 회전 2가지 |
| 틈 | 상자를 5 mm 키워서 격자에 놓는다 (상자 사이 최소 5 mm) |
| 받침 | 바닥 면적의 80 % 이상이 받쳐져야 한다 |
| 진입 통로 | 놓는 순간 상자 앞(출구 쪽)이 상자 바닥 높이보다 낮아야 한다 |
| 높이 | 상자 윗면이 480 mm 이하 |
| 안정성 | 고른 자리에 상자를 MuJoCo로 놓아 보고, 3 mm 넘게 움직이거나 2° 넘게 기울면 실패 |

하차 순서는 금지가 아니라 벌점입니다. 먼저 내릴 상자(지점 번호가 작은 상자)의 앞이나 위에 나중에 내릴 상자를 두면 위반 1건으로 셉니다.

## 환경 규격

```python
import gymnasium as gym
import pack_env                       # "ManiBoxPack-v0" 등록

env = gym.make("ManiBoxPack-v0")      # 또는 pack_env.PackEnv(pack_env.Config(...))
obs, info = env.reset(seed=0)
mask = info["action_mask"]            # 둘 수 있는 행동만 True
obs, reward, terminated, truncated, info = env.step(action)
```

**관측** (모두 0 ~ 1로 정규화)

| 키 | 모양 | 내용 |
|---|---|---|
| `heightmap` | (82, 48) | 칸마다 지금까지 쌓인 높이 / 480 |
| `box` | (9,) | 지금 상자: 종류 원핫 5 + (L, W, H) / 160 + 하차 지점 / 지점 수 |
| `next` | (5, 9) | 다음 5개 상자 (없으면 0) |
| `progress` | (2,) | 진행률, 부피 사용률 |

**행동** `Discrete(2 × 82 × 48)`

```
action = o × (82 × 48) + i × 48 + j
o  방향. 0: 놓은 뒤 (x, y, z) 치수 = (H, L, W) / 1: (H, W, L)
i  x 칸 (0 = 안쪽 벽),  j  y 칸
```

상자는 고른 칸에서 시작하는 영역의 가운데에 놓이고, 높이는 아래 받침으로 정해집니다. `env.unwrapped.decode(action)`이 (o, i, j)를 돌려줍니다.

**보상** `10 × (상자 부피 / 공간 부피) − 0.2 × (이번 상자의 하차 순서 위반 수)`. 가중치는 `Config`의 `w_util`, `w_block`, `w_fail`입니다.

**종료** `info["reason"]`

| 값 | 뜻 |
|---|---|
| `done` | 모든 상자를 놓았다 |
| `no_space` | 다음 상자를 둘 자리가 없다 |
| `unstable` | 놓은 상자가 넘어지거나 밀렸다 (보상 −1) |
| `invalid_action` | 마스크 밖의 행동을 골랐다 (보상 −1) |

**에피소드** 하차 지점 6 ~ 12곳, 지점당 상자 3 ~ 10개를 무작위로 만듭니다. 도착 순서는 나중에 내릴 지점부터입니다(`order="sorted"`). 지점 1을 가장 먼저 내립니다. 정해진 목록으로 시작하려면 `env.reset(options={"job": [(지점, 종류), ...]})`을 씁니다.

## 설정 (`Config`)

| 항목 | 기본값 | 뜻 |
|---|---|---|
| `grid` | 10 | 격자 한 칸 (mm) |
| `gap` | 5 | 상자 사이 최소 틈 (mm) |
| `support_min` | 0.8 | 받침 비율 하한 |
| `shares` | None | 종류별 도착 비율 (None이면 균등). 예: `{1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}` |
| `stops`, `per_stop` | (6, 12), (3, 10) | 지점 수, 지점당 상자 수 범위 |
| `order` | "sorted" | "random"이면 도착 순서를 섞는다 |
| `lookahead` | 5 | 미리 보여 주는 상자 수 |
| `physics` | True | False면 안정성 물리 검사를 끈다 (더 빠름) |

## 결과 형식

`env.unwrapped.export()`가 놓은 상자 목록을 돌려줍니다. `--save-json`은 여기에 로봇 좌표를 더해 저장합니다.

```json
{"id": "b001", "type": 5, "stop": 9,
 "center": [55.0, 85.0, 80.0],        // 상자 중심, 컨테이너 좌표 mm
 "dims_xyz": [100, 160, 160],         // 놓은 뒤 x, y, z 방향 치수 mm (방향을 나타낸다)
 "robot_center_m": [...], "robot_size_xyz_m": [...]}   // 모션 플래너 입력
```

목록의 순서는 도착 순서와 같습니다.

## 기준선

`pack_env.greedy`는 안쪽 → 낮은 곳 → y가 작은 쪽 순으로 첫 자리를 고릅니다. seed 0 ~ 3에서 부피 사용률 41 ~ 59 %, 하차 순서 위반 0건, 상자 하나에 약 5 ms입니다. 학습한 정책은 이 값과 비교하면 됩니다.

## 버퍼 적재 (버퍼는 문 쪽 2칸만, 나머지 바닥은 적재, 학습 정책, 로봇 이동 출력)

[README_buffer.md](README_buffer.md) 참고. 이 문서의 환경(`pack_env.py`)은 그대로 두고 그 위에 더했다.
- 버퍼 최소화: 문 쪽 줄 2칸(투입 칸 + 재배치 1칸)만 버퍼, 나머지 바닥 적재 (`stable`, 큰 상자 `big`)
- 적재 목록 → 투입 순서 + 전체 이동 (오프라인, `plan_moves.py --manifest`)
- 버퍼 구역까지 채우기 (`zone`)
- 무작위 도착 결과
- R17 + Piper 모델로 맞춘 모션 검사 (`rig/`, `check_plan.py`)
