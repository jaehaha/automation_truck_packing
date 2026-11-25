# 통합 프레임워크: 여러 장치를 API 한 번으로 움직이기

팔(Piper, CAN), 레일(아두이노), 흡착 밸브(아두이노), 카메라 2대를 **하나의 `Robot` 객체** 뒤에 두고,
위에서는 `robot.pick_and_place(...)` 같은 호출만 쓰게 만든다.

## 전체 파이프라인 (드라이브 `final/Pipeline` 슬라이드 기준)

```
[id1, id2, ..., idn]
        |
   Packing Model  ------------>  [[id1: goal pos], [id2: goal pos], ...]   (놓는 순서대로)
                                          |  앞에서부터 하나씩 (순차 탐색)
   상자 표                                v
   id: {box size, current position} -> Motion Planning -> 실행 -> 로봇
        ^                                                          |
        +---------- 완료 신호 시 goal pos 로 current pos 갱신 -----+
```

| 슬라이드의 블록 | 구현 |
|---|---|
| Packing Model | handoff 의 적재 정책. 상자 목록 전체를 한 번에 넣고 순서와 자리를 받는다 (`plan_moves.py --manifest` → `feed_order`, `moves[].to`) |
| 상자 표 | `BoxTable`: `id -> {size, current, goal, state}`. 프레임워크가 가진 **유일한 장면 정보**다 |
| Motion Planning | `sim_final` 의 `Stacker.plan`. 집는 곳 = 표의 `current`, 놓는 곳 = `goal`, 장애물 = 표에서 이미 놓인(`placed`) 상자들 |
| 실행 | `skills.pick_and_place(robot, plan)` |
| 완료 신호 → 갱신 | 실행이 예외 없이 끝나면 `table.done(id)`: `current = goal`, `state = placed` |

슬라이드에 없지만 정해야 하는 것:

- **`current position` 의 첫 값**: 아직 싣지 않은 상자의 위치는 카메라(ArUco)로 잰 값을 표에 넣는다 (`aruco.md`). 사람이 투입 위치에 하나씩 놓는 방식이면 "투입 위치 + 측정 오차"가 된다.
- **Motion Planning 이 실패한 상자**: 순서대로 가다가 한 상자의 경로가 안 나오면 그 뒤 자리도 전제가 깨진다. 그 상자의 자리를 금지 목록에 넣고, **남은 상자만으로 Packing Model 을 다시 불러** 뒤쪽 목록을 바꾼다 (이미 놓은 상자는 표에서 고정).
- **완료 신호의 근거**: 흡착을 끄고 물러난 것까지를 완료로 본다. 전체뷰 카메라가 있으면 놓인 위치를 재서 `current` 를 goal 대신 **잰 값**으로 넣을 수 있다 (오차가 다음 상자의 장애물 계산에 반영된다).
- **Plan B (컨베이어 + 앞단 로봇)**: 표의 `current` 가 컨베이어 위 위치로 바뀌고 앞단 로봇이 투입 위치로 옮기는 단계가 하나 더 생길 뿐, 이 고리는 그대로다.

## 1. 층 구조

```
작업      job.py        Packing Model 호출 -> 목록을 순서대로: 표 조회 -> 경로(Stacker) -> 실행 -> 표 갱신
          boxes.py      BoxTable (상자 표)
동작      skills.py     pick_and_place, measure_box, go_ready
로봇      robot.py      Robot: follow(경유점), suction(on), state()   <- API 는 여기까지만 노출
장치      arm.py / rail.py / suction.py / camera.py
백엔드    real (Piper SDK, 시리얼, OpenCV)  |  sim (sim_final 의 MuJoCo 모델)
```

위층은 아래층의 함수만 부른다. 작업 코드는 CAN 이나 시리얼을 직접 건드리지 않는다.

## 2. 핵심 API

```python
robot = Robot.real(can='can0', arduino='/dev/ttyACM0')     # 또는 Robot.sim()

robot.state()                      # {'rail': m, 'q': [6], 'suction': bool, 'holding': bool|None, 'fault': str|None}
robot.follow(waypoints, speed=0.3) # [[레일, 관절1..6], ...] 을 순서대로. 레일과 팔이 함께 도착
robot.suction(True)                # 켜고 붙을 때까지 대기. 실패하면 예외
robot.measure_box()                # 손목 카메라 -> (dx, dy, yaw). 마커가 안 보이면 예외
robot.stop()                       # 그 자리에서 정지. 흡착 상태는 그대로 둔다

# 동작 하나 = 위 호출의 정해진 조합
skills.pick_and_place(robot, plan) # approach -> suction on -> carry -> suction off -> retreat
```

작업 층 (슬라이드의 고리 그대로):

```python
table = BoxTable(manifest)                     # id -> {size, current, goal, state}
order = packing_model(table.ids())             # [(id, goal pos), ...] 놓는 순서
for box_id, goal in order:
    table.set_goal(box_id, goal)
    table.set_current(box_id, robot.measure_box())        # 카메라로 잰 현재 위치
    plan = planner.plan(table[box_id], obstacles=table.placed())
    if not plan.ok:
        order = packing_model(table.remaining(), forbid=goal)   # 남은 상자만 다시 계획
        continue
    skills.pick_and_place(robot, plan)
    table.done(box_id)                         # 완료 신호: current = goal, state = placed
```

경유점 형식은 `sim_final` 의 `Stacker` 가 내는 `[레일, 관절1..6]` 그대로 받는다 (`mp.md` 참고).

## 3. 장치 인터페이스

장치마다 같은 모양의 작은 클래스로 만들고, 실물용과 시뮬레이션용을 둘 다 둔다.

| 장치 | 메서드 | 실물 | 시뮬레이션 |
|---|---|---|---|
| `Arm` | `read() -> q`, `command(q)` | Piper SDK `JointCtrl` (지금 `piper_keyboard.py` 방식) | MuJoCo `ctrl[1:7]` |
| `Rail` | `read() -> m`, `command(m)`, `home()` | 아두이노 시리얼 (`linear.md`) | MuJoCo `ctrl[0]` |
| `Suction` | `on()`, `off()`, `holding()` | 아두이노 시리얼 (`vacuum.md`) | 상자를 공구에 붙였다 떼기 |
| `Camera` | `frame()`, `markers()` | OpenCV + ArUco (`aruco.md`) | MuJoCo 렌더 (`wrist`, `overview` 카메라가 모델에 있다) |

`Robot.real()` 과 `Robot.sim()` 은 어떤 장치 묶음을 쓰는지만 다르다.
같은 작업 코드를 시뮬레이션에서 먼저 돌려 보고 실물로 옮길 수 있다.

## 4. 팔과 레일을 함께 움직이기

`follow()` 가 프레임워크의 중심이다.

1. 경유점 사이를 7축 직선으로 잇고, 축마다 속도·가속도 한계를 넘지 않게 구간 시간을 정한다 (가장 느린 축에 맞춘다).
2. 제어 루프 하나(100 ~ 200 Hz)가 시간에 따라 보간한 목표를 **팔과 레일에 같은 주기로** 보낸다.
3. 실제 위치를 읽어 목표와 차이가 한계를 넘으면 진행을 멈춘다 (어느 한쪽이 못 따라오는 경우).

레일이 "목표 하나 보내고 끝날 때까지 대기" 방식이면 2번이 안 된다. 그때는 둘 중 하나로 한다.
- 아두이노 펌웨어를 목표를 계속 받아 따라가는 방식으로 바꾼다 (권장).
- 레일과 팔이 동시에 움직이는 구간을 "레일만 → 팔만" 으로 나눈 경로만 쓴다 (계획기에서 그런 경로만 허용).

## 5. 스레드와 통신

- **제어 스레드 1개**가 CAN 과 시리얼 포트를 혼자 쓴다. 주기적으로 상태를 읽고 목표를 보낸다.
- 아두이노 하나에 레일과 밸브가 같이 있으면 포트를 여는 객체는 하나만 두고 `Rail`, `Suction` 이 그것을 나눠 쓴다.
- API 호출(`follow`, `suction`)은 제어 스레드에 요청을 넘기고 끝날 때까지 기다린다. 한 번에 한 동작만 실행한다.
- 카메라는 별도 스레드에서 최신 프레임만 유지한다.
- 다른 프로그램이나 다른 PC 에서 부르려면 이 파이썬 API 위에 얇은 HTTP 서버를 나중에 얹는다. 먼저 한 프로세스 안에서 완성한다.

## 6. 오류와 안전

- 모든 API 는 실패하면 예외를 낸다. 동작(`skills`)은 예외가 나면 `robot.stop()` 만 하고 그대로 올린다.
- 정지할 때 흡착은 **끄지 않는다** (들고 있던 상자가 떨어진다). 끄는 것은 사람이 확인한 뒤 따로 한다.
- 관절·레일 한계와 속도 한계는 `Robot` 안에서 한 번 더 확인하고, 벗어나는 명령은 보내지 않는다.
- 시작할 때: 현재 자세를 읽어 목표로 삼는다 (튀지 않게), 레일 원점 확인, 통신 주기 확인.
- 모든 명령과 상태를 시간과 함께 파일로 남긴다. 실물에서 문제가 생기면 같은 기록을 시뮬레이션에 재생해 본다.
- `speed` 는 전체 속도 비율이다. 처음에는 0.1 로 돌린다.

## 7. 파일 구성

```
manipulator/
  manibox/
    devices/arm.py rail.py suction.py camera.py     # 실물
    sim/backend.py                                  # sim_final 모델로 같은 인터페이스
    robot.py                                        # Robot, follow(), 제어 스레드
    skills.py                                       # pick_and_place, measure_box
    boxes.py                                        # BoxTable: 상자 표 (크기, 현재 위치, 목표 위치, 상태)
    job.py                                          # Packing Model -> 순서대로 경로 계획 -> 실행 -> 표 갱신
    config.yaml                                     # 포트, 한계, 대기 시간, 카메라 보정값
  firmware/rail_vacuum.ino
```

## 8. 만드는 순서

1. `Robot.sim()` + `follow()`: `Stacker` 계획을 `sim_final` 모델에서 재생
2. `Arm` 실물: 같은 `follow()` 로 빈손, 레일 고정, 느리게
3. `Rail` 실물: 팔과 동시 이동, 추종 오차 확인
4. `Suction` 실물: 고정 위치에서 집고 놓기
5. `Camera`: 보정 → `measure_box()` 값 검증
6. `BoxTable` + `job`: 시뮬레이션에서 파이프라인 고리 전체 (Packing Model → 순차 계획 → 실행 → 표 갱신)
7. `skills.pick_and_place` → `job` 을 실물에 연결

각 단계는 앞 단계가 실물에서 확인된 뒤에 넘어간다.
