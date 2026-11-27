0. main.py 실행

## 적재 알고리즘
1. 입력 시퀀스 유저 입력 완료 신호 (컨베이어와 연결 안돼있으니깐 수동)
2. 모델 출력 (출력물 위치는 manager/sequence/)
---
## 모션 플래닝 - Packing  (actions.packing(robot, box, table))

각 단계의 "호출 경로" 는 **manager/actions.py** → 아래 모듈 중 하나.

1. manager/actions.py 
    a. modules/arm/skills.py: approach_to_buffer() 호출
    b. modules/arm/skills.py: alignment(pack) 호출

2. manager/actions.py (→ manager/arduino.py ON 명령)
    a. modules/suction/suction.py: Suction.grip() 호출

3. manager/actions.py 
    a. modules/arm/skills.py: mv_pos() 호출

4. manager/actions.py (→ manager/arduino.py M<mm> 명령)
    a. modules/rail/rail.py: Rail.mv(x) 호출
        이 때, x는 모델이 출력한 박스 위치의 CF x (m) 를 말함.
        mv() 안에서 APPROACH_X_OFFSET 을 빼서 팔 베이스 뒤로 이동.
        CF x → 레일 명령값 변환은 각 백엔드 _cf_x_to_rail_m() 에 있음.

5. manager/actions.py 
    a. modules/arm/skills.py: approach_to_box(pack)
        approach_to_box 함수 안에는 pack, unpack으로 나뉘도록 구현.  
        각각 정해진 규칙대로 움직이도록 할 예정

6. manager/actions.py (→ manager/arduino.py OFF 명령)
    a. modules/suction/suction.py: Suction.un_grip() 호출

7. manager/actions.py
    a. modules/arm/skills.py: mv_pos() 호출

8. manager/actions.py (→ manager/arduino.py HOME/M0 명령)
    a. modules/rail/rail.py: Rail.mv(0) 호출
        0을 인자로 받으면 시작 위치로 돌아옴 (내부에서 home() 호출)

9. manager/actions.py
    a. manager/boxes.py: BoxTable.done(box.id) — current ← goal, state = placed

---
## 모션 플래닝 - UnPacking  (actions.unpacking(robot, box, table))

1. manager/actions.py (→ manager/arduino.py M<mm> 명령)
    a. modules/rail/rail.py: Rail.mv(x) 호출  (x = box.current CF x)

2. manager/actions.py 
    a. modules/arm/skills.py: approach_to_box(unpack) 호출
    b. modules/arm/skills.py: alignment(unpack) 호출

3. manager/actions.py (→ manager/arduino.py ON 명령)
    a. modules/suction/suction.py: Suction.grip() 호출

4. manager/actions.py 
    a. modules/arm/skills.py: mv_pos() 호출

5. manager/actions.py (→ manager/arduino.py HOME/M0 명령)
    a. modules/rail/rail.py: Rail.mv(0) 호출

6. manager/actions.py 
    a. modules/arm/skills.py: approach_to_buffer() 호출
    b. modules/arm/skills.py: put_to_buffer() 호출
        들고 있는 박스 정보를 BoxTable에서 받아올 수 있으니깐 높이에 맞는 위치로 이동

7. manager/actions.py (→ manager/arduino.py OFF 명령)
    a. modules/suction/suction.py: Suction.un_grip() 호출

8. manager/actions.py
    a. modules/arm/skills.py: mv_pos() 호출

9. manager/actions.py
    a. manager/boxes.py: BoxTable.remove(box.id) — 버퍼로 꺼낸 상자를 표에서 제거

---
## 참고

- Suction.grip()/un_grip() 는 밸브 ON/OFF 만 담당 (manager/arduino.py 를 통해 ON/OFF 문자열 전송).
  접촉까지의 arm 접근 motion (호버 → 상자 면 접촉) 은 **skills.alignment(mode)** 가 두 단계로 처리.
- Rail.mv(x_cf) 안에 CF↔레일 명령값 매핑 (실측 캘리브레이션) 이 들어감.
  real: cf_x_to_rail_mm (transform.py) → zero_offset_m 통해 RC m 로.
  sim : CF x - (_ORIGIN_MW[0] - CF_ORIGIN_IN_MW[0]).
- 사이클 어느 단계에서든 예외 발생 → actions 가 받아 올리면 상위 (main.py) 에서 robot.stop() + table.fail(id).
  흡착은 **끄지 않는다** (들고 있던 상자가 떨어진다).


## GUI FLOW
1. 메인화면에는 '실행', '개발자 모드'

### 실행 선택 시
1. 입력파일과 출력 파일 디렉토리, 추론 스킵 선택 화면
2-1. 다음 누르면 가운데 로딩 모양으로 추론 실행 중인 거 보여줌 (%로 진행상황 보여지게)
2-2. 스킵 누르면 추론 넘은 다음 3번 화면으로 넘어가짐
3. 적재 실행을 누르면 시뮬레이션 화면 켜지면서 적재시작 (여기서는 무조건 모드 선택 없이 로봇 동기화)

### 개발자 모드 선택 시
1. 선택화면에 알고리즘 개발, arm, suction, camera, rail 개발이 각각 있음

2-1. 알고리즘 개발 선택 시, 입력 파일 위치, 출력 파일 위치, 추론 스킵이 있음. 선택하고 추론을 하거나 추론 스킵을 해서 어쨌든 추론 과정이 끝나면 추론 완료라는 알림 창이 뜨도록 구현

2-2. arm 선택 시, demo 기록, 각 함수 호출 테스트 2가지 메뉴 존재. 각 함수는 팔의 움직임을 제어하는 함수들을 호출해서 실물과 sim에서 잘 돌아가는지 테스트하는 목적.

2-3. suction 선택 시, ON/OFF가 있어서 누르면 suction을 동작하도록 하는 main함수에서 호출하는 함수를 호출해서 동작하는지 테스트

2-4. camera 선택 시, 우선 wrist cam(lerobot so101 arm에 장착되는 wrist cam), overview cam(intel d435)의 RGB 상태를 양쪽에 출력

2-5. rail 선택 시, 기존 존재하는 rail() 함수를 호출해서 실제로 레일 모듈이 잘 동작하는지 테스트 용도