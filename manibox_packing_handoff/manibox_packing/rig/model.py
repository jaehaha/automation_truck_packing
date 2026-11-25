"""ManiBox R17 리그 MJCF: 알루미늄 프레임, 바닥판, 1 m 레일 위 캐리지, 거꾸로 단 Piper + 흡착 컵, 상자 칸.

월드 좌표 (m, pack_env 와 같다): 프레임 입구 쪽 바닥 모서리 = 원점, x = 1600 길이 방향 (입구 x = 0),
y = 560 폭 방향, z = 위. 바닥판 윗면 z = 0.043. 컨테이너 원점 C_ORIGIN = (1.52, 0.52, 0.043).

치수는 ManiBox_1000_Design R17 (프레임 1600 x 560 x 780, 기둥 620, 레일 이동 1000) 과 팀 메모에서 읽은 값이다.
레일 0 에서 팔 베이스 x = 0.2 m 로 가정했다 (이동 1000 mm 동안 0.2 ~ 1.2 m). 툴 길이는 실측해서 바꿔야 한다.
Piper 는 agilexrobotics/agx_arm_urdf (MIT, piper/LICENSE) 의 URDF·메시를 그대로 쓴다.

접촉 그룹: 프레임·바닥·상자 contype 1, 로봇 contype 2 (프레임·상자와는 닿고 로봇끼리는 안 닿는다).
팔은 운동학으로만 움직이고 MuJoCo 접촉은 충돌 검사로만 쓴다.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PIPER = HERE / 'piper'
ARM_JOINTS = tuple(f'joint{i}' for i in range(1, 7))
LINKS = ('base_link', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6')

FRAME = (1.600, .560, .780)
PROFILE = .040
FLOOR_TOP = .043
INNER_Y = (.040, .520)
BEAM_BOTTOM = .620
CROSS_BOTTOM = .700
MOUNT_Z = .672                  # 로봇 장착면 (용접 받침 아랫면). 로봇은 그 아래로 매달린다
BASE_X0 = .200                  # 레일 0 일 때 팔 베이스 x
TRAVEL = 1.000                  # 레일 이동
BASE_Y = .280
TOOL_LENGTH = .080              # 플랜지 -> 컵 면 (가정: 축 + 30 mm 주름 컵). 실제 툴을 재서 바꾼다
CUP_RADIUS = .015
PARK = (-.6, .0, -.5)           # 쓰지 않는 상자 칸: 바닥 아래, 접촉 없음


def _quat_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(y / 2), np.sin(y / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def _f(v):
    return ' '.join(f'{x:.6g}' for x in v)


def _box(name, lo, hi, rgba, contype=1, conaffinity=1, group=0):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return (f'<geom name="{name}" type="box" pos="{_f((lo + hi) / 2)}" size="{_f((hi - lo) / 2)}" rgba="{_f(rgba)}" '
            f'contype="{contype}" conaffinity="{conaffinity}" group="{group}"/>')


def _rig() -> list:
    L, W, H = FRAME
    P = PROFILE
    alu, dark = (.78, .80, .83, 1), (.30, .34, .37, 1)
    g = []
    for y0 in (0., W - P):
        g.append(_box(f'floor_long_{y0:.2f}', (0, y0, 0), (L, y0 + P, P), alu))
        g.append(_box(f'beam_{y0:.2f}', (0, y0, BEAM_BOTTOM), (L, y0 + P, H), alu))
    for x0 in (0., L - P):
        g.append(_box(f'floor_cross_{x0:.2f}', (x0, P, 0), (x0 + P, W - P, P), alu))
        g.append(_box(f'cross_{x0:.2f}', (x0, P, CROSS_BOTTOM), (x0 + P, W - P, H), alu))
        for y0 in (0., W - P):
            g.append(_box(f'pillar_{x0:.2f}_{y0:.2f}', (x0, y0, P), (x0 + P, y0 + P, BEAM_BOTTOM), alu))
    for x0 in (.53, 1.05):
        g.append(_box(f'floor_mid_{x0:.2f}', (x0, P, 0), (x0 + P, W - P, P), alu))
    g.append(_box('floor_board', (P, INNER_Y[0], P), (L - P, INNER_Y[1], FLOOR_TOP), (.55, .57, .60, 1)))
    g.append(_box('centre_support', (0, .223, .700), (L, .273, .740), alu))
    g.append(_box('actuator', (.13, .222, .748), (1.29, .274, .790), dark, 0, 0))
    return g


def _piper(tool_length: float) -> str:
    """공식 URDF 의 Piper 몸체 (운동학·관절 범위·STL 메시) + 흡착 컵."""
    urdf = ET.parse(PIPER / 'urdf/piper_description.urdf').getroot()
    joints = {j.find('parent').get('link'): j for j in urdf.findall('joint') if j.get('type') == 'revolute'}

    def body(link, pos='0 0 0', quat='1 0 0 0', joint=None):
        s = f'<body name="{link}" pos="{pos}" quat="{quat}"><inertial pos="0 0 0" mass=".3" diaginertia=".002 .002 .002"/>'
        if joint is not None:
            ax, lim = joint.find('axis').get('xyz'), joint.find('limit')
            s += f'<joint name="{joint.get("name")}" type="hinge" axis="{ax}" range="{lim.get("lower")} {lim.get("upper")}"/>'
        s += (f'<geom name="{link}_geom" type="mesh" mesh="{link}" rgba=".86 .87 .88 1" contype="2" conaffinity="1" '
              f'group="1" density="0"/>')
        if link == 'link6':
            s += (f'<geom name="cup_stem" type="cylinder" fromto="0 0 0 0 0 {tool_length - .03:.4f}" size=".008" '
                  f'rgba=".2 .2 .2 1" contype="2" conaffinity="1" group="1" density="0"/>')
            s += (f'<geom name="cup" type="cylinder" fromto="0 0 {tool_length - .03:.4f} 0 0 {tool_length:.4f}" '
                  f'size="{CUP_RADIUS}" rgba=".1 .1 .1 1" contype="2" conaffinity="1" group="1" density="0"/>')
            s += f'<site name="tcp" pos="0 0 {tool_length:.4f}" size=".006" rgba="1 0 0 1"/>'
        if link in joints:
            j = joints[link]
            o = j.find('origin')
            rpy = [float(v) for v in o.get('rpy').split()]
            s += body(j.find('child').get('link'), o.get('xyz'), _f(_quat_rpy(*rpy)), j)
        return s + '</body>'
    return body('base_link')


def build(n_boxes: int, tool_length: float = TOOL_LENGTH) -> str:
    """상자 칸 n_boxes 개 (장애물) + 들고 가는 상자 1개 (carried) 가 있는 MJCF 문자열. 상자 크기는 실행 중에 바꾼다."""
    meshes = ''.join(f'<mesh name="{n}" file="{PIPER / "meshes" / (n + ".stl")}"/>' for n in LINKS)
    cs = MOUNT_Z
    carriage = (
        f'<body name="carriage" pos="{BASE_X0} {BASE_Y} 0">'
        f'<joint name="rail" type="slide" axis="1 0 0" range="0 {TRAVEL}"/>'
        '<inertial pos="0 0 .75" mass="3" diaginertia=".05 .05 .05"/>'
        + _box('top_plate', (-.11, -.284, .810), (.13, .284, .822), (.88, .72, .41, 1), 0, 0)
        + _box('cradle_bottom', (-.11, -.152, cs), (.13, .152, cs + .012), (.63, .71, .76, 1), 0, 0)
        + f'<body name="mount" pos="0 0 {cs}" quat="0 1 0 0">{_piper(tool_length)}</body></body>')
    boxes = ''.join(
        f'<body name="box{i}" mocap="true" pos="{PARK[0] - .2 * (i % 20)} {PARK[1] + .2 * (i // 20)} {PARK[2]}">'
        f'<geom name="box{i}" type="box" size=".04 .04 .04" rgba=".7 .6 .45 1" contype="0" conaffinity="0"/></body>'
        for i in range(n_boxes))
    # 들고 가는 상자는 자유 몸체로 둔다 (정적 몸체끼리는 MuJoCo 가 접촉을 보지 않는다). 시뮬레이션은 돌리지 않는다
    carried = (f'<body name="carried" pos="{PARK[0]} {PARK[1] - .5} {PARK[2]}"><freejoint name="carried"/>'
               f'<geom name="carried" type="box" size=".04 .04 .04" mass=".3" rgba=".9 .5 .2 1" contype="0" conaffinity="0"/></body>')
    return f'''<mujoco model="manibox_r17">
<compiler angle="radian" meshdir="/" autolimits="true"/>
<option timestep=".002"/>
<size memory="64M"/>
<visual><global offwidth="1280" offheight="720"/></visual>
<asset>{meshes}</asset>
<worldbody>
<geom name="ground" type="plane" size="3 3 .1" rgba=".8 .8 .78 1" contype="1" conaffinity="1"/>
{''.join(_rig())}
{carriage}
{boxes}
{carried}
</worldbody>
</mujoco>'''
