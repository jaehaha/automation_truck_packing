"""하차 영상: 계획 파일(plan_moves.py 결과) 의 적재를 지점 1부터 하차하는 모습.

지점마다 그 지점 상자 중 앞(문 쪽)과 위가 빈 것을 문 쪽으로 빼낸다. 다른 지점 상자가 앞이나 위를 막고 있으면
그 상자를 위로 들어 올려 잠시 치우고 (재작업), 그 지점이 끝나면 제자리로 돌린다. 규칙은 buffer_core.unload_plan
(= eval 의 하차 재작업 지표) 과 같다. 색 = 지점 (1 빨강, 2 주황, 3 노랑, 4 초록, 5 파랑, 6 보라).

  uv run --with "imageio[ffmpeg]" --with pillow python unload_video.py moves.json --out unload.mp4
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from pack_env import C_ORIGIN, STOP_COLORS, to_world  # noqa: E402
from buffer_core import unload_plan  # noqa: E402
from buffer_env import RAINBOW  # noqa: E402

COLOR_NAMES = ('red', 'orange', 'yellow', 'green', 'blue', 'purple')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def scene_xml(placements, depth_mm, colors):
    D = depth_mm / 1000
    ox, oy, oz = C_ORIGIN
    W, H = .48, .568
    g = [f'<geom type="plane" size="3 3 .1" material="grid" contype="0" conaffinity="0"/>',
         f'<geom type="box" pos="{ox - D / 2} {oy - W / 2} {oz - .0015}" size="{D / 2} {W / 2} .0015" rgba=".55 .56 .58 1"/>',
         f'<geom type="box" pos="{ox + .0015} {oy - W / 2} {oz + H / 2}" size=".0015 {W / 2} {H / 2}" rgba=".55 .56 .58 .4"/>',
         f'<geom type="box" pos="{ox - D / 2} {oy + .0015} {oz + H / 2}" size="{D / 2} .0015 {H / 2}" rgba=".55 .56 .58 .2"/>',
         f'<geom type="box" pos="{ox - D / 2} {oy - W - .0015} {oz + H / 2}" size="{D / 2} .0015 {H / 2}" rgba=".55 .56 .58 .2"/>']
    if depth_mm < 1200:                              # 버퍼 구역 (흰 판)
        g.append(f'<geom type="box" pos="{ox - (D + 1.2) / 2} {oy - W / 2} {oz - .0015}" size="{(1.2 - D) / 2} {W / 2} .0015" '
                 f'rgba=".95 .95 .93 1"/>')
    b = []
    for n, p in enumerate(placements):
        c, s = to_world(p['center'], p['dims_xyz'])
        b.append(f'<body name="b{n}" mocap="true" pos="{c[0]} {c[1]} {c[2]}"><geom type="box" size="{s[0] / 2 - .0005} '
                 f'{s[1] / 2 - .0005} {s[2] / 2 - .0005}" rgba="{colors[n][0]} {colors[n][1]} {colors[n][2]} 1" contype="0" conaffinity="0"/></body>')
    return f'''<mujoco><visual><global offwidth="1280" offheight="720"/></visual>
<asset><texture name="grid" type="2d" builtin="checker" rgb1=".85 .85 .85" rgb2=".75 .75 .75" width="300" height="300"/>
<material name="grid" texture="grid" texrepeat="8 8"/></asset>
<worldbody><light pos="{ox - D / 2} {oy - W / 2} 2.5" dir="0 0 -1" directional="true"/>{"".join(g)}{"".join(b)}</worldbody></mujoco>'''


class Video:
    def __init__(self, plan, out, fps=24, frames_per_step=6, title=''):
        import imageio
        from PIL import ImageFont
        self.P = [p for p in plan['placements']]
        self.n_stop = max(p['stop'] for p in self.P)
        pal = RAINBOW if self.n_stop <= len(RAINBOW) else [c[:3] for c in STOP_COLORS]
        self.colors = [pal[(p['stop'] - 1) % len(pal)] for p in self.P]
        depth = max(830, max(p['center'][0] + p['dims_xyz'][0] / 2 for p in self.P))
        self.m = mujoco.MjModel.from_xml_string(scene_xml(self.P, 1200 if depth > 831 else 830, self.colors))
        self.d = mujoco.MjData(self.m)
        self.home = np.array([to_world(p['center'], p['dims_xyz'])[0] for p in self.P])
        self.mid = [self.m.body_mocapid[self.m.body(f'b{n}').id] for n in range(len(self.P))]
        mujoco.mj_forward(self.m, self.d)
        self.r = mujoco.Renderer(self.m, 720, 1280)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:], self.cam.distance, self.cam.azimuth, self.cam.elevation = (1.0, .28, .25), 2.05, 22.0, -28.0
        self.w = imageio.get_writer(out, fps=fps, quality=8, macro_block_size=8)
        self.fps, self.k = fps, frames_per_step
        self.font = ImageFont.truetype(FONT, 30)
        self.small = ImageFont.truetype(FONT, 22)
        self.title = title
        self.caption = ''

    def frame(self, n=1):
        from PIL import Image, ImageDraw
        self.r.update_scene(self.d, self.cam)
        im = Image.fromarray(self.r.render())
        dr = ImageDraw.Draw(im)
        dr.rectangle((0, 0, 1280, 92), fill=(255, 255, 255))
        dr.text((18, 10), self.title, fill=(20, 20, 20), font=self.font)
        dr.text((18, 52), self.caption, fill=(40, 40, 40), font=self.small)
        for s in range(self.n_stop):                  # 색 범례
            c = tuple(int(255 * v) for v in (RAINBOW[s] if self.n_stop <= len(RAINBOW) else STOP_COLORS[s][:3]))
            x = 1280 - 60 * (self.n_stop - s) - 10
            dr.rectangle((x, 104, x + 44, 134), fill=c)
            dr.text((x + 14, 136), str(s + 1), fill=(30, 30, 30), font=self.small)
        a = np.asarray(im)
        for _ in range(n):
            self.w.append_data(a)

    def move(self, ids, targets, k=None):
        k = k or self.k
        start = {i: self.d.mocap_pos[self.mid[i]].copy() for i in ids}
        for t in np.linspace(0, 1, k + 1)[1:]:
            for i in ids:
                self.d.mocap_pos[self.mid[i]] = start[i] + t * (targets[i] - start[i])
            mujoco.mj_forward(self.m, self.d)
            self.frame()

    def close(self):
        self.w.close()
        self.r.close()


def render(plan, out, title='', fps=24, frames_per_step=6):
    P = plan['placements']
    placed = [dict(stop=p['stop'], lo=np.array(p['center']) - np.array(p['dims_xyz']) / 2,
                   hi=np.array(p['center']) + np.array(p['dims_xyz']) / 2) for p in P]
    steps = unload_plan(placed)
    v = Video(plan, out, fps, frames_per_step, title)
    total = len(P)
    moved_ever, moved_count, done = set(), 0, 0
    v.caption = f'{total} boxes loaded. Unloading starts at the door (stop 1 first).'
    v.frame(fps * 2)
    lift = np.array([0, 0, .32])
    out_dir = np.array([-.35, 0, 0])                 # 문 쪽 = 월드 -x
    cur = None
    stop_moved = 0
    for s, kind, ids in steps:
        if s != cur:
            cur, stop_moved = s, 0
            name = COLOR_NAMES[s - 1] if v.n_stop <= len(COLOR_NAMES) else ''
            v.caption = f'Stop {s} ({name}): take out its {sum(p["stop"] == s for p in P)} boxes'
            v.frame(fps // 2)
        if kind == 'out':
            v.move(ids, {i: v.home[i] + out_dir for i in ids})
            for i in ids:
                v.d.mocap_pos[v.mid[i]] = (-5, -5, -5)   # 하차 끝
            done += len(ids)
        elif kind == 'aside':
            stop_moved += len(ids)
            moved_count += len(ids)
            moved_ever |= set(ids)
            v.caption = (f'Stop {s}: blocked -> lift {stop_moved} box(es) of later stops out of the way   '
                         f'(re-handled so far: {moved_count})')
            v.move(ids, {i: v.home[i] + lift for i in ids})
            v.frame(fps // 4)
        else:
            v.caption = f'Stop {s} done: put back {len(ids)} box(es)   (re-handled so far: {moved_count})'
            v.move(ids, {i: v.home[i] for i in ids})
        mujoco.mj_forward(v.m, v.d)
    v.caption = (f'Done. Boxes that had to be moved at least once: {len(moved_ever)} of {total} '
                 f'({len(moved_ever) / max(1, total) * 100:.0f}%), re-handling moves: {moved_count}')
    v.frame(fps * 3)
    v.close()
    return dict(boxes=total, rework_boxes=len(moved_ever), rework=moved_count)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('plan')
    ap.add_argument('--out', default='unload.mp4')
    ap.add_argument('--title', default='')
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--frames-per-step', type=int, default=6)
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    print(render(plan, a.out, a.title, a.fps, a.frames_per_step), '->', a.out)


if __name__ == '__main__':
    main()
