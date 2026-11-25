"""모델 XML 을 뷰어로 열어두고, 파일이 바뀌면 자동으로 다시 불러온다.

  mjpython view_model.py                                  # 기본: mp/model/manibox_modified.xml
  mjpython view_model.py model/manibox_modified.xml       # 다른 파일

XML 을 저장할 때마다 뷰어가 새 모델로 바뀐다. 문법 오류가 있으면 콘솔에만 찍히고
기존 모델은 그대로 둔다.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer


def load(path):
    try:
        m = mujoco.MjModel.from_xml_path(str(path))
        return m, mujoco.MjData(m), None
    except Exception as e:
        return None, None, str(e)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "model" / "manibox_modified.xml"
    if not path.exists():
        raise SystemExit(f"파일 없음: {path}")

    print(f"뷰어 시작: {path}")
    print("XML 을 저장하면 자동으로 다시 불러온다. 창을 닫으면 끝난다.")

    while True:
        m, d, err = load(path)
        if m is None:
            print(f"XML 오류: {err}\n파일을 고치면 다시 시도한다.")
            while True:
                time.sleep(0.5)
                m, d, err = load(path)
                if m is not None:
                    break
                time.sleep(1.5)
        mtime = path.stat().st_mtime
        print(f"로딩됨 ({time.strftime('%H:%M:%S')})")
        with mujoco.viewer.launch_passive(m, d) as v:
            while v.is_running():
                time.sleep(0.1)
                try:
                    cur = path.stat().st_mtime
                except OSError:
                    continue
                if cur != mtime:
                    time.sleep(0.2)                  # 저장이 끝나기를 기다린다
                    print("파일이 바뀜 → 다시 불러옴")
                    break
            if not v.is_running():
                print("창을 닫음")
                return


if __name__ == "__main__":
    main()
