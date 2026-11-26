"""컨테이너 좌표계 (Container Frame, CF) — 모든 측정·명령·좌표가 공유하는 단일 기준.

원점: 바깥 프레임 왼쪽-아래 모서리 (문 쪽 + 가까운 벽 쪽 + 바닥).
축:
  x: 문 → 정면 벽 (0 ~ 1600 mm)
  y: 가까운 벽 → 먼 벽 (0 ~ 560 mm)
  z: 바닥 → 위
단위: 전부 m (내부 계산). mm 가 필요한 하드웨어 인터페이스는 경계에서만 환산.

이 모듈이 CF 와 다른 좌표계를 이어 준다:
  CF  — 이 모듈이 정의
  MW  — MuJoCo world (modules/arm/model/manibox_modified.xml 안의 좌표)
  RC  — Robot Coords, stacking.py 의 ORIGIN(팔 베이스가 레일 0 일 때의 MW 좌표) 기준
  RM  — Rail mm, 아두이노 modules/rail_suction.ino 의 0 ~ 1000 mm
  CC  — Camera coords, OpenCV 가 반환하는 카메라 광학 중심 기준

TODO (사용자 보정 필요):
  CF_ORIGIN_IN_MW            — CF 원점이 MuJoCo world 어디에 있는지 (실측 or 모델 정비)
  RAIL_ZERO_CF_X             — 아두이노 레일 0 mm 가 CF x 어디에 있는지 (실측)
  RAIL_DIR                   — 레일 mm 증가 방향 (+1 이면 CF +x, -1 이면 CF -x)
  WRIST_CAM_EXTR_IN_EEF      — 손목 카메라 광학축의 EEF(TCP) 기준 pose (실측 or CAD)
  OVERVIEW_CAM_EXTR_IN_CF    — 전체뷰 카메라 광학축의 CF 기준 pose (ArUco 로 자동 보정 가능)
  ARUCO_BOX_SIZE_M           — 상자에 붙이는 ArUco 마커 한 변 길이 (실측)
  ARUCO_FIDUCIAL_IDS         — 프레임에 붙여 둔 고정 마커들 id → CF 위치 (사용자 설치 후)
"""
from __future__ import annotations

import numpy as np

# ===== 사용자 보정 필요 — 실측값으로 채운다 =====
# CF 원점의 MuJoCo world 좌표 (m).
# 바깥 프레임 왼쪽-아래 모서리가 MW 에서 어디에 있는지.
# MuJoCo 모델의 프레임 바닥 레일 pos("0.80 0.02 0.02") size("0.76 0.02 0.02") 로부터
# 추정: 바깥 프레임 외곽 = (0.04, 0.00, 0.00) + 모서리 폭 바깥.
# 임시로 0 으로 둠. 모델 정비 시 아래 값을 조정.
CF_ORIGIN_IN_MW: np.ndarray = np.array([0.000, 0.000, 0.000])

# 아두이노 레일 0 mm 가 CF 에서 어느 x (m) 에 있는지. 실측 필요.
RAIL_ZERO_CF_X: float = 0.040   # 가정: CF 왼쪽 프레임 안쪽면 (edge_profile = 40 mm)

# 레일 mm 증가 방향 (+1 이면 mm 올라갈수록 CF +x 로 이동)
RAIL_DIR: int = +1

# ArUco — 상자 윗면 마커 한 변 (m)
ARUCO_BOX_SIZE_M: float = 0.040

# ArUco — 프레임에 붙인 고정 기준 마커 (id -> CF 위치 m)
# 전체뷰 카메라 extrinsic 자동 보정에 쓴다.
# TODO: 실제 설치 위치를 넣는다. 최소 3개 이상, 가능하면 코너·중앙에 분산.
ARUCO_FIDUCIAL_IDS: dict[int, np.ndarray] = {}

# 손목 카메라 광학축의 EEF(TCP) 기준 자세 (4x4 동차 변환, 카메라→TCP).
# 모델의 wrist 카메라 pos(-0.0565 0 0.0735) + xyaxes 참고. 실물 캘리브레이션 결과로 교체.
WRIST_CAM_IN_EEF: np.ndarray = np.eye(4)   # placeholder — 실제 값으로 교체

# 전체뷰 카메라 extrinsic (카메라→CF). 매 프레임 ArUco 로 재추정 가능.
OVERVIEW_CAM_IN_CF: np.ndarray = np.eye(4)
# =============================================

# 모델 상수 — 순환 import 피하려고 복사해 둠.
# modules/arm/stacking.py :31 의 ORIGIN 값과 반드시 일치해야 한다.
_ORIGIN_MW: np.ndarray = np.array([0.700, 0.280, 0.672])


# -------- 변환 --------
def mw_to_cf(xyz):
    """MuJoCo world → Container Frame."""
    return np.asarray(xyz, float) - CF_ORIGIN_IN_MW


def cf_to_mw(xyz):
    """Container Frame → MuJoCo world."""
    return np.asarray(xyz, float) + CF_ORIGIN_IN_MW


def rc_to_cf(xyz):
    """Robot coords (packing 출력, stacking center 입력) → CF."""
    return mw_to_cf(np.asarray(xyz, float) + _ORIGIN_MW)


def cf_to_rc(xyz):
    """CF → Robot coords (stacking 에 넣을 수 있는 형태)."""
    return cf_to_mw(xyz) - _ORIGIN_MW


def rail_mm_to_cf_x(rail_mm: float) -> float:
    """Arduino 레일 mm → CF x (m)."""
    return RAIL_ZERO_CF_X + RAIL_DIR * rail_mm / 1000.0


def cf_x_to_rail_mm(cf_x: float) -> float:
    """CF x (m) → Arduino 레일 mm."""
    return (cf_x - RAIL_ZERO_CF_X) / RAIL_DIR * 1000.0


def cam_to_cf(xyz_cam: np.ndarray, cam_extr_in_cf: np.ndarray) -> np.ndarray:
    """카메라 광학축 기준 점 → CF. extr 는 카메라→CF 4x4 동차."""
    p = np.append(np.asarray(xyz_cam, float).ravel(), 1.0)
    return (cam_extr_in_cf @ p)[:3]


def eef_to_cf(xyz_eef: np.ndarray, eef_pose_in_cf: np.ndarray) -> np.ndarray:
    """EEF(TCP) 기준 점 → CF. eef_pose_in_cf 는 TCP→CF 4x4."""
    p = np.append(np.asarray(xyz_eef, float).ravel(), 1.0)
    return (eef_pose_in_cf @ p)[:3]
