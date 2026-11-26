"""카메라 (손목 카메라 + 전체뷰 카메라) 장치 인터페이스.

frame()    -> 지금 프레임 (H×W×3 uint8, BGR)
markers()  -> 검출된 ArUco 마커 목록 [(id, xyz_cf, yaw_cf)] — 전부 Container Frame 좌표계로 통일

RealCamera 는 OpenCV + ArUco 로 검출하고 `modules/transform/transform.py` 를 통해 CF 로 변환.
  - 손목 카메라: TCP(EEF) 기준으로 측정 → EEF pose + WRIST_CAM_IN_EEF → CF
  - 전체뷰 카메라: 직접 CF 에 설치 → OVERVIEW_CAM_IN_CF (ArUco fiducial 로 재추정 가능)
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "transform"))
import transform as cf                                    # noqa: E402


class Camera(ABC):
    @abstractmethod
    def frame(self) -> np.ndarray: ...
    @abstractmethod
    def markers(self) -> list[tuple[int, np.ndarray, float]]: ...


class SimCamera(Camera):
    """MuJoCo 모델 안의 카메라 (wrist, overview 등) 로 렌더링."""

    def __init__(self, model, data, cam_name: str = "wrist", width: int = 640, height: int = 480):
        import mujoco
        self.m, self.d, self.cam_name = model, data, cam_name
        self._renderer = mujoco.Renderer(model, width=width, height=height)

    def frame(self):
        import mujoco
        mujoco.mj_forward(self.m, self.d)
        self._renderer.update_scene(self.d, camera=self.cam_name)
        return self._renderer.render()

    def markers(self):
        # Sim 에서는 상자의 ground-truth 위치를 CF 로 돌려주는 식도 가능.
        # 지금은 빈 리스트 (더미). actions.pick_and_place 가 sim 에서는 이 메서드를 안 쓴다.
        return []


class RealCamera(Camera):
    """실물: OpenCV 로 USB 캡처 + ArUco 검출 → Container Frame 좌표.

    role:
      'wrist'    — 손목 카메라. markers() 는 **TCP 기준** 포즈를 모은 뒤 호출자가 현재 EEF pose 를
                   추가로 곱해 CF 로 변환. 또는 pose_in_cf 를 생성 시에 주입.
      'overview' — 전체뷰 카메라. ArUco fiducial 로 자가 보정 또는 고정 extrinsic 사용.
    """

    def __init__(self, device: int = 0, role: str = "overview",
                 camera_matrix: np.ndarray | None = None,
                 dist_coeffs: np.ndarray | None = None,
                 extrinsics_cam_to_cf: np.ndarray | None = None,
                 aruco_dict_name: str = "DICT_4X4_50",
                 marker_size_m: float | None = None):
        import cv2
        import cv2.aruco as aruco

        self.cv2 = cv2
        self.role = role
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 {device} 를 열 수 없음")

        if camera_matrix is None:
            raise ValueError("camera_matrix (내부 파라미터, 3x3) 필요. "
                             "OpenCV 체커보드 캘리브레이션으로 얻는다.")
        self.K = np.asarray(camera_matrix, float)
        self.dist = np.asarray(dist_coeffs if dist_coeffs is not None else np.zeros(5), float)
        self.extr = extrinsics_cam_to_cf if extrinsics_cam_to_cf is not None else cf.OVERVIEW_CAM_IN_CF
        self.marker_size = marker_size_m if marker_size_m is not None else cf.ARUCO_BOX_SIZE_M
        self._dict = aruco.getPredefinedDictionary(getattr(aruco, aruco_dict_name))
        self._params = aruco.DetectorParameters()

    def frame(self):
        ok, img = self.cap.read()
        if not ok:
            raise RuntimeError("카메라에서 프레임을 못 가져옴")
        return img

    def markers(self):
        """ArUco 검출 → [(id, xyz_cf, yaw_cf)]. 못 보면 빈 리스트."""
        import cv2
        import cv2.aruco as aruco
        img = self.frame()
        corners, ids, _ = aruco.detectMarkers(img, self._dict, parameters=self._params)
        if ids is None or len(ids) == 0:
            return []
        # 각 마커의 카메라 좌표 pose
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, self.K, self.dist)
        out = []
        for i, mid in enumerate(ids.flatten()):
            t_cam = tvecs[i].ravel()
            R_cam, _ = cv2.Rodrigues(rvecs[i])
            # 카메라 좌표 → CF
            xyz_cf = cf.cam_to_cf(t_cam, self.extr)
            # yaw: 마커 로컬 z 축이 CF 에서 어느 방향을 보는지 — 상자는 윗면 마커라 z 가 위
            R_cf = self.extr[:3, :3] @ R_cam
            yaw_cf = float(np.arctan2(R_cf[1, 0], R_cf[0, 0]))
            out.append((int(mid), xyz_cf, yaw_cf))
        return out

    def recalibrate_extrinsics_from_fiducials(self):
        """프레임에 붙여 둔 고정 마커 (ARUCO_FIDUCIAL_IDS) 들을 보고 extr 재추정.
        overview 카메라 전용. 손목 카메라는 TCP 기준이라 이 방법 안 됨."""
        if not cf.ARUCO_FIDUCIAL_IDS:
            raise RuntimeError("frame.ARUCO_FIDUCIAL_IDS 가 비어 있음. 고정 마커 위치를 등록한다.")
        import cv2
        import cv2.aruco as aruco
        img = self.frame()
        corners, ids, _ = aruco.detectMarkers(img, self._dict, parameters=self._params)
        if ids is None:
            raise RuntimeError("보이는 fiducial 없음")
        obj_pts, img_pts = [], []
        for i, mid in enumerate(ids.flatten()):
            mid = int(mid)
            if mid not in cf.ARUCO_FIDUCIAL_IDS:
                continue
            cf_xyz = cf.ARUCO_FIDUCIAL_IDS[mid]
            # 마커 네 모서리 (CF 기준)
            s = self.marker_size / 2
            # 마커 평면은 z 로 위, x 가 오른쪽, y 가 아래 — ArUco 규약. 설치할 때 통일할 것.
            obj_pts.extend([
                cf_xyz + np.array([-s, +s, 0]),
                cf_xyz + np.array([+s, +s, 0]),
                cf_xyz + np.array([+s, -s, 0]),
                cf_xyz + np.array([-s, -s, 0]),
            ])
            img_pts.extend(corners[i].reshape(-1, 2))
        if len(obj_pts) < 4:
            raise RuntimeError(f"유효 fiducial 모서리 {len(obj_pts)} 개 — 최소 4개 필요")
        obj_pts = np.asarray(obj_pts, np.float32)
        img_pts = np.asarray(img_pts, np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.K, self.dist)
        if not ok:
            raise RuntimeError("solvePnP 실패")
        R, _ = cv2.Rodrigues(rvec)
        # solvePnP 는 CF→cam 을 돌려줌. 우리는 cam→CF 가 필요.
        R_cam_to_cf = R.T
        t_cam_to_cf = -R.T @ tvec.ravel()
        self.extr = np.eye(4)
        self.extr[:3, :3] = R_cam_to_cf
        self.extr[:3, 3] = t_cam_to_cf
        return self.extr


class DualCamera(Camera):
    """실물 + 시뮬. frame() 과 markers() 는 실물 기준."""

    def __init__(self, real: Camera, sim: Camera):
        self.real, self.sim = real, sim

    def frame(self):
        return self.real.frame()

    def markers(self):
        return self.real.markers()
