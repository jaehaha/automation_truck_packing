# camera: 카메라 (손목 + 전체뷰)

`Camera` 인터페이스: `frame() -> ndarray`, `markers() -> [(id, xyz, yaw)]`.

| 파일 | 내용 |
|---|---|
| `camera.py` | `Camera` (추상), `SimCamera` (MuJoCo 렌더링), `RealCamera` (OpenCV + ArUco, 미구현) |

쓰는 쪽: [`manager/robot.py`](../../manager/robot.py). 설계: [`../../pipeline/details_by_modules/camera.md`](../../pipeline/details_by_modules/camera.md).
