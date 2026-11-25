"""편집하는 곳: 어떤 상황에 어떻게 움직일지 정한 값.

고친 뒤에는  uv run --project ../mp python run.py  로 전체를 다시 돌려 성공 수가 어떻게 바뀌는지 본다.
(이 파일이 바뀌면 저장해 둔 경로(cache/)는 자동으로 다시 계산된다.)
"""

# 투입 구역: 상자의 문 쪽 면이 이 값(월드 x, m)보다 문 쪽이면 싣지 않는 자리로 본다.
# 집어서 세우는 동작이 이 구역을 쓴다. None 이면 모든 자리를 시도한다.
KEEPOUT_X = 0.70

# sim_final 의 상자 6종(stacking.BOXES)에 더할 상자: 종류 번호 -> (가로, 세로, 높이) m, 색
EXTRA_BOXES = {7: ((0.16, 0.16, 0.08), "0.40 0.75 0.75")}

# 놓는 동작 후보. 위에서부터 차례로 시도해 처음 통과한 것을 쓴다.
#   camera : 놓는 순간 손목 카메라 받침(공구 옆으로 82 mm 나온 쪽)이 향하는 곳.
#            "up" 위, "down" 아래, "+y" / "-y" 옆. 받침이 이웃 상자나 바닥에 걸리지 않는 쪽을 앞에 둔다.
#            낮은 자리에서는 손목 관절 한계 때문에 "up" 자세가 나오지 않는다.
#   x_ref  : 팔 자세를 계산할 때 기준으로 삼는 놓는 위치 (월드 x, m). 팔을 뻗는 정도와 레일 여유가 달라진다.
#            None 이면 그 자리 자신의 x (1 cm 단위). 레일 범위 끝에 걸리는 문 쪽 자리를 위한 마지막 후보다.
PLACE_CANDIDATES = [dict(camera=c, x_ref=x) for x in (1.10, 0.80, 1.40, None) for c in ("up", "+y", "-y", "down")]


def place_candidates(box):
    """box: dict(type, height, la('y'|'z'), y, z, x). 상황에 따라 후보 순서를 바꾸고 싶으면 여기에 조건을 적는다.
    예)  if box["z"] > 0.40: return [dict(camera="down", x_ref=1.10)] + PLACE_CANDIDATES"""
    return PLACE_CANDIDATES
