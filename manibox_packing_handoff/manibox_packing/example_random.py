"""학습 루프 뼈대: 마스크 안에서 무작위로 고르는 정책. 여기의 행동 선택을 학습할 정책으로 바꾸면 된다."""
import gymnasium as gym
import numpy as np

import pack_env  # noqa: F401  ("ManiBoxPack-v0" 등록)

env = gym.make("ManiBoxPack-v0")
rng = np.random.default_rng(0)

for episode in range(3):
    obs, info = env.reset(seed=episode)
    done, total = False, 0.0
    while not done:
        mask = info["action_mask"]                       # 둘 수 있는 행동만 True
        action = int(rng.choice(np.flatnonzero(mask)))   # <- 정책으로 교체
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated
    print(f"에피소드 {episode}: 보상 {total:.2f}, 적재 {info['placed']}/{info['total']}, "
          f"사용률 {info['util'] * 100:.1f}%, 종료 {info['reason']}")

print("마지막 에피소드 배치 3개:", env.unwrapped.export()[:3])
