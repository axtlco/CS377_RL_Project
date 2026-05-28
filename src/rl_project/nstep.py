from __future__ import annotations

from collections import deque

from .replay import Transition


class NStepTransitionBuffer:
    def __init__(self, n_step: int, gamma: float) -> None:
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self._queue: deque[Transition] = deque()

    def append(self, transition: Transition) -> list[Transition]:
        self._queue.append(transition)
        ready: list[Transition] = []
        if len(self._queue) >= self.n_step:
            ready.append(self._build_first())
            self._queue.popleft()
        if transition.done or transition.truncated:
            while self._queue:
                ready.append(self._build_first())
                self._queue.popleft()
        return ready

    def _build_first(self) -> Transition:
        reward_ext = 0.0
        reward_train = 0.0
        reward_ride = 0.0
        last = self._queue[0]
        actual_n = 0
        for idx, item in enumerate(self._queue):
            discount = self.gamma**idx
            reward_ext += discount * item.reward_ext
            reward_train += discount * float(item.reward_train)
            reward_ride += discount * item.reward_ride
            last = item
            actual_n += 1
            if actual_n >= self.n_step or item.done or item.truncated:
                break
        first = self._queue[0]
        return Transition(
            obs=first.obs,
            action=first.action,
            reward_ext=reward_ext,
            next_obs=last.next_obs,
            done=last.done,
            truncated=last.truncated,
            env_seed=first.env_seed,
            episode_id=first.episode_id,
            timestep=first.timestep,
            picked_key=first.picked_key,
            opened_door=first.opened_door,
            entered_second_room=first.entered_second_room,
            reached_goal=last.reached_goal,
            timeout=last.timeout,
            pickup_attempt=first.pickup_attempt,
            toggle_attempt=first.toggle_attempt,
            cell_position=first.cell_position,
            actual_n=actual_n,
            reward_train=reward_train,
            reward_ride=reward_ride,
            ride_count_scale=first.ride_count_scale,
        )
