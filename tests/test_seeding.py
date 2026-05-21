from __future__ import annotations

from rl_project.seeding import SeedStream, fixed_eval_seeds


def test_seed_stream_is_reproducible_and_disjoint() -> None:
    a = SeedStream(7)
    b = SeedStream(7)

    assert [a.env_seed(i) for i in range(3)] == [b.env_seed(i) for i in range(3)]
    assert fixed_eval_seeds(7, 3) == fixed_eval_seeds(7, 3)
    assert set(fixed_eval_seeds(7, 3)).isdisjoint({a.env_seed(i) for i in range(3)})
