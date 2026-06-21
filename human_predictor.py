from __future__ import annotations

import os
import pickle
from typing import Sequence, Tuple, List

Pos = Tuple[int, int]

MODEL_PATH = os.path.join("models", "direction_model.pkl")

VALID_DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

FEATURE_COLS = [
    "rel_row",
    "rel_col",
    "wall_up",
    "wall_down",
    "wall_left",
    "wall_right",
    "prev1",
    "prev2",
]

# 这里必须和你队友训练数据里的 prev1 / prev2 编码一致
DIRECTION_TO_ID = {
    "up": 0.0,
    "down": 1.0,
    "left": 2.0,
    "right": 3.0,
}


class HumanDirectionPredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model not found: {model_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict_direction(self, feature_dict: dict) -> str:
        x = [[float(feature_dict[col]) for col in FEATURE_COLS]]
        return str(self.model.predict(x)[0])

    def predict_proba(self, feature_dict: dict) -> dict[str, float]:
        x = [[float(feature_dict[col]) for col in FEATURE_COLS]]
        probs = self.model.predict_proba(x)[0]
        return {
            str(direction): float(prob)
            for direction, prob in zip(self.model.classes_, probs)
        }


def wrap_pos(pos: Pos, grid: Sequence[Sequence[int]]) -> Pos:
    return pos[0] % len(grid), pos[1] % len(grid[0])


def is_floor(grid: Sequence[Sequence[int]], pos: Pos) -> bool:
    row, col = wrap_pos(pos, grid)
    return int(grid[row][col]) == 0


def build_prediction_features(
    grid: Sequence[Sequence[int]],
    human: Pos,
    monster: Pos,
    history: List[str],
) -> dict:
    """
    Build features in exactly the same format as the Random Forest training code.

    features:
    rel_row, rel_col,
    wall_up, wall_down, wall_left, wall_right,
    prev1, prev2
    """

    h = len(grid)
    w = len(grid[0])

    feature = {}

    feature["rel_row"] = (human[0] - monster[0]) / h
    feature["rel_col"] = (human[1] - monster[1]) / w

    for direction, (dr, dc) in VALID_DIRECTIONS.items():
        nr = (human[0] + dr) % h
        nc = (human[1] + dc) % w
        feature[f"wall_{direction}"] = 1 if int(grid[nr][nc]) == 1 else 0

    prev1 = history[-1] if len(history) >= 1 else "up"
    prev2 = history[-2] if len(history) >= 2 else "up"

    feature["prev1"] = DIRECTION_TO_ID.get(prev1, 0.0)
    feature["prev2"] = DIRECTION_TO_ID.get(prev2, 0.0)

    return feature


def predicted_intercept_target(
    grid: Sequence[Sequence[int]],
    human: Pos,
    predicted_direction: str,
    distance: int = 3,
) -> Pos:
    """
    Convert predicted next direction into an interception target.

    Example:
    predicted_direction = "right"
    human = (10, 10)
    target ≈ (10, 13)

    If the target direction is blocked by walls, use the furthest reachable
    floor cell along that direction.
    """

    if predicted_direction not in VALID_DIRECTIONS:
        return human

    dr, dc = VALID_DIRECTIONS[predicted_direction]
    h = len(grid)
    w = len(grid[0])

    best = human

    for step in range(1, distance + 1):
        candidate = (
            (human[0] + dr * step) % h,
            (human[1] + dc * step) % w,
        )

        if is_floor(grid, candidate):
            best = candidate
        else:
            break

    return best