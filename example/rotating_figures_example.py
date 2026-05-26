"""Rotating figures perspective-taking example.

A Python port of the rotating-figures task from the sibling repo
'visual-perspective-taking', adapted to the conventions of this tool:

- ``generate_fn(spec, rng)`` takes an explicit ``random.Random`` (no module-level
  random state).
- One ``Stimulus`` per call; the (stimulus_set, question_type) condition matrix is
  expressed as a flat ``list[SampleSpec]`` instead of nested loops over images.
- Images are PIL-rendered and embedded inline via ``ContentImage.from_bytes`` so
  the dataset folder is self-contained.

The five conditions:

- ``control_1`` / ``control_2``: no perspective-taking demand.
- ``level_1``: spatial perspective taking (demand ``spatial_perspective``).
- ``level_2``: visual perspective taking (demand ``visual_perspective``).
- ``level_3``: both demands together.

Each condition is paired with both a visual and a spatial question, giving 10
conditions in total. ``n_reps`` controls how many independent stimuli are
generated per condition.

Requires Pillow (in the ``dev`` dep group) and inspect-ai (in the ``inspect``
dep group). Install both with ``uv sync --group dev --group inspect``.
"""

from __future__ import annotations

import io
import math
import os
import random
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, system_message
from PIL import Image

from dataset.stimulus import ContentImage, ContentText, Message, Stimulus
from evaluation.inspect_utils import load_dataset
from generation.generate import SampleSpec
from generation.runner import run
from generation.validation import validates


ASSETS_DIR = Path(__file__).parent / "rotating_figures_assets"


SYMBOL_DICT: dict[str, dict[str, Any]] = {
    "6_9": {"image_file": "im_of_6.jpg", "rotations": {True: "6", False: "9"}},
    "u_n": {"image_file": "im_of_u.jpg", "rotations": {True: "u", False: "n"}},
    "p_d": {"image_file": "im_of_p.jpg", "rotations": {True: "p", False: "d"}},
    "m_w": {"image_file": "im_of_m.jpg", "rotations": {True: "m", False: "w"}},
}


SET_CONFIG: dict[str, dict[str, Any]] = {
    "control_1": {
        "placement_zones": ["left", "right"],
        "rotation_range": (70, 110),
        "use_number": True,
        "rotate_number": False,
    },
    "control_2": {
        "placement_zones": ["front", "behind", "left", "right"],
        "rotation_range": (0, 360),
        "use_number": False,
        "rotate_number": False,
    },
    "level_1": {
        "placement_zones": ["front", "behind"],
        "rotation_range": (0, 360),
        "use_number": True,
        "rotate_number": False,
    },
    "level_2": {
        "placement_zones": ["front_left", "front_right"],
        "rotation_range": (0, 360),
        "use_number": True,
        "rotate_number": True,
    },
    "level_3": {
        "placement_zones": ["front_left", "front_right"],
        "rotation_range": (0, 360),
        "use_number": True,
        "rotate_number": True,
    },
}


DEMAND_MAP: dict[str, set[str]] = {
    "control_1": set(),
    "control_2": set(),
    "level_1": {"spatial_perspective"},
    "level_2": {"visual_perspective"},
    "level_3": {"spatial_perspective", "visual_perspective"},
}


PROMPTS: dict[str, dict[str, str]] = {
    "control_1": {
        "context": "There is a person standing in the centre of the image, and a number or letter on the floor nearby.\n",
        "visual": "What number or letter can you see in the image?\n\nRespond with a single number or letter:\n\n",
        "spatial": "Is the number or letter on the left or right side of the image?\n\nRespond with a single word: LEFT or RIGHT\n\n",
    },
    "control_2": {
        "context": "There is a person standing in the centre of the image. There are coloured walls along the border.\n",
        "visual": "What colour is the wall directly in front of the person?\n\nRespond with a single word: RED, GREEN, BLUE or BLACK\n\n",
        "spatial": "Which side of the image is directly in front of the person?\n\nRespond with a single word: LEFT, RIGHT, TOP or BOTTOM\n\n",
    },
    "level_1": {
        "context": "There is a person standing in the centre of the image, and a number or letter on the floor nearby.\n",
        "visual": "Can the person see the number or letter?\n\nRespond with either: CAN SEE or CANNOT SEE\n\n",
        "spatial": "Is the number or letter in front of or behind the person?\n\nRespond with a single word: FRONT or BEHIND\n\n",
    },
    "level_2": {
        "context": "There is a person standing in the centre of the image, and a number or letter on the floor nearby.\n",
        "visual": "What number or letter can the person see?\n\nRespond with a single number or letter:\n\n",
        "spatial": "Is the number or letter on the person's left or right?\n\nRespond with a single word: LEFT or RIGHT\n\n",
    },
    "level_3": {
        "context": "There is a person standing in the centre of the image, and 2 numbers or letters on the floor nearby\n",
        "visual": "What number or letter can the person see on their {side} side?\n\nRespond with a single number or letter:\n\n",
        "spatial": "Is the number or letter {symbol} on the persons left or right?\n\nRespond with a single word: LEFT or RIGHT\n\n",
    },
}


COLOR_MAP = {"LEFT": "RED", "RIGHT": "GREEN", "TOP": "BLUE", "BOTTOM": "BLACK"}


# === Geometry (rng-refactored from visual-perspective-taking/stimulus_generator.py) ===


def _point_in_triangle(
    px: float, py: float,
    p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float],
) -> bool:
    denom = (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1])
    if abs(denom) < 1e-10:
        return False
    a = ((p2[1] - p3[1]) * (px - p3[0]) + (p3[0] - p2[0]) * (py - p3[1])) / denom
    b = ((p3[1] - p1[1]) * (px - p3[0]) + (p1[0] - p3[0]) * (py - p3[1])) / denom
    c = 1 - a - b
    return all(coord >= 0 for coord in (a, b, c))


def _is_in_cone(
    px: float, py: float, cx: float, cy: float,
    cone_angle: float, fov_angle: float, distance: float,
) -> bool:
    rot = math.radians(cone_angle)
    half_fov = math.radians(fov_angle) / 2
    apex = (cx, cy)
    left = (cx + distance * math.sin(rot - half_fov), cy + distance * math.cos(rot - half_fov))
    right = (cx + distance * math.sin(rot + half_fov), cy + distance * math.cos(rot + half_fov))
    return _point_in_triangle(px, py, apex, left, right)


_PLACEMENT_OFFSETS: dict[str, float] = {
    "front": 90, "behind": -90, "left": 180, "right": 0,
    "front_left": 125, "front_right": 55,
    "behind_left": -125, "behind_right": -55,
}


def _get_cone_angle(base_rotation: float, placement_zone: str) -> float:
    if placement_zone not in _PLACEMENT_OFFSETS:
        raise ValueError(f"Unknown placement zone: {placement_zone}")
    return (base_rotation + _PLACEMENT_OFFSETS[placement_zone]) % 360


def _find_valid_position(
    rng: random.Random,
    bg_size: tuple[int, int],
    fg_box: tuple[int, int, int, int],
    num_size: tuple[int, int],
    margin: int,
    center: tuple[int, int],
    base_rotation: float,
    fov: float,
    dist: float,
    placement_zone: str,
) -> tuple[int, int]:
    bg_w, bg_h = bg_size
    num_w, num_h = num_size
    cone_angle = _get_cone_angle(base_rotation, placement_zone)
    for _ in range(2000):
        x = rng.randint(margin, bg_w - num_w - margin)
        y = rng.randint(margin, bg_h - num_h - margin)
        num_center = (x + num_w // 2, y + num_h // 2)
        num_box = (x, y, x + num_w, y + num_h)
        fg_left, fg_top, fg_right, fg_bottom = fg_box
        overlap = not (
            num_box[2] < fg_left or num_box[0] > fg_right
            or num_box[3] < fg_top or num_box[1] > fg_bottom
        )
        if overlap:
            continue
        if _is_in_cone(num_center[0], num_center[1], center[0], center[1], cone_angle, fov, dist):
            return x, y
    raise RuntimeError(f"Couldn't find position in '{placement_zone}' after 2000 attempts.")


def _calculate_number_rotation(
    rng: random.Random, base_angle: float, as_six: bool,
    location: str, jitter_range: float = 10,
) -> float:
    primary = location.split("_")[0]
    offsets = {
        "front": -90 if as_six else 90,
        "behind": 90 if as_six else -90,
        "left": 0 if as_six else 180,
        "right": 180 if as_six else 0,
    }
    offset = offsets.get(primary)
    if offset is None:
        raise ValueError(f"Invalid location: {location}.")
    jitter = rng.uniform(-jitter_range, jitter_range)
    return (base_angle + offset + jitter) % 360


# === Asset loading ===


def _load_image(filename: str) -> Image.Image:
    return Image.open(ASSETS_DIR / filename).convert("RGBA")


def _figure_files() -> list[str]:
    figures_dir = ASSETS_DIR / "figures_arrow"
    files = [f for f in os.listdir(figures_dir) if f.lower().endswith((".jpg", ".png"))]
    return sorted(files)


# === Target extraction ===


_INVERT_SYMBOL = {
    "6": "9", "9": "6",
    "u": "n", "n": "u",
    "p": "d", "d": "p",
    "m": "w", "w": "m",
}


def _angle_to_position(angle: float) -> str:
    if 45 < angle <= 135:
        return "TOP"
    if 135 < angle <= 225:
        return "LEFT"
    if 225 < angle <= 315:
        return "BOTTOM"
    return "RIGHT"


def _extract_target(meta: dict, stimulus_set: str, question_type: str) -> str:
    if stimulus_set == "level_1":
        loc = meta["relative_location"]
        if question_type == "visual":
            return {"front": "CAN SEE", "behind": "CANNOT SEE"}[loc]
        return loc.upper()

    if stimulus_set in ("level_2", "level_3"):
        if question_type == "visual":
            return meta["number_appearance"]
        return meta["relative_location"].split("_")[-1].upper()

    if stimulus_set == "control_1":
        if question_type == "visual":
            appearance = meta["number_appearance"]
            angle = meta["number_rotation"]
            if 90 <= angle <= 270:
                return _INVERT_SYMBOL.get(appearance, appearance)
            return appearance
        return meta["number_location_x"]

    if stimulus_set == "control_2":
        position = _angle_to_position(meta["figure_rotation"])
        if question_type == "visual":
            return COLOR_MAP[position]
        return position

    raise ValueError(f"Unknown stimulus_set: {stimulus_set}")


# === Prompt assembly ===


def _vocab_options(symbol_key: str) -> tuple[str, str]:
    rots = SYMBOL_DICT[symbol_key]["rotations"]
    return rots[True], rots[False]


def _find_symbol_key(appearance: str) -> str:
    for key, cfg in SYMBOL_DICT.items():
        if appearance in (cfg["rotations"][True], cfg["rotations"][False]):
            return key
    raise ValueError(f"Unknown symbol appearance: {appearance}")


def _build_prompt(
    rng: random.Random, stimulus_set: str, question_type: str, meta: dict,
) -> str:
    p = PROMPTS[stimulus_set]
    prompt = p["context"] + p[question_type]

    if question_type == "visual" and stimulus_set in ("control_1", "level_2", "level_3"):
        primary = meta["number_appearance"]
        opt_a, opt_b = _vocab_options(_find_symbol_key(primary))
        options = [opt_a, opt_b]
        if stimulus_set == "level_3":
            alt = meta["alternate_number"]
            opt_c, opt_d = _vocab_options(_find_symbol_key(alt))
            options = [opt_a, opt_b, opt_c, opt_d]
        rng.shuffle(options)
        if stimulus_set == "level_3":
            prompt += f" {options[0]}, {options[1]}, {options[2]} or {options[3]}"
        else:
            prompt += f" {options[0]} or {options[1]}"

    if stimulus_set == "level_3":
        if question_type == "visual":
            side = meta["relative_location"].split("_")[-1]
            prompt = prompt.format(side=side)
        else:
            prompt = prompt.format(symbol=meta["number_appearance"])

    return prompt


# === Generator ===


def generate_rotating_figure(spec: SampleSpec, rng: random.Random) -> Stimulus:
    stimulus_set: str = spec.params["stimulus_set"]
    question_type: str = spec.params["question_type"]
    cfg = SET_CONFIG[stimulus_set]
    image_size: tuple[int, int] = spec.params.get("image_size", (400, 400))
    figure_scale: float = spec.params.get("figure_scale", 1.5)
    margin: int = spec.params.get("margin", 200)
    fov_angle: float = spec.params.get("fov_angle", 60)
    view_distance: float = spec.params.get("view_distance", 5000)
    jitter_range: float = spec.params.get("jitter_range", 20)

    background = _load_image("background_with_colours.jpg")
    bg_w, bg_h = background.size

    figure_file = rng.choice(_figure_files())
    figure = _load_image(f"figures_arrow/{figure_file}")
    figure = figure.transpose(Image.Transpose.ROTATE_90)

    if cfg["use_number"]:
        symbol_key = rng.choice(sorted(SYMBOL_DICT.keys()))
        symbol_cfg = SYMBOL_DICT[symbol_key]
        number = _load_image(symbol_cfg["image_file"])
    else:
        symbol_key = ""
        symbol_cfg = None
        number = _load_image("blank.png")

    figure_angle = rng.uniform(*cfg["rotation_range"])
    appears_unrotated: bool = rng.choice([True, False])
    placement_zone = rng.choice(cfg["placement_zones"])

    fg_w, fg_h = figure.size
    figure = figure.resize(
        (int(fg_w * figure_scale), int(fg_h * figure_scale)), resample=Image.Resampling.LANCZOS,
    )
    fg_w, fg_h = figure.size
    fg_x = (bg_w - fg_w) // 2
    fg_y = (bg_h - fg_h) // 2
    figure_box = (fg_x, fg_y, fg_x + fg_w, fg_y + fg_h)
    figure_center = (fg_x + fg_w // 2, fg_y + fg_h // 2)

    if cfg["rotate_number"]:
        number_appearance_type = "figure_perspective"
        number_angle = _calculate_number_rotation(
            rng, figure_angle, appears_unrotated, placement_zone, jitter_range,
        )
    else:
        number_appearance_type = "viewer_perspective"
        number_angle = _calculate_number_rotation(
            rng, 0, appears_unrotated, "left", jitter_range,
        )

    num_position = _find_valid_position(
        rng, (bg_w, bg_h), figure_box, number.size, margin,
        figure_center, figure_angle, fov_angle, view_distance, placement_zone,
    )

    figure_rotated = figure.rotate(figure_angle, expand=True)
    number_rotated = number.rotate(number_angle, expand=True)
    fg_x = (bg_w - figure_rotated.size[0]) // 2
    fg_y = (bg_h - figure_rotated.size[1]) // 2
    composed = background.copy()
    composed.paste(figure_rotated, (fg_x, fg_y), figure_rotated)
    composed.paste(number_rotated, num_position, number_rotated)

    alternative_appearance = ""
    if stimulus_set == "level_3":
        opposite_zone = "front_left" if placement_zone == "front_right" else "front_right"
        other_keys = [k for k in sorted(SYMBOL_DICT.keys()) if k != symbol_key]
        alt_cfg = SYMBOL_DICT[rng.choice(other_keys)]
        alt_number = _load_image(alt_cfg["image_file"])
        alt_position = _find_valid_position(
            rng, (bg_w, bg_h), figure_box, alt_number.size, margin,
            figure_center, figure_angle, fov_angle, view_distance, opposite_zone,
        )
        
        flipped = alt_number.rotate(number_angle + 180 % 360, expand=True)
        composed.paste(flipped, alt_position, flipped)
        alternative_appearance = alt_cfg["rotations"][appears_unrotated]

    resized = composed.resize(image_size, resample=Image.Resampling.LANCZOS)
    scale_x = image_size[0] / bg_w
    scaled_num_x = round(num_position[0] * scale_x)
    resized_center_x = image_size[0] // 2
    resized_num_center_x = scaled_num_x + int(number.size[0] * scale_x / 2)
    number_location_x = "left" if resized_num_center_x < resized_center_x else "right"

    if cfg["use_number"] and symbol_cfg is not None:
        appearance = symbol_cfg["rotations"][appears_unrotated]
    else:
        appearance = "none"

    buf = io.BytesIO()
    resized.convert("RGB").save(buf, format="JPEG")

    meta: dict[str, Any] = {
        "stimulus_set": stimulus_set,
        "question_type": question_type,
        "figure_type": figure_file.split(".")[0],
        "figure_rotation": round(figure_angle, 2),
        "number_rotation": round(number_angle, 2),
        "relative_location": placement_zone,
        "number_appearance_type": number_appearance_type,
        "number_appearance": appearance,
        "number_location_x": number_location_x,
        "alternate_number": alternative_appearance,
    }

    target = _extract_target(meta, stimulus_set, question_type)
    prompt = _build_prompt(rng, stimulus_set, question_type, meta)

    return Stimulus(
        spec=spec,
        messages=[
            Message(
                role="user",
                content=[
                    ContentText(text=prompt),
                    ContentImage.from_bytes(buf.getvalue(), suffix="jpg"),
                ],
            ),
        ],
        target=target,
        metadata=meta,
    )


# === Validators ===


_LETTER_VOCAB = {"6", "9", "u", "n", "p", "d", "m", "w"}


def _expected_targets(stimulus_set: str, question_type: str) -> set[str]:
    if stimulus_set in ("control_1", "level_2", "level_3") and question_type == "visual":
        return _LETTER_VOCAB
    if stimulus_set == "control_1" and question_type == "spatial":
        return {"left", "right"}
    if stimulus_set == "control_2" and question_type == "visual":
        return {"RED", "GREEN", "BLUE", "BLACK"}
    if stimulus_set == "control_2" and question_type == "spatial":
        return {"LEFT", "RIGHT", "TOP", "BOTTOM"}
    if stimulus_set == "level_1" and question_type == "visual":
        return {"CAN SEE", "CANNOT SEE"}
    if stimulus_set == "level_1" and question_type == "spatial":
        return {"FRONT", "BEHIND"}
    if stimulus_set in ("level_2", "level_3") and question_type == "spatial":
        return {"LEFT", "RIGHT"}
    raise ValueError(f"No vocab for ({stimulus_set}, {question_type})")


@validates(name="user_message_has_image", demand="*")
def user_message_has_image(stimulus: Stimulus, spec: SampleSpec) -> None:
    user_msgs = [m for m in stimulus.messages if m.role == "user"]
    assert user_msgs, "no user message"
    content = user_msgs[0].content
    assert isinstance(content, list), "user content must be a list of Content items"
    text_items = [c for c in content if isinstance(c, ContentText)]
    image_items = [c for c in content if isinstance(c, ContentImage)]
    assert len(text_items) == 1, f"expected 1 ContentText, found {len(text_items)}"
    assert len(image_items) == 1, f"expected 1 ContentImage, found {len(image_items)}"


@validates(name="target_non_empty", demand="*")
def target_non_empty(stimulus: Stimulus, spec: SampleSpec) -> None:
    assert isinstance(stimulus.target, str) and stimulus.target.strip(), (
        "target must be a non-empty string"
    )


@validates(name="target_in_vocabulary", demand="*")
def target_in_vocabulary(stimulus: Stimulus, spec: SampleSpec) -> None:
    stimulus_set = spec.params["stimulus_set"]
    question_type = spec.params["question_type"]
    expected = _expected_targets(stimulus_set, question_type)
    assert stimulus.target in expected, (
        f"target {stimulus.target!r} not in {sorted(expected)} "
        f"for ({stimulus_set}, {question_type})"
    )


@validates(name="metadata_matches_spec", demand="*")
def metadata_matches_spec(stimulus: Stimulus, spec: SampleSpec) -> None:
    assert stimulus.metadata.get("stimulus_set") == spec.params["stimulus_set"]
    assert stimulus.metadata.get("question_type") == spec.params["question_type"]


@validates(name="placement_zone_in_set", demand="spatial_perspective")
def placement_zone_in_set(stimulus: Stimulus, spec: SampleSpec) -> None:
    stimulus_set = spec.params["stimulus_set"]
    allowed = set(SET_CONFIG[stimulus_set]["placement_zones"])
    actual = stimulus.metadata.get("relative_location")
    assert actual in allowed, (
        f"relative_location {actual!r} not in {sorted(allowed)} for {stimulus_set}"
    )


@validates(name="number_is_figure_rotated", demand="visual_perspective")
def number_is_figure_rotated(stimulus: Stimulus, spec: SampleSpec) -> None:
    appearance_type = stimulus.metadata.get("number_appearance_type")
    assert appearance_type == "figure_perspective", (
        f"visual_perspective stimulus should rotate the symbol to the figure's view; "
        f"got number_appearance_type={appearance_type!r}"
    )


@validates(name="level_3_has_alternate", demand="visual_perspective")
def level_3_has_alternate(stimulus: Stimulus, spec: SampleSpec) -> None:
    if spec.params["stimulus_set"] != "level_3":
        return
    alt = stimulus.metadata.get("alternate_number")
    assert alt, f"level_3 stimulus missing alternate_number (got {alt!r})"


# === Inspect task wrapper ===


@task
def rotating_figures_task(dataset_dir: str) -> Task:
    dataset = load_dataset(dataset_dir)
    return Task(
        dataset=dataset,
        solver=[
            system_message(
                "You are shown an image of a scene. Look at it carefully and answer "
                "the question about it. Respond with only the answer requested."
            ),
            generate(),
        ],
        scorer=includes(),
    )


# === CLI entrypoint ===


if __name__ == "__main__":
    specs: list[SampleSpec] = []
    for set_name in SET_CONFIG:
        for q_type in ("visual", "spatial"):
            specs.append(
                SampleSpec(
                    demands=set(DEMAND_MAP[set_name]),
                    params={
                        "stimulus_set": set_name,
                        "question_type": q_type,
                        "image_size": (400, 400),
                        "figure_scale": 1.5,
                        "margin": 200,
                        "fov_angle": 60,
                        "view_distance": 5000,
                        "jitter_range": 20,
                    },
                )
            )

    out = Path(__file__).parent / "rotating_figures_dataset"
    run(
        generate_rotating_figure,
        specs,
        n_reps=3,
        output_dir=out,
        seed=12345,
        max_workers=4,
    )
