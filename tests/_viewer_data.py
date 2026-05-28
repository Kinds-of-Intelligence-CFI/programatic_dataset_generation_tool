"""Self-contained dataset builders for the viewer tests.

The example datasets under ``example/`` are gitignored, so viewer tests build
their own fixtures here (including a real inline image asset) via the library's
own ``write_dataset`` -- guaranteeing the tests run anywhere.
"""

import io
from pathlib import Path

from PIL import Image

from dataset.stimulus import ContentImage, ContentText, Message, Stimulus
from dataset.writer import write_dataset
from generation.generate import SampleSpec


def _png(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(buf, format="PNG")
    return buf.getvalue()


def build_image_dataset(out: Path) -> Path:
    """A two-stimulus dataset where each user message carries text + an image.

    Inline images land at ``assets/inline/<id>_0_1.png`` (text is content 0,
    image is content 1).
    """
    specs = [
        SampleSpec(
            demands={},
            params={"stimulus_set": "control_1", "question_type": "visual"},
        ),
        SampleSpec(
            demands={"spatial_perspective": 1},
            params={"stimulus_set": "level_1", "question_type": "spatial"},
        ),
    ]
    stimuli = [
        Stimulus(
            spec=spec,
            messages=[
                Message(
                    role="user",
                    content=[
                        ContentText(text="What do you see?"),
                        ContentImage.from_bytes(_png((255, 0, 0)), suffix="png"),
                    ],
                )
            ],
            target=str(i),
            sample_id=str(i),
            metadata={"figure_type": "arrowman_1"},
            validators_ran=["target_non_empty", "user_message_has_image"],
        )
        for i, spec in enumerate(specs)
    ]
    write_dataset(
        out,
        name="image_ds",
        stimuli=stimuli,
        specs=specs,
        global_seed=7,
        n_reps=1,
    )
    return out


def build_text_dataset(out: Path) -> Path:
    """A two-stimulus, asset-free, text-only dataset."""
    specs = [
        SampleSpec(demands={}, params={"topic": "geo", "level": 1}),
        SampleSpec(demands={"recall": 1}, params={"topic": "geo", "level": 2}),
    ]
    stimuli = [
        Stimulus(
            spec=spec,
            messages=[Message(role="user", content=[ContentText(text="hi")])],
            target=target,
            sample_id=str(i),
        )
        for i, (spec, target) in enumerate(zip(specs, ["a", "b"]))
    ]
    write_dataset(
        out,
        name="text_ds",
        stimuli=stimuli,
        specs=specs,
        global_seed=1,
        n_reps=1,
    )
    return out
