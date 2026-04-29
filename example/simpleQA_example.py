import random
from pathlib import Path

from dataset.stimulus import Content, Message, Stimulus
from generation.generate import Spec
from generation.runner import run
from generation.validation import validates


def generate_simple_QA(spec: Spec, rng: random.Random) -> Stimulus:
    question = "What is 2 + 2?"
    answer = "4"
    return Stimulus(
        sample_id=str(rng.randint(0, 10**9)),
        spec=spec,
        messages=[
            Message(
                role="system",
                content=[
                    Content(type="text", data="You are a helpful assistant that answers questions."),
                ],
            ),
            Message(role="user", content=[Content(type="text", data=question)]),
        ],
        target=answer,
        metadata={},
    )


@validates(name="target_is_integer", capability="basic_arithmetic")
def target_is_integer(stimulus: Stimulus, spec: Spec) -> None:
    if not isinstance(stimulus.target, str) or not stimulus.target.lstrip("-").isdigit():
        raise AssertionError(
            f"basic_arithmetic target must be an integer string, got {stimulus.target!r}"
        )


if __name__ == "__main__":
    specs = [
        Spec(capabilities={"basic_arithmetic"}, params={}),
    ]
    out = Path(__file__).parent / "simpleQA_dataset"
    run(generate_simple_QA, specs, n_reps=2, output_dir=out, seed=12345, max_workers=4)
