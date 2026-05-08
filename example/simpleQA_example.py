import math
import random
import re
from itertools import combinations
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, system_message

from dataset.stimulus import Message, Stimulus
from evaluation.inspect_utils import load_dataset
from generation.generate import Spec
from generation.runner import run
from generation.utils import grid
from generation.validation import validates


_QUESTION_PREFIX = "Can you tell me what the answer is to "
_QUESTION_SUFFIX = "?"

_SYMBOLS = {
    "addition": "+",
    "subtraction": "-",
    "multiplication": "*",
    "division": "/",
}


def generate_simple_QA(spec: Spec, rng: random.Random) -> Stimulus:
    num_operations = spec.params.get("num_operations", 3)
    if len(spec.capabilities) > num_operations:
        raise ValueError(
            f"cannot generate a sample with {num_operations} operations "
            f"that has capabilities {spec.capabilities}"
        )

    min_value = spec.params.get("min_values", 1)
    max_value = spec.params.get("max_values", 10)

    remaining = set(spec.capabilities)
    question = str(rng.randint(min_value, max_value))
    answer: float = int(question)

    for _ in range(num_operations):
        choices = remaining if remaining else spec.capabilities
        op = rng.choice(sorted(choices))
        remaining.discard(op)

        value = rng.randint(min_value, max_value)
        question = f"({question} {_SYMBOLS[op]} {value})"

        if op == "addition":
            answer = answer + value
        elif op == "subtraction":
            answer = answer - value
        elif op == "multiplication":
            answer = answer * value
        elif op == "division":
            answer = answer / value

    return Stimulus(
        spec=spec,
        messages=[
            Message(role="user", content=f"{_QUESTION_PREFIX}{question}{_QUESTION_SUFFIX}"),
        ],
        target=str(answer),
        metadata={},
    )


def _question_text(stimulus: Stimulus) -> str:
    content = stimulus.messages[0].content
    assert isinstance(content, str)
    return content


def _expression(stimulus: Stimulus) -> str:
    text = _question_text(stimulus)
    return text.removeprefix(_QUESTION_PREFIX).removesuffix(_QUESTION_SUFFIX)


@validates(name="contains_addition", capability="addition")
def contains_addition(stimulus: Stimulus, spec: Spec) -> None:
    assert "+" in _question_text(stimulus), "addition sample missing '+'"


@validates(name="contains_subtraction", capability="subtraction")
def contains_subtraction(stimulus: Stimulus, spec: Spec) -> None:
    assert "-" in _question_text(stimulus), "subtraction sample missing '-'"


@validates(name="contains_multiplication", capability="multiplication")
def contains_multiplication(stimulus: Stimulus, spec: Spec) -> None:
    assert "*" in _question_text(stimulus), "multiplication sample missing '*'"


@validates(name="contains_division", capability="division")
def contains_division(stimulus: Stimulus, spec: Spec) -> None:
    assert "/" in _question_text(stimulus), "division sample missing '/'"


@validates(name="check_num_ops", capability="*")
def check_num_ops(stimulus: Stimulus, spec: Spec) -> None:
    expected = spec.params.get("num_operations")
    if expected is None:
        return
    text = _question_text(stimulus)
    count = sum(text.count(s) for s in "+-*/")
    assert count == expected, f"expected {expected} operations, found {count}"


@validates(name="values_in_range", capability="*")
def values_in_range(stimulus: Stimulus, spec: Spec) -> None:
    min_value = spec.params.get("min_values")
    max_value = spec.params.get("max_values")
    if min_value is None and max_value is None:
        return
    text = _question_text(stimulus)
    for literal in re.findall(r"\d+", text):
        n = int(literal)
        if min_value is not None:
            assert n >= min_value, f"value {n} below min {min_value}"
        if max_value is not None:
            assert n <= max_value, f"value {n} above max {max_value}"


@validates(name="no_unrequested_ops", capability="*")
def no_unrequested_ops(stimulus: Stimulus, spec: Spec) -> None:
    text = _question_text(stimulus)
    for capability, symbol in _SYMBOLS.items():
        if capability not in spec.capabilities:
            assert symbol not in text, (
                f"sample contains '{symbol}' but '{capability}' is not in spec.capabilities"
            )


@validates(name="target_matches_question", capability="*")
def target_matches_question(stimulus: Stimulus, spec: Spec) -> None:
    expression = _expression(stimulus)
    expected = eval(expression)  # safe: we generate the text ourselves
    actual = float(stimulus.target)
    assert math.isclose(actual, expected, rel_tol=1e-9), (
        f"target {actual} does not match expression value {expected}"
    )


@validates(name="user_message_present", capability="*")
def user_message_present(stimulus: Stimulus, spec: Spec) -> None:
    user_messages = [m for m in stimulus.messages if m.role == "user"]
    assert user_messages, "stimulus has no user message"
    content = user_messages[0].content
    assert isinstance(content, str) and content.strip(), "user message is empty"


@task
def simple_qa_task(dataset_dir: str) -> Task:
    dataset = load_dataset(dataset_dir)
    return Task(
        dataset=dataset,
        solver=[
            system_message("You are a helpful agent which answers questions from the user."),
            generate(),
        ],
        scorer=includes(),
    )


if __name__ == "__main__":
    ops = ["addition", "subtraction", "multiplication", "division"]
    capability_combos = [
        set(combo)
        for r in range(1, len(ops) + 1)
        for combo in combinations(ops, r)
    ]
    # Some (capabilities, num_operations) pairs are infeasible (more capabilities
    # than operations) and the generator will raise; the runner surfaces those
    # as failures, which is a useful demo of the framework's error path.
    specs = grid({
        "capabilities": capability_combos,
        "num_operations": [3, 4, 5],
        "min_values": [1],
        "max_values": [5, 10, 20],
    })

    out = Path(__file__).parent / "simpleQA_dataset"
    run(generate_simple_QA, specs, n_reps=2, output_dir=out, seed=12345, max_workers=4)
