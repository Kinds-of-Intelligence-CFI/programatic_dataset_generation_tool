import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

from dataset.stimulus import Stimulus
from dataset.writer import write_dataset
from generation.generate import GenerateFn, Spec
from generation.validation import (
    registered_capabilities,
    run_validators,
)


def run(
    generate_fn: GenerateFn,
    specs: list[Spec],
    n_reps: int,
    output_dir: Path,
    seed: int,
    *,
    name: str | None = None,
    max_workers: int | None = None,
) -> None:
    """Generate stimuli in parallel, validate per-sample, write to output_dir.

    Samples are written to stimuli.jsonl in deterministic (spec_index, rep_index)
    order regardless of completion order. A validator failure aborts generation
    and cancels pending work; partial JSONL is left on disk for debugging.
    """
    _warn_about_missing_validators(specs)

    jobs = [
        (spec_index, rep_index)
        for spec_index in range(len(specs))
        for rep_index in range(n_reps)
    ]

    def _run_one(spec_index: int, rep_index: int) -> Stimulus:
        rng = random.Random(hash((seed, spec_index, rep_index)))
        stimulus = generate_fn(specs[spec_index], rng)
        run_validators(stimulus, specs[spec_index])
        return stimulus

    def _ordered_results() -> Iterator[Stimulus]:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_run_one, si, ri): idx
                for idx, (si, ri) in enumerate(jobs)
            }
            buffer: dict[int, Stimulus] = {}
            next_to_write = 0
            try:
                for fut in as_completed(futures):
                    idx = futures[fut]
                    stimulus = fut.result()
                    buffer[idx] = stimulus
                    while next_to_write in buffer:
                        yield buffer.pop(next_to_write)
                        next_to_write += 1
            except BaseException:
                for f in futures:
                    f.cancel()
                raise

    write_dataset(
        output_dir,
        name=name if name is not None else output_dir.name,
        stimuli=_ordered_results(),
        specs=specs,
        global_seed=seed,
        n_reps=n_reps,
    )


def _warn_about_missing_validators(specs: list[Spec]) -> None:
    all_caps: set[str] = set()
    for spec in specs:
        all_caps.update(spec.capabilities)
    missing = sorted(all_caps - registered_capabilities())
    if missing:
        warnings.warn(
            "No validators registered for capabilities: " + ", ".join(missing),
            stacklevel=3,
        )
