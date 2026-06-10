import contextvars
import difflib
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from dataset.stimulus import Stimulus
from dataset.writer import write_dataset
from generation.generate import GenerateFn, SampleSpec
from generation.glossary import DEFAULT_GLOSSARY, Glossary
from generation.validation import (
    registered_demands,
    run_validators,
)

_current_output_dir: contextvars.ContextVar[Path] = contextvars.ContextVar(
    "programatic_dataset_generation_tool.current_output_dir"
)


def current_output_dir() -> Path:
    """Return the output dir of the currently running ``run()``.

    Call from inside a ``generate_fn`` (or validator) to learn where the dataset
    is being written, e.g. to save extra files alongside ``stimuli.jsonl``.
    Raises ``RuntimeError`` when called outside a ``run()``.
    """
    try:
        return _current_output_dir.get()
    except LookupError:
        raise RuntimeError(
            "current_output_dir() can only be called from within a generate_fn "
            "or validator executed by run()."
        ) from None


def run(
    generate_fn: GenerateFn,
    specs: list[SampleSpec],
    n_reps: int,
    output_dir: Path,
    seed: int,
    *,
    name: str | None = None,
    max_workers: int | None = None,
    functional: SampleSpec | None = None,
    glossary: Glossary | None = DEFAULT_GLOSSARY,
) -> None:
    """Generate stimuli in parallel, validate per-sample, write to output_dir.

    Samples are written to stimuli.jsonl in deterministic (spec_index, rep_index)
    order regardless of completion order. A validator failure aborts generation
    and cancels pending work; partial JSONL is left on disk for debugging.

    `functional` carries demands/params that apply identically to every
    sample. The runner merges them into the per-sample (experimental) spec
    before calling the generator and validators, then stores the two pieces
    separately on the returned Stimulus so they serialise as two top-level
    fields - making the constants visible at a glance in the JSONL and manifest.

    `glossary` constrains the allowed demand names. By default it is the
    team-wide `DEFAULT_GLOSSARY`; any demand used in `specs` or `functional`
    that is not in it raises before generation begins. Pass `glossary=None` to
    disable the check, or another `Glossary` for a different vocabulary.
    """
    if functional is not None:
        _check_param_conflicts(functional, specs)
        _check_demand_conflicts(functional, specs)
    if glossary is not None:
        _check_glossary(glossary, specs, functional)
    _warn_about_missing_validators(specs, functional)

    jobs = [
        (spec_index, rep_index)
        for spec_index in range(len(specs))
        for rep_index in range(n_reps)
    ]

    def _run_one(spec_index: int, rep_index: int) -> Stimulus:
        rng = random.Random(hash((seed, spec_index, rep_index)))
        token = _current_output_dir.set(output_dir)
        try:
            experimental = specs[spec_index]
            effective = _merge(functional, experimental)
            stimulus = generate_fn(effective, rng)
            run_validators(stimulus, effective)
            stimulus.spec = experimental
            stimulus.functional = functional
        finally:
            _current_output_dir.reset(token)
        return stimulus

    def _ordered_results() -> Iterator[Stimulus]:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_run_one, si, ri): idx
                for idx, (si, ri) in enumerate(jobs)
            }
            buffer: dict[int, Stimulus] = {}
            seen_ids: set[str] = set()
            next_to_write = 0
            with tqdm(total=len(jobs), desc="generating") as bar:
                try:
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        stimulus = fut.result()
                        bar.update(1)
                        buffer[idx] = stimulus
                        while next_to_write in buffer:
                            stim = buffer.pop(next_to_write)
                            if stim.sample_id is None:
                                stim.sample_id = str(next_to_write)
                            if stim.sample_id in seen_ids:
                                si_, ri_ = jobs[next_to_write]
                                raise ValueError(
                                    f"Duplicate sample_id {stim.sample_id!r} "
                                    f"at yield index {next_to_write} "
                                    f"(spec_index={si_}, rep_index={ri_})"
                                )
                            seen_ids.add(stim.sample_id)
                            yield stim
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
        functional=functional,
        glossary=glossary,
    )


def _merge(functional: SampleSpec | None, spec: SampleSpec) -> SampleSpec:
    if functional is None:
        return spec
    return SampleSpec(
        demands={**functional.demands, **spec.demands},
        params={**functional.params, **spec.params},
    )


def _check_param_conflicts(functional: SampleSpec, specs: list[SampleSpec]) -> None:
    conflicts: set[str] = set()
    for spec in specs:
        conflicts.update(functional.params.keys() & spec.params.keys())
    if conflicts:
        raise ValueError(
            "Param keys appear in both functional and experimental specs: "
            + ", ".join(sorted(conflicts))
            + ". A functional param is by definition the same for every sample; "
            "remove it from the experimental specs or from functional."
        )


def _check_demand_conflicts(functional: SampleSpec, specs: list[SampleSpec]) -> None:
    conflicts: set[str] = set()
    for spec in specs:
        conflicts.update(functional.demands.keys() & spec.demands.keys())
    if conflicts:
        raise ValueError(
            "Demand keys appear in both functional and experimental specs: "
            + ", ".join(sorted(conflicts))
            + ". A functional demand is by definition the same for every sample; "
            "remove it from the experimental specs or from functional."
        )


def _check_glossary(
    glossary: Glossary, specs: list[SampleSpec], functional: SampleSpec | None
) -> None:
    names: set[str] = set()
    for spec in specs:
        names.update(spec.demands)
    if functional is not None:
        names.update(functional.demands)

    unknown = glossary.unknown(names)
    if not unknown:
        return

    valid = list(glossary.names)
    lines: list[str] = []
    for n in unknown:
        suggestions = difflib.get_close_matches(n, valid, n=3)
        if suggestions:
            hint = ", ".join(repr(s) for s in suggestions)
            lines.append(f"  {n!r} (did you mean: {hint}?)")
        else:
            lines.append(f"  {n!r}")
    raise ValueError(
        f"Demands not in glossary {glossary.name!r}:\n"
        + "\n".join(lines)
        + "\nAdd them to the glossary, or pass glossary=None to disable this check."
    )


def _warn_about_missing_validators(
    specs: list[SampleSpec], functional: SampleSpec | None = None
) -> None:
    all_demands: set[str] = set()
    for spec in specs:
        all_demands.update(spec.demands)
    if functional is not None:
        all_demands.update(functional.demands)
    missing = sorted(all_demands - registered_demands())
    if missing:
        warnings.warn(
            "No validators registered for demands: " + ", ".join(missing),
            stacklevel=3,
        )
