# Programatic dataset generation tool

This tool is designed to make it easier to create datasets systematically and control the types of variation in the stimuli. This tool cannot do the generation for you as that will be specific to your individual experiment however it provides the tools and structure to make it as easy as possible to use this approach. Additionally, we provide examples of how this tool can be used to create different datasets and the different approaches you can take to implement the generation.

## Setup

The tool targets **Python 3.14** and uses [`uv`](https://docs.astral.sh/uv/) for dependency management. If you do not already have `uv` installed:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

The tool is not yet published to PyPI, so the workflow is to clone the repository and run your generation scripts from inside it (or add it as a path/workspace dependency in your own project).

```bash
git clone https://github.com/<your-fork-or-this-repo>/programatic_dataset_generation_tool.git
cd programatic_dataset_generation_tool

# install core dependencies into a managed .venv
uv sync

# (optional) also install the inspect_ai adapter if you want to load
# generated datasets directly into inspect_ai for evaluation
uv sync --group inspect

# verify the install by running the test suite
uv run pytest
```

To use the tool, write your generation script anywhere inside the repo (see `./example/` for full scripts) and run it with `uv run python your_script.py`. The packages you will import from are `dataset`, `generation`, and `evaluation` — for example:

```python
from generation.generate import SampleSpec
from generation.runner import run
from generation.utils import grid
from generation.validation import validates
from dataset.stimulus import Stimulus, Message, ContentText, ContentImage
from evaluation.inspect_utils import load_dataset  # requires the inspect group
```

## Quickstart

Here is a complete hello-world script. It defines one demand (`addition`), one `SampleSpec`, and one generation function, then writes a small dataset to disk.

```python
import random
from pathlib import Path

from dataset.stimulus import Stimulus, Message
from generation.generate import SampleSpec
from generation.runner import run


def generate_stimulus(spec: SampleSpec, rng: random.Random) -> Stimulus:
    a = rng.randint(1, 9)
    b = rng.randint(1, 9)
    return Stimulus(
        spec=spec,
        messages=[Message(role="user", content=f"What is {a} + {b}?")],
        target=str(a + b),
    )


specs = [SampleSpec(demands={"addition": 1})]

run(
    generate_stimulus,
    specs,
    n_reps=3,
    output_dir=Path("hello_world_dataset"),
    seed=42,
)
```

Save this as `hello.py` and run it with `uv run python hello.py`. After it finishes, `./hello_world_dataset/` will contain three generated samples plus a manifest describing the run.

## Output folder layout

`run()` writes a self-contained dataset folder. For the quickstart above, you would get:

```
hello_world_dataset/
├── stimuli.jsonl       # one JSON-encoded Stimulus per line, in (spec_index, rep_index) order
├── manifest.json       # name, library_version, timestamp, global_seed, n_reps, n_stimuli, specs
└── assets/             # only created if any stimuli contain non-text content
    ├── inline/         # bytes embedded as data URIs (e.g. ContentImage.from_bytes(...))
    │                   # named <sample_id>_<message_index>_<content_index>.<ext>
    └── files/          # files referenced from disk, content-hash deduped
                        # named <original_stem>__<sha256[:12]>.<ext>
```

A few things worth knowing:
- A pure-text dataset (like the quickstart) will not have an `assets/` folder at all.
- `http://...` and `https://...` references are left untouched in `stimuli.jsonl` — they are not downloaded.
- `manifest.json` captures everything needed to reproduce the dataset (library version, full spec list, global seed, n_reps). Re-running the same script gives byte-identical output.

## Structure

Since each experiment requires different stimuli in different formats this tool does not generate the actual stimuli itself. Rather you the user provide a python function of the format
```python
def generate_stimuli(spec: SampleSpec, rng: random.Random) -> Stimulus:
    ...
```
where the `spec` argument defines the specific requirement for this individual stimulus. This can include the specific demands being tested, if the sample is a control or test sample or other metadata about the generation such as the resolution of any images to generate or similar. Your generate function will then need to generate a stimulus object with all the requirements specified in the spec and anything not specified randomised using `rng` as a generator. Once this is done you can specify what combinations of requirements you want in your dataset along with other values such as how many samples and the seed value. The tool with then create the samples with each combination you want, validate that each one correctly tests the demands you want it to test and will save the results to a folder including all of the stimuli, assets such as images and a manifest of the dataset describing how and when the dataset was generated for easy replication.

Additionally, we provide tools to load your datasets into `inspect_ai` and provide examples of how to run evaluations on them. However, if you wish to use the datasets in any other framework you can simply parse the json files and load them into the specific format for that framework.

## Examples

In this section we will cover a variety of examples of how to implement different experiments using this tool. The code for each of these can be found in `./example`.

### Simple QA (example workflow)

This example is about a simple question answer experiment. These types of experiment are not well suited to this type of dataset generation (unless you generate the questions using a knowledge base such as wiki data) however the point of this example is to show you the features of the tool without complicated generation logic and show a full run through of the process.

#### planning
First we want to specify the different demands and combinations we want to test. In this case we are building an experiment where we ask a language model to solve maths problems and so we might decide that our demands are `addition`, `subtraction`, `multiplication` and `division`. Demands are stored as a `dict[str, int]` mapping each demand name to a *level*, which lets you describe not just that a demand is present but how much of it the stimulus requires (e.g. working memory level 3 versus level 5). The level values follow three conventions:

- `name: N` where `N > 0` means the sample must exhibit that demand at level `N`.
- `name: 0` means the sample must explicitly **not** exhibit that demand. This is how you ask for a guaranteed-absent demand rather than just leaving it unconstrained.
- A name that is **absent** from the dict is unspecified: the generator may produce any level (including absent), and no validators registered for that demand will run.

If your experiment has no meaningful notion of "level" (as with these maths operations, where a demand is simply present or not) just use `1` for present and `0`/omission for absent. Next, we will define other values we want to control that are not demands themselves, for example we will probably want to control the number of operations in the question and the range of values for the inputs to the operations. We will give these the names `num_operations` and `min_values`/`max_values`.

Now that we have defined our demands and parameters we can build a list of specs with one spec corresponding to a single sample in your dataset. This can be done manually:
```python
specs = [
    SampleSpec(demands={"addition": 1}, params={"num_operations" : 3, "min_values" : 1, "max_values" : 5}),
    SampleSpec(demands={"addition": 1, "subtraction": 1}, params={"num_operations" : 3, "min_values" : 1, "max_values" : 5}),
    SampleSpec(demands={"multiplication": 1}, params={"num_operations" : 3, "min_values" : 1, "max_values" : 5}),
    ...
]
```
or we provide methods to create combinations from smaller subsets which is much easier.
```python
from itertools import combinations
from generation.utils import grid

ops = ["addition", "subtraction", "multiplication", "division"]
demand_combos = [
    {op: 1 for op in combo}
    for r in range(1, len(ops) + 1)
    for combo in combinations(ops, r)
]

specs = grid({
    "demands": demand_combos,
    "num_operations": [3, 4, 5],
    "min_values": [1],
    "max_values": [5, 10, 20],
})
```
more examples of these can be found in `./generation/utils.py`.

#### generation
Now that we have defined the names for our demands and params we can start on implementing the generation. The first step in generation is often to check if the demands and params given are a valid combination. In this example a simple check we might do is to check if the number of demands required is less than the number of operations, as we cannot generate a sample that required both addition and multiplication using only one operation. When we encounter a sample we cannot generate we will throw an error, in this case our function will look something like this:

```python
def generate_stimulus(spec: SampleSpec, rng: random.Random) -> Stimulus:
    num_operations = spec.params.get("num_operations", 3)

    if len(spec.demands) > num_operations:
        raise ValueError(f"cannot generate a sample with {num_operations} that has demands {spec.demands}")
```
next we will generate the specific question and calculate the answer. in our case we will simply pick a starting number and append a random operation and number (prioritising operations that have not yet been added) and adding brackets around the question each time and tracking what the answer is until we reach the number of operations required.

```python
    min_value = spec.params.get("min_values", 1)
    max_value = spec.params.get("max_values", 10)

    symbols = {
        "addition": "+",
        "subtraction": "-",
        "multiplication": "*",
        "division": "/",
    }

    remaining = set(spec.demands)
    question = str(rng.randint(min_value, max_value))
    answer = int(question)

    for _ in range(num_operations):
        choices = remaining if remaining else spec.demands
        op = rng.choice(sorted(choices))
        remaining.discard(op)

        value = rng.randint(min_value, max_value)
        question = f"({question} {symbols[op]} {value})"

        if op == "addition":
            answer += value
        elif op == "subtraction":
            answer -= value
        elif op == "multiplication":
            answer *= value
        elif op == "division":
            answer /= value
```

Now we have a question and an answer that satisfies the requirements set out by the spec, it is now time to return the stimulus object. The `Stimulus` class is designed to be as flexible as possible but still fitted to how most llm experiments are designed, use the stimulus however is best for you and your experiment, the structure is only there to help you keep things organised. The main part of the stimulus class is `messages` field, this field is designed to hold a list of messages which have a source (either system, user, assistant or tool) and some content, be that images text or whatever. The idea is that the messages should be what you want to be the input context for the model you are testing and is useful if you want to pre-load the context with a conversation in progress. In our case we will just add a message from the user with the text `Can you tell me what the answer is to {question}?`. Next we want to fillout the `target` field, this field should hold the answer to your sample if there is one, in our case we can just set our answer as the target. Next we set the spec to the spec we were given so we know all the requirements for this sample and we can fill out the metadata with any extra information we might need during evaluation or analysis, in our case none. Finally we leave the `validators_ran` field empty and we'll cover that later.

```python
return Stimulus(
    spec=spec,
    messages=[Message(role="user", content=f"Can you tell me what the answer is to {question}?")],
    target=str(answer),
)
```
Now we have a full generation function for our experiment and the specs for all the samples we want we can run the dataset builder and output a dataset to a given folder.
```python
run(generate_stimulus, specs, n_reps=2, output_dir=out_dir, seed=12345, max_workers=4)
```

#### validation

This is an idealised example and in real examples with more complicated generation it is useful to add validation to check that each sample requires the demands in its spec. To that end you can define validator functions which will automatically run on each sample with a given demand, for example if we might want to add a validator for each of our demands to check if the question contains the correct symbol.
```python
@validates(name="contains + symbol", demand="addition")
def check_contains_addition(stimulus: Stimulus, spec: SampleSpec) -> None:
    assert "+" in stimulus.messages[0].content
```
By default a validator runs whenever its demand appears in the spec, at any level (including `0`). If you want a validator to only run at a specific demand level you can pass the optional `level` argument. The most useful case for this is `level=0`, which lets you assert that a demand really is absent from the stimulus:
```python
@validates(name="no addition symbol", demand="addition", level=0)
def check_no_addition(stimulus: Stimulus, spec: SampleSpec) -> None:
    assert "+" not in stimulus.messages[0].content
```
This validator only runs on samples whose spec contains `"addition": 0`; for any other level it is skipped.

We can also specify validators that will run on all samples regardless of the demands, this is useful for checking the parameters are being followed.
```python
@validates(name="check number of operations", demand="*")
def check_num_ops(stimulus: Stimulus, spec: SampleSpec) -> None:
    expected = spec.params.get("num_operations")
    if expected is None:
        return
    text = stimulus.messages[0].content
    count = sum(text.count(s) for s in "+-*/")
    assert count == expected, f"expected {expected} operations, found {count}"
```

#### evaluation

now that we have the dataset and it is fully validated we can move to running the evaluation itself. You can use whatever framework you like but we provide some utils to load the dataset into inspect including loading the messages. Moreover, as a little cheat, if there is only one message and it is from the user, the content of that message is loaded into inspect as the input for the sample, which is then automatically inserted as a user message. 
```python
from inspect_ai import Task, task, eval as inspect_eval
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, system_message

from evaluation.inspect_utils import add_messages_from_metadata, load_dataset


@task
def simple_qa_task(dataset_dir: str)->Task:
    dataset = load_dataset(dataset_dir)
    return Task(
        dataset=dataset,
        solver=[system_message("You are a helpful agent which answers questions from the user."), generate()],
        scorer=includes(),
    )
```

### Rotating figures (using images example)

In this example we will focus on how to use images in your stimuli. This example is based on the rotating figure task created for the visual perspective taking paper and the details of the experiment are not important, all you need to know is that we want to show the language model an image of a person in a room along with some numbers on the floor. In your own experiment you can create your images however you want, in our case we use the PIL library to arrange clipart images of a person and a symbol on the ground infront of them, but anything could work here.


#### Adding images to messages
Once we have the images and anything we need for the stimulus, such as the question the user wants to ask, we can then add them to the messages field of the stimulus. To add an image to a message you set the content field of the message to an object of type `ContentImage` which has a similar format to inspect. 
```python
message = Message(
                role="user",
                content=[
                    ContentText(text=prompt),
                    ContentImage.from_bytes(buf.getvalue(), suffix="jpg"),
                ],
            ),
```
the data for an image can either come from a string of bytes including the data of the image, a path to a file containing the image or a URL pointing to the image. The same is true for each of the other datatypes supported by the tool but images are the easiest place to get started for most people. Additionally, when the sample is finished being generated and passes all the validators, each image (or other content) is then saved to the dataset under the assets folder and the uri changed to point to the saved file. Byte strings are saved to the `inline` subfolder and paths to files are copied to `files`. Content from URLs is allowed but is not copied to the dataset for security reasons. 


#### Adding arbitry files

Sometimes when creating a dataset you might want to add some files to the dataset that are not part of any messages but are needed during evaluation. An example of this might be if you have a docker file for each sample defining the environment the agent acts in. For these situations you can add the file yourself during the generate function using the context variable `current_output_dir()` to get the output path of the dataset. We recommend avoiding this if possible as you will need to handle file collisons and cleanup if the sample fails to generate or pass validation. However the option exists for those usecases where there are no other options.


### Intuit (using templates to generate stimuli)



### agentic ToM (using metadata during evalution)
