# Programatic dataset generation tool

This tool is designed to make it easier to create datasets systematically and control the types of variation in the stimuli. This tool cannot do the generation for you as that will be specific to your individual experiment however it provides the tools and structure to make it as easy as possible to use this approach. Additionally, we provide examples of how this tool can be used to create different datasets and the different approaches you can take to implement the generation.

## Structure

Since each experiment requires different stimuli in different formats this tool does not generate the actual stimuli itself. Rather you the user provide a python function of the format
```python
def generate_stimuli(spec: Spec, rng: random.Random) -> Stimulus:
    ...
```
where the `spec` argument defines the specific requirement for this individual stimulus. This can include the specific capabilities being tested, if the sample is a control or test sample or other metadata about the generation such as the resolution of any images to generate or similar. Your generate function will then need to generate a stimulus object with all the requirements specified in the spec and anything not specified randomised using `rng` as a generator. Once this is done you can specify what combinations of requirements you want in your dataset along with other values such as how many samples and the seed value. The tool with then create the samples with each combination you want, validate that each one correctly tests the capabilities you want it to test and will save the results to a folder including all of the stimuli, assets such as images and a manifest of the dataset describing how and when the dataset was generated for easy replication.

Additionally, we provide tools to load your datasets into `inspect_ai` and provide examples of how to run evaluations on them. However, if you wish to use the datasets in any other framework you can simply parse the json files and load them into the specific format for that framework.

## Examples

In this section we will cover a variety of examples of how to implement different experiments using this tool. The code for each of these can be found in `./example`.

### Simple QA

This example is about a simple question answer experiment. These types of experiment are not well suited to this type of dataset generation (unless you generate the questions using a knowledge base such as wiki data) however the point of this example is to show you the features of the tool without complicated generation logic and show a full run through of the process.

#### planning
First we want to specify the different capabilities and combinations we want to test. In this case we are building an experiment where we ask a language model to solve maths problems and so we might decide that our capabilities are `addition`, `subtraction`, `multiplication` and `division`. Currently, capabilities are simply strings and if the string is present in spec the stimuli should require it to solve, however in the future they are likely to also support having a value indicating what level of capability is needed. Next, we will define other values we want to control that are not capabilities themselves, for example we will probably want to control the number of operations in the question and the range of values for the inputs to the operations. We will give these the names `num_operations` and `min_values`/`max_values`.

Now that we have defined our capabilities and parameters we can build a list of specs with one spec corresponding to a single sample in your dataset. This can be done manually:
```python
specs = [
    Spec(capabilities={"addition"}, params={"num_operations" : 3, "min_values" : 1, "max_values" : 5}),
    Spec(capabilities={"addition", "subtraction"}, params={"num_operations" : 3, "min_values" : 1, "max_values" : 5}),
    Spec(capabilities={"multiplication"}, params={"num_operations" : 3, "min_values" : 1, "max_values" : 5}),
    ...
]
```
or we provide methods to create combinations from smaller subsets which is much easier.
```python
from itertools import combinations
from generation.utils import grid

ops = ["addition", "subtraction", "multiplication", "division"]
capability_combos = [
    set(combo)
    for r in range(1, len(ops) + 1)
    for combo in combinations(ops, r)
]

specs = grid({
    "capabilities": capability_combos,
    "num_operations": [3, 4, 5],
    "min_values": [1],
    "max_values": [5, 10, 20],
})
```
more examples of these can be found in `./generation/utils.py`.

#### generation
Now that we have defined the names for our capabilities and params we can start on implementing the generation. The first step in generation is often to check if the capabilities and params given are a valid combination. In this example a simple check we might do is to check if the number of capabilities required is less than the number of operations, as we cannot generate a sample that required both addition and multipication using only one operation. When we encounter a sample we cannot generate we will throw an error, in this case our function will look something like this:

```python
def generate_stimulus(spec: Spec, rng: random.Random) -> Stimulus:
    num_operations = spec.params.get("num_operations", 3)

    if len(spec.capabilities) > num_operations:
        raise ValueError(f"cannot generate a sample with {num_operations} that has capabilities {spec.capabilities}")
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

    remaining = set(spec.capabilities)
    question = str(rng.randint(min_value, max_value))
    answer = int(question)

    for _ in range(num_operations):
        choices = remaining if remaining else spec.capabilities
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

Now we have a question and an answer that satifies the requirements set out by the spec, it is now time to return the stimulus object. The `Stimulus` class is designed to be as flexible as possible but still fitted to how most llm experiments are designed, use the stimulus however is best for you and your experiment, the structure is only there to help you keep things organised. The main part of the stimulus class is `messages` field, this field is designed to hold a list of messages which have a source (either system, user, assistant or tool) and some content, be that images text or whatever. The idea is that the messages should be what you want to be the input context for the model you are testing and is useful if you want to pre-load the context with a conversation in progress. In our case we will just add a message from the user with the text `Can you tell me what the answer is to {question}?`. Next we want to fillout the `target` field, this field should hold the answer to your sample if there is one, in our case we can just set our answer as the target. Next we set the spec to the spec we were given so we know all the requrirements for this sample and we can fill out the metadata with any extra information we might need during evaluation or analysis, in our case none. Finally we leave the `validators_ran` field empty and we'll cover that later.

```python
return Stimulus(
    spec=spec,
    messages=[Message(role="user", content="Can you tell me what the answer is to {question}?")]
    target=str(answer)
)
```
Now we have a full generation function for our experiment and the specs for all the samples we want we can run the dataset builder and output a dataset to a given folder.
```python
run(generate_stimulus, specs, n_reps=2, output_dir=out_dir, seed=12345, max_workers=4)
```

#### validation

This is an idealised example and in real examples with more complicated generation it is useful to add validation to check that each sample requires the capabilities in its spec. To that end you can define validator functions which will automatically run on each sample with a given capability, for example if we might want to add a validator for each of our capabilities to check if the question contains the correct symbol.
```python
@validates(name="contains + symbol", capability="addition")
def check_contains_addition(stimulus: Stimulus, spec: Spec) -> None:
    assert "+" in stimulus.messages[0].content
```
We can also specify validators that will run on all samples regardless of the capabilities, this is useful for checking the parameters are being followed.
```python
@validates(name="check number of operations", capability="*")
def check_num_ops(stimulus: Stimulus, spec: Spec) -> None:
    expected = spec.params.get("num_operations")
    if expected is None:
        return
    text = stimulus.messages[0].content
    count = sum(text.count(s) for s in "+-*/")
    assert count == expected, f"expected {expected} operations, found {count}"
```

#### evaluation

now that we have the dataset and it is fully validated we can move to running the evaluation itself. You can use whatever framework you like but we provide some utils to load the dataset into inspect including loading the messages. Moreover, as a little cheat, if there is only one message and it is from the user, the content of that message is loaded into inspect as the input for the sample, which is then automatically insterted as a user message. 
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
