"""LLM judge for answer correctness, as an Inspect task.

The code-level checks in generate.py prove citations are real; they can't tell
whether the answer is *right*. Here a stronger model (Sonnet) compares each
generated answer with the human-reviewed reference answer from the golden set.

The answers already exist (data/runs/generations__*.jsonl), so the task does
not generate anything: a `replay` solver puts the stored answer in place and
only the grader calls the API.

    inspect eval guichet/judge.py -T variant=pages5 --model mockllm/model

(`--model mockllm/model` because the solver makes no model call; the grader
model is set below.) Results land in ./logs; scripts/judge_agreement.py
compares them with hand grades.

Only answerable questions where the system gave an answer are judged:
refusals and unanswerable items are already scored mechanically.
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import Generate, TaskState, solver

ROOT = Path(__file__).resolve().parents[1]  # Inspect runs tasks from their own folder
JUDGE_MODEL = "anthropic/claude-sonnet-5-5"
GENERATOR = "claude-haiku-5-5"

TEMPLATE = """You are grading an answer to a question an international student asked about French administration.
A human wrote the reference answer from the official service-public.gouv.fr page and checked it.

[Question]
{question}

[Reference answer]
{criterion}

[Submitted answer]
{answer}

Grade the submitted answer against the reference only. Ignore style, length, language
(French or English) and extra details that do not contradict the reference.

- C (correct): states the reference's key facts (numbers, deadlines, conditions, the yes/no) with nothing that contradicts it.
- P (partially correct): nothing false, but it misses a key fact or a condition that would change what the student does.
- I (incorrect): contradicts the reference, gets a key fact wrong, or does not answer the question.

{instructions}"""

INSTRUCTIONS = """First explain your reasoning in two or three sentences, comparing the specific facts.
Then end with exactly one line: GRADE: C, GRADE: P or GRADE: I"""


def load_samples(variant: str, generator: str = GENERATOR) -> list[Sample]:
    golden = {}
    with (ROOT / "evals/golden.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            golden[item["item_id"]] = item
    samples = []
    path = ROOT / f"data/runs/generations__{generator}__{variant}.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            gen = json.loads(line)
            item = golden[gen["item_id"]]
            if not item["answerable"] or not item["reviewed"] or gen.get("found") is not True:
                continue
            language = gen["language"]
            samples.append(Sample(
                id=f"{gen['item_id']}::{language}",
                input=item[f"question_{language}"],
                target=item[f"answer_{language}"],
                metadata={"item_id": gen["item_id"], "language": language, "variant": variant,
                          "submitted": gen.get("answer", "")},
            ))
    return samples


@solver
def replay():
    """Use the stored answer instead of generating one."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.output = ModelOutput.from_content(model="replay", content=state.metadata["submitted"])
        return state
    return solve


@task
def guichet_judge(variant: str = "pages5", generator: str = GENERATOR, judge: str = JUDGE_MODEL):
    return Task(
        dataset=MemoryDataset(load_samples(variant, generator), name=f"guichet-{variant}"),
        solver=replay(),
        scorer=model_graded_qa(template=TEMPLATE, instructions=INSTRUCTIONS,
                               partial_credit=True, model=judge),
    )
