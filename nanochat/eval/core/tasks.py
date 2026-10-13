"""Evaluate the CORE metric (DCLM, https://arxiv.org/abs/2406.11794) for a base
model.

Task definitions, few-shot counts and per-task data ship in one bundle
(`eval_bundle.zip`), downloaded once into the nanochat base dir.
`run_core_eval` is the public entrypoint. Runs single-process or under torchrun,
where `evaluate_task` strides examples across ranks.
"""
import os
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile

from nanochat.eval.core.harness import evaluate_task
from nanochat.paths import download_file_with_lock, get_base_dir
from nanochat.runtime import print0

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(file_path):
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")

def get_eval_bundle_dir():
    """Path to the local eval bundle, downloading + unzipping it on first use."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
    return eval_bundle_dir

def evaluate_model(model, tokenizer, device, max_per_task=-1):
    """Evaluate a base model on the CORE benchmark. `max_per_task` crops each
    task to that many examples for testing (-1 = disable)."""
    eval_bundle_dir = get_eval_bundle_dir()
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # The data often arrives ordered, so shuffle before any max_per_task crop.
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        end_time = time.time()
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {end_time - start_time:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out


class EvalModel:
    """Callable `(input_ids) -> logits` view of a model, with the max context
    length the harness should truncate prompts to. This is the harness's model
    contract."""

    def __init__(self, model, max_seq_len: int):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids):
        return self.model(input_ids)


def run_core_eval(model, tokenizer, device, max_per_task=-1):
    """Public entrypoint: CORE-evaluate a `(model, tokenizer)` pair on `device`.

    `model` must be callable as `model(input_ids) -> logits` of shape
    (B, T, vocab); an optional `model.max_seq_len` caps prompt length. Returns a
    flat W&B-summary metric dict:

        {"core_metric": <mean centered accuracy over all tasks>,
         "core/<task>": <centered accuracy>,        # per task
         "core_acc/<task>": <raw accuracy>}         # per task

    Centered accuracy rescales each task by its random-guessing baseline b
    (in percent): (acc - b/100) / (1 - b/100), so 0 = chance and 1 = perfect.
    """
    out = evaluate_model(model, tokenizer, device, max_per_task=max_per_task)
    flat = {"core_metric": out["core_metric"]}
    for label, centered in out["centered_results"].items():
        flat[f"core/{label}"] = centered
    for label, acc in out["results"].items():
        flat[f"core_acc/{label}"] = acc
    return flat
