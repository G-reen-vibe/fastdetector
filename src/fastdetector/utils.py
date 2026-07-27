from typing import Dict, Optional

from huggingface_hub import HfApi, hf_hub_download
from datasets import Dataset, load_dataset, get_dataset_config_names, concatenate_datasets


SHARD_CONFIG_PREFIX = "shard_"


def shard_config_name(batch_id: int) -> str:
    """Return the HF config name a given batch id reads from and writes to."""
    return f"{SHARD_CONFIG_PREFIX}{batch_id}"


def load_dataset_auto_shard(
    dataset_name: str,
    split: str = "train",
    subset_index: Optional[int] = 0,
) -> Dataset:
    """Load one shard of a dataset from the Hugging Face Hub.

    Shards are resolved **by name** (``shard_<i>``), matching the name every
    writer in the pipeline pushes to. Resolving by position instead would let
    a job read one shard and write its results to a different one, silently
    overwriting another machine's output, because config ordering on the Hub
    is not guaranteed to track the numeric shard index.

    Args:
        dataset_name: HF Hub dataset repo ID (e.g. "G-reen/cc-2021-rewritten").
        split: Dataset split (default "train").
        subset_index: Shard/batch id. ``None`` loads the default config
            without shard resolution.

    Returns:
        The loaded Dataset.

    Raises:
        RuntimeError: if the repo's configs cannot be listed, or if the
            requested shard cannot be resolved unambiguously. Failing here is
            deliberate: the previous behaviour silently fell back to the
            default config, which makes every machine in a batched run
            process the same rows.
    """
    if subset_index is None:
        print(f"Loading default config of {dataset_name}...")
        return load_dataset(dataset_name, split=split)

    try:
        configs = get_dataset_config_names(dataset_name)
    except Exception as e:
        raise RuntimeError(
            f"Could not list configs for '{dataset_name}' "
            f"({type(e).__name__}: {e}). Refusing to guess which shard to "
            f"load, because falling back to the default config would make "
            f"every batched job process the same rows."
        ) from e

    expected = shard_config_name(subset_index)

    if expected in configs:
        config_name = expected
    elif len(configs) == 1 and subset_index == 0:
        # Single-config dataset (e.g. filter.py ran without output_shards).
        # Only batch id 0 may claim it; any other id is a duplicate-work bug.
        config_name = configs[0]
        print(
            f"Dataset '{dataset_name}' has no '{expected}' config but exactly "
            f"one config ('{config_name}'); loading it for batch id 0."
        )
    else:
        raise RuntimeError(
            f"Cannot resolve shard {subset_index} of '{dataset_name}': no "
            f"config named '{expected}'. Available configs: {configs}. "
            f"Re-shard the dataset with the '{SHARD_CONFIG_PREFIX}<i>' naming "
            f"scheme, or set override_dataset_input in globals.toml to point "
            f"at the dataset you mean."
        )

    print(f"Loading config '{config_name}' of {dataset_name}...")
    return load_dataset(dataset_name, name=config_name, split=split)


def load_dataset_all_shards(
    dataset_name: str,
    split: str = "train",
) -> Dataset:
    """Load all configs/shards of a dataset from Hugging Face Hub and concatenate them into a single Dataset.

    Args:
        dataset_name: HF Hub dataset repo ID.
        split: Dataset split (default "train").

    Returns:
        The concatenated Dataset containing all rows from all shards.
    """
    try:
        configs = get_dataset_config_names(dataset_name)
    except Exception as e:
        print(f"Notice: Could not list configs for '{dataset_name}': {e}. Loading default config.")
        configs = []

    if configs:
        print(f"Loading all {len(configs)} configs ({configs}) for dataset '{dataset_name}'...")
        shards = []
        for cfg in configs:
            shards.append(load_dataset(dataset_name, name=cfg, split=split))
        if len(shards) == 1:
            return shards[0]
        return concatenate_datasets(shards)
    else:
        print(f"Loading default config for dataset '{dataset_name}'...")
        return load_dataset(dataset_name, split=split)



def upload_readme(
    dataset_name: str,
    files: Optional[Dict[str, bytes]] = None,
    readme_content: str = "",
    append_readme_source: Optional[str] = None,
) -> None:
    """Upload a README and associated files to the Hugging Face Hub.

    Args:
        dataset_name: The name of the dataset to upload to.
        files: Additional files to upload (filename -> bytes), such as charts.
        readme_content: The content of the readme.
        append_readme_source: If set, download the README from this dataset
            and prepend it to *readme_content*.

    Returns:
        None.
    """
    def _extract_yaml(text: str) -> tuple[str, str]:
        """Extract YAML frontmatter from a markdown string.

        Args:
            text: Markdown text potentially starting with '---'.

        Returns:
            Tuple of (yaml_header_with_newline, remaining_body).
        """
        if text.startswith("---"):
            idx = text.find("\n---", 3)
            if idx != -1:
                idx += 4 # length of \n---
                return text[:idx] + "\n", text[idx:].lstrip()
        return "", text

    dataset_yaml = ""
    try:
        print(f"Checking for existing YAML config on '{dataset_name}'...")
        curr_readme_path = hf_hub_download(repo_id=dataset_name, filename="README.md", repo_type="dataset")
        with open(curr_readme_path, "r", encoding="utf-8") as f:
            curr_text = f.read()
            dataset_yaml, _ = _extract_yaml(curr_text)
    except Exception as e:
        print(f"Notice: No existing README or YAML config found on '{dataset_name}' ({e}).")

    prev_readme = ""
    if append_readme_source:
        try:
            print(f"Downloading README.md from '{append_readme_source}' to append new README content...")
            readme_path = hf_hub_download(repo_id=append_readme_source, filename="README.md", repo_type="dataset")
            with open(readme_path, "r", encoding="utf-8") as f:
                prev_text = f.read()
                _, prev_readme = _extract_yaml(prev_text)
        except Exception as e:
            print(
                f"Warning: Could not download README.md from "
                f"'{append_readme_source}': {e}. Using new README content only."
            )

    if prev_readme:
        if not prev_readme.endswith("\n"):
            prev_readme += "\n"
        if not prev_readme.endswith("\n\n"):
            prev_readme += "\n"
        combined_readme = dataset_yaml + prev_readme + readme_content
    else:
        combined_readme = dataset_yaml + readme_content

    if files is None:
        files = {}

    api = HfApi()
    try:
        print(f"Uploading README.md to '{dataset_name}'...")
        api.upload_file(
            path_or_fileobj=combined_readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=dataset_name,
            repo_type="dataset"
        )

        if files:
            for filename, data in files.items():
                print(f"Uploading file '{filename}' to '{dataset_name}'...")
                api.upload_file(
                    path_or_fileobj=data,
                    path_in_repo=filename,
                    repo_id=dataset_name,
                    repo_type="dataset"
                )
        print("README and files uploaded successfully.")
    except Exception as e:
        # Previously this only printed, so a run whose upload failed still
        # exited 0 and looked successful.
        raise RuntimeError(
            f"Failed to upload README/files to '{dataset_name}': "
            f"{type(e).__name__}: {e}"
        ) from e


def apply_filter_conditions(
    dataset: Dataset,
    conditions: list,
    filter_type: str = "AND",
) -> Dataset:
    """Filter a dataset using structured ConditionConfig conditions.

    Supported operators: ``==``, ``!=``, ``>``, ``<``, ``>=``, ``<=``.

    For numeric operators (``>``, ``<``, ``>=``, ``<=``), values are coerced
    to float. If coercion fails (ValueError/TypeError), the condition is
    treated as False (i.e. the row is filtered out).

    A ``None`` cell is treated as non-matching, but a condition naming a
    column that does not exist is an error rather than a silent
    non-match: with filter_type="AND" a single typo would otherwise drop
    every row and upload an empty dataset without warning.

    Args:
        dataset: The dataset to filter.
        conditions: List of ConditionConfig objects (each with .column,
            .operator, .value).
        filter_type: "AND" (all conditions must match) or "OR" (any).

    Returns:
        The filtered dataset.

    Raises:
        KeyError: if any condition names a column absent from the dataset.
    """
    if not conditions:
        return dataset

    missing = sorted({c.column for c in conditions} - set(dataset.column_names))
    if missing:
        raise KeyError(
            f"Filter conditions reference column(s) not present in the "
            f"dataset: {missing}. Available columns: "
            f"{sorted(dataset.column_names)}"
        )

    print("Filtering dataset with parsed conditions:")
    for c in conditions:
        print(c)

    def filter_func(example: dict) -> bool:
        """Evaluate filter conditions on a single dataset example row.

        Args:
            example: Dictionary representing a dataset row.

        Returns:
            True if row meets the conditions according to filter_type, else False.
        """
        bools = []
        for cond in conditions:
            col = cond.column
            op = cond.operator
            val = cond.value

            if example[col] is None:
                bools.append(False)
                continue

            ex_val = example[col]

            try:
                if op == '==': bools.append(ex_val == val)
                elif op == '!=': bools.append(ex_val != val)
                elif op == '>': bools.append(float(ex_val) > float(val))
                elif op == '<': bools.append(float(ex_val) < float(val))
                elif op == '>=': bools.append(float(ex_val) >= float(val))
                elif op == '<=': bools.append(float(ex_val) <= float(val))
                else: bools.append(False)
            except (ValueError, TypeError):
                if op == '==': bools.append(ex_val == val)
                elif op == '!=': bools.append(ex_val != val)
                else: bools.append(False)

        if filter_type.upper() == "AND":
            return all(bools)
        else:
            return any(bools)

    return dataset.filter(filter_func, num_proc=4)
