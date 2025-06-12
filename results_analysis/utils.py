from dataclasses import dataclass
import re
from typing import Optional
from pathlib import Path
import os


@dataclass
class LogMetrics:
    timestamp: Optional[str] = None
    method_details: Optional[str] = None
    explained_variance_first_2: Optional[tuple[float, float]] = None
    total_explained_variance_100_comps: Optional[float] = None
    components_for_90_variance: Optional[int] = None
    wall_time_s: Optional[float] = None
    cpu_time_user_s: Optional[float] = None
    cpu_time_sys_s: Optional[float] = None
    cpu_time_total_s: Optional[float] = None
    gpu_kernel_time_s: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_model: Optional[str] = None
    gpu_driver: Optional[str] = None
    points: Optional[int] = None
    original_dims: Optional[int] = None
    pca_dims_saved: Optional[int] = None
    hostname: Optional[str] = None


@dataclass
class ExperimentResult:
    method: str
    architecture: str
    dataset_type: str
    metrics: LogMetrics
    precomputed_pca: bool
    original_method_arch_string: str


def parse_log_metrics(log_content: str) -> LogMetrics:
    data = LogMetrics()
    for line in log_content.splitlines():
        line = line.strip()
        if not line:
            continue

        if data.timestamp is None and line.startswith("["):
            match_header = re.match(r"\[(.*?)\]\s*(.*)", line)
            if match_header:
                data.timestamp = match_header.group(1)
                data.method_details = match_header.group(2)
                continue

        m = re.search(r"Wall time: ([\d.]+) s", line)
        if m:
            data.wall_time_s = float(m.group(1))
            continue

        m = re.search(
            r"CPU times: user ([\d.]+) s, sys ([\d.]+) s, total ([\d.]+) s", line
        )
        if m:
            data.cpu_time_user_s = float(m.group(1))
            data.cpu_time_sys_s = float(m.group(2))
            data.cpu_time_total_s = float(m.group(3))
            continue

        m = re.search(
            r"Explained variance \(first 2\): ([\d.]+)%, ([\d.]+)%", line
        )
        if m:
            data.explained_variance_first_2 = (
                float(m.group(1)),
                float(m.group(2)),
            )
            continue

        m = re.search(
            r"Total explained variance \((\d+) comps\): ([\d.]+)%", line
        )
        if m:
            data.total_explained_variance_100_comps = float(m.group(2))
            continue

        m = re.search(r"Components needed for >=90% variance: (\d+)", line)
        if m:
            data.components_for_90_variance = int(m.group(1))
            continue

        m = re.search(r"GPU kernel time: ([\d.]+) s", line)
        if m:
            data.gpu_kernel_time_s = float(m.group(1))
            continue

        m = re.search(r"GPU memory used: ([\d.]+) MB / ([\d.]+) MB", line)
        if m:
            data.gpu_memory_used_mb = float(m.group(1))
            data.gpu_memory_total_mb = float(m.group(2))
            continue

        m = re.search(r"GPU model: (.*?), Driver: (.*)", line)
        if m:
            data.gpu_model = m.group(1).strip()
            data.gpu_driver = m.group(2).strip()
            continue

        m = re.search(
            r"Points: (\d+), Original dims: (\d+), PCA dims saved: (\d+)", line
        )
        if m:
            data.points = int(m.group(1))
            data.original_dims = int(m.group(2))
            data.pca_dims_saved = int(m.group(3))
            continue

        m = re.search(r"Hostname: (.*)", line)
        if m:
            data.hostname = m.group(1).strip()
            continue
    return data


def match(parampacmap_substr: str, methods: list[str]) -> str | bool:
    parampacmap_substr = parampacmap_substr.lower()
    for m in methods:
        m_lower = m.lower()
        if m_lower in parampacmap_substr:
            intersection = "".join([ch for ch in m_lower if ch in m])
            return intersection
    return False


def parse_experiments(
    methods: list[str],
    architectures: list[str],
    dataset_options: list[str],
    RESULTS_DIR: Path,
    TEST_DIR_TO_SKIP: str,
) -> tuple[list, list]:
    log_metrics_data = []
    all_experiment_results = []

    for time_file_path in RESULTS_DIR.rglob("*_time.txt"):
        if TEST_DIR_TO_SKIP in str(time_file_path):
            continue
        if time_file_path.is_file():
            # Store log metrics
            log_content = time_file_path.read_text(encoding="utf-8")
            log_metrics = parse_log_metrics(log_content)
            log_metrics_data.append(log_metrics)

            # Parse path to extract method, architecture, dataset type, and precomputed PCA
            relative_to_project_root = time_file_path.relative_to(RESULTS_DIR)
            path = os.sep + str(relative_to_project_root)
            path = path[1:]  # Remove first slash

            parts_with_os_sep = path.split(os.sep)

            method = None
            normalized = "non_normalized"
            arch = "cpu"
            precomputed_pca = False

            for part in parts_with_os_sep:
                part = part.lower()  # just to make sure

                if "_" in part:
                    original_part = part
                    parts = part.split("_")  # Output: ['pacmap', 'cpu']
                    for subpart in parts:
                        if "cmlu" in subpart:
                            arch = "gpu"
                        intersection = match(
                            subpart, methods
                        )  # paramapcmap & pacmap -> pacmap
                        if subpart in methods:
                            method = subpart
                        elif intersection:
                            method = intersection
                        elif subpart in dataset_options:
                            if subpart == "raw":
                                normalized = "non_normalized"
                            normalized = subpart
                        elif subpart in architectures:
                            arch = subpart
                        elif "pca" in subpart:
                            precomputed_pca = True

                if "cmlu" in part:
                    arch = "gpu"
                if part in methods:
                    method = part
                elif part in dataset_options:
                    if part == "raw" or part == "non-normalized":
                        part = "non_normalized"
                    normalized = part
                elif part in architectures:
                    arch = part
                elif "pca" in part:
                    precomputed_pca = True

            experiment = ExperimentResult(
                method=method,
                architecture=arch,
                dataset_type=normalized,
                metrics=log_metrics,
                precomputed_pca=precomputed_pca,
                original_method_arch_string=original_part,
            )
            all_experiment_results.append(experiment)

    return log_metrics_data, all_experiment_results
