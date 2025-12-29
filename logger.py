"""Metric logging utilities for training."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import typer


class MetricLogger:
    """
    A unified metric logger that logs metrics to:
    1. stdout (formatted as "  key: value")
    2. JSONL file (one JSON object per line) - only if log_dir is provided
    3. wandb (if configured)

    Usage:
        logger = MetricLogger(log_dir="./logs")

        # single call logs to all destinations
        logger.log({
            "loss": 0.5,
            "score": 1234,
            "pct_512": 85.0,
        }, step=100)

        # output:
        # --- Step 100 ---
        #   loss: 0.5000
        #   score: 1234
        #   pct_512: 85.0000
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        experiment_name: str = "train",
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
        wandb_config: dict[str, Any] | None = None,
    ):
        """
        Initialize the metric logger.

        Args:
            log_dir: Directory for JSONL logs. If None, file logging is disabled.
            experiment_name: Base name for log files (e.g., "train" -> "train_001.jsonl")
            use_wandb: Whether to log to Weights & Biases.
            wandb_project: W&B project name (required if use_wandb=True).
            wandb_run_name: Optional W&B run name.
            wandb_config: Config dict to log to W&B.
        """
        self.use_wandb = use_wandb
        self.wandb_run = None
        self.log_file = None
        self._file_handle = None

        # set up log directory only if provided
        if log_dir is not None:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # find unique filename
            self.log_file = self._get_unique_filename(experiment_name)
            self._file_handle = open(self.log_file, "a")
            typer.echo(f"Logging to: {self.log_file}")

        # initialize wandb if requested
        if use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config=wandb_config,
                    reinit=True,
                )
            except ImportError:
                typer.echo(
                    "Warning: wandb not installed. Install with 'pip install wandb'"
                )
                self.use_wandb = False

    def _get_unique_filename(self, base_name: str) -> Path:
        """Find a unique filename by incrementing suffix if file exists."""
        timestamp = datetime.now().strftime("%Y%m%d")
        suffix = 1

        while True:
            filename = self.log_dir / f"{base_name}_{timestamp}_{suffix:03d}.jsonl"
            if not filename.exists():
                return filename
            suffix += 1

    def _format_value(self, value: Any) -> str:
        """Format a value for console output."""
        if isinstance(value, float):
            if abs(value) < 0.01 or abs(value) >= 10000:
                return f"{value:.2e}"
            return f"{value:.2f}"
        return str(value)

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        header: str | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Log metrics to JSONL file and wandb, optionally to stdout.

        Args:
            metrics: Dictionary of metric name -> value.
            step: Optional step number.
            header: Optional header line (default: "--- Step {step} ---").
            verbose: If True, also print to stdout. If False, only log to file/wandb.
        """
        # print to stdout only if verbose
        if verbose:
            if header is not None:
                typer.echo(header)
            elif step is not None:
                typer.echo(f"--- Step {step} ---")

            for key, value in metrics.items():
                formatted = self._format_value(value)
                typer.echo(f"  {key}: {formatted}")

        # write to JSONL file only if file handle exists
        if self._file_handle is not None:
            log_entry = {"step": step, "timestamp": datetime.now().isoformat()}
            log_entry.update(metrics)

            self._file_handle.write(json.dumps(log_entry) + "\n")
            self._file_handle.flush()

        # log to wandb
        if self.use_wandb and self.wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step)

    def print(self, message: str = "") -> None:
        """Print a raw message to stdout only (not to file/wandb)."""
        typer.echo(message)

    def close(self) -> None:
        """Close file handle and finish wandb run."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

        if self.use_wandb and self.wandb_run is not None:
            import wandb

            wandb.finish()
            self.wandb_run = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

