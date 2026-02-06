import os
from pathlib import Path
from rich import print
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, IntPrompt
from rich.console import Console

import reproduction
import temporal
import imgconversion

DATASETS_DIR = Path("datasets")
REPRO_DATASET_DIR = DATASETS_DIR / "malimg_paper_dataset_imgs"
MALWARE_DATASET_DIR = DATASETS_DIR / "pe-machine-learning-dataset"

REPRO_URL = "https://www.dropbox.com/scl/fi/wdb6omeiu2lg796qvt9l7/malimg_dataset.zip?rlkey=63q2xqmtlm66gilf6idd2c9k7&e=2&dl=0"
MALWARE_URL = "https://practicalsecurityanalytics.com/pe-malware-machine-learning-dataset/"

EXPERIMENTS = {
    1: "Reproduction",
    2: "Detection Baseline",
    3: "Forward",
    4: "Backward",
}

def check_and_create_datasets_dir():
    if not DATASETS_DIR.exists():
        DATASETS_DIR.mkdir()
        print(f"[green]Created directory: {DATASETS_DIR}[/green]")

def check_dataset(experiment_id: int):
    check_and_create_datasets_dir()
    
    if experiment_id == 1:
        # Reproduction dataset
        if not REPRO_DATASET_DIR.exists():
            print(f"[yellow]Required dataset not found at: {REPRO_DATASET_DIR}[/yellow]")
            print(f"Please download and extract the dataset from: [blue link={REPRO_URL}]{REPRO_URL}[/blue link]")
            print("Ensure the extracted folder matches the expected path.")
            return False
        return True
    else:
        # Malware experiments dataset
        if not MALWARE_DATASET_DIR.exists():
            print(f"[red bold]WARNING: Live Malware Required[/red bold]")
            print(f"[yellow]Required dataset not found at: {MALWARE_DATASET_DIR}[/yellow]")
            print("This experiment requires handling live malware samples.")
            print(f"Please download and extract the dataset from: [blue link={MALWARE_URL}]{MALWARE_URL}[/blue link]")
            print("Ensure the extracted folder matches the expected path.")
            return False
        return True

def convert_images():
    if (MALWARE_DATASET_DIR / 'images').exists():
        print("images directory already exists, skipping PE to PNG conversion")
        return
    imgconversion.convert(MALWARE_DATASET_DIR)

def run_experiment(experiment_name: str):
    print(f"[bold green]Starting experiment: {experiment_name}[/bold green]")
    runs = IntPrompt.ask("How many train + val runs?", default=5)
    epochs = 25
    lr = 0.001

    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)

    if experiment_name == EXPERIMENTS[1]:
        reproduction.run_reproduction(results_dir=results_dir, dataset=REPRO_DATASET_DIR, epochs=epochs, lr=lr, runs=runs)
    elif experiment_name == EXPERIMENTS[2]:
        temporal.run_temporal(results_dir=results_dir, dataset=REPRO_DATASET_DIR, epochs=epochs, lr=lr, runs=runs, experiment=temporal.Experiment.BASELINE)
    elif experiment_name == EXPERIMENTS[3]:
        temporal.run_temporal(results_dir=results_dir, dataset=REPRO_DATASET_DIR, epochs=epochs, lr=lr, runs=runs, experiment=temporal.Experiment.FORWARD)
    elif experiment_name == EXPERIMENTS[4]:
        temporal.run_temporal(results_dir=results_dir, dataset=REPRO_DATASET_DIR, epochs=epochs, lr=lr, runs=runs, experiment=temporal.Experiment.BACKWARD)
    else:
        print("Invalid Experiment")
        return


    print(f"Your CSV files can be found at: {results_dir.resolve()}")

def main():
    console = Console()
    panel = Panel(Text("Testing Temporal Robustness in Image Based Malware Detection", justify="center", style="bold blue"))
    print(panel)
    print("")

    print("[bold]Available Experiments:[/bold]")
    for key, name in EXPERIMENTS.items():
        print(f"[{key}] {name}")
    print("")

    choice = IntPrompt.ask("Select an experiment", choices=[str(k) for k in EXPERIMENTS.keys()])
    experiment_name = EXPERIMENTS[choice]

    print(f"\nSelected: [bold]{experiment_name}[/bold]")

    if not check_dataset(choice):
        print("\n[bold red]Setup incomplete. Please install the required dataset and retry.[/bold red]")
        return

    if choice != 1:
        convert_images()
    run_experiment(experiment_name)

if __name__ == "__main__":
    main()
