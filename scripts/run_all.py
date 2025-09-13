import argparse
import yaml
import warnings
import os
import random
import numpy as np
from scripts.preprocess import run as run_pre
from scripts.performance import run as run_perf
from scripts.emotion import run as run_emo
from scripts.body import run as run_body
from scripts.integrate import run as run_integrate
from scripts.visualize import run as run_visualize
from scripts.build_datasets import run as run_build
from scripts.report import run as run_report

warnings.filterwarnings("ignore")  # suppress pandas / matplotlib warnings


def main(config_path, seed=None):
    # Set global seed for reproducibility
    if seed is not None:
        try:
            seed = int(seed)
        except Exception:
            seed = None
    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        print(f"Global random seed set to: {seed}")
    print(f"Loading config: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    outputs = {}

    try:
        print("1) Preprocessing...")
        outputs['preprocess'] = run_pre(config_path)
        print("Done. Outputs:", outputs['preprocess'])
    except Exception as e:
        print("Preprocessing failed:", e)
        return

    try:
        print("2) Performance analysis...")
        outputs['performance'] = run_perf(config_path)
        print("Done. Outputs:", outputs['performance'])
    except Exception as e:
        print("Performance analysis failed:", e)
        return

    try:
        print("3) Emotion analysis...")
        outputs['emotion'] = run_emo(config_path)
        print("Done. Outputs:", outputs['emotion'])
    except Exception as e:
        print("Emotion analysis failed:", e)
        return

    try:
        print("4) Body motion analysis...")
        outputs['body'] = run_body(config_path)
        print("Done. Outputs:", outputs['body'])
    except Exception as e:
        print("Body motion analysis failed:", e)
        return

    try:
        print("5) Integrated analysis...")
        outputs['integrated'] = run_integrate(config_path)
        print("Done. Outputs:", outputs['integrated'])
    except Exception as e:
        print("Integrated analysis failed:", e)
        return

    try:
        print("6) Visualizations...")
        outputs['visualize'] = run_visualize(config_path)
        print("Done. Outputs:", outputs['visualize'])
    except Exception as e:
        print("Visualizations failed:", e)
        return

    try:
        print("7) Build delivery datasets...")
        outputs['build_datasets'] = run_build(config_path)
        print("Done. Outputs:", outputs['build_datasets'])
    except Exception as e:
        print("Build datasets failed:", e)
        return

    try:
        print("8) Generate report...")
        outputs['report'] = run_report(config_path)
        print("Done. Outputs:", outputs['report'])
    except Exception as e:
        print("Report generation failed:", e)
        return

    print("Pipeline completed successfully!")
    print("Summary of outputs per step:")
    for step, out in outputs.items():
        print(f"- {step}: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--seed", required=False, help="Optional integer seed for reproducibility")
    args = parser.parse_args()
    main(args.config, seed=args.seed)
