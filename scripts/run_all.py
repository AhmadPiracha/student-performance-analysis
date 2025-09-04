import argparse, yaml
from scripts.preprocess import run as run_pre
from scripts.performance import run as run_perf
from scripts.emotion import run as run_emo
from scripts.body import run as run_body
from scripts.integrate import run as run_integrate
from scripts.visualize import run as run_visualize
from scripts.build_datasets import run as run_build
from scripts.report import run as run_report


def main(config):
    print("1) Preprocessing...")
    print(run_pre(config))
    print("2) Performance analysis...")
    print(run_perf(config))
    print("3) Emotion analysis...")
    print(run_emo(config))
    print("4) Body motion analysis...")
    print(run_body(config))
    print("5) Integrated analysis...")
    print(run_integrate(config))
    print("6) Visualizations...")
    print(run_visualize(config))
    print("7) Build delivery datasets...")
    print(run_build(config))
    print("8) Generate report...")
    print(run_report(config))
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
