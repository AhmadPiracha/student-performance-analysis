import argparse
from scripts import preprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    print("Running Preprocessing...")
    out = preprocess.run(args.config)
    print("Outputs:", out)
