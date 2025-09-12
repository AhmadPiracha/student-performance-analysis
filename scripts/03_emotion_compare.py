import argparse
from scripts import emotion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    print("Running Emotion Analysis...")
    out = emotion.run(args.config)
    print("Outputs:", out)
