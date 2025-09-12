import argparse
from scripts import body

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    print("Running Body Motion Analysis...")
    out = body.run(args.config)
    print("Outputs:", out)
