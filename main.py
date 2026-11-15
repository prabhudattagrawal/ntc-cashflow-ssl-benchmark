import argparse
from src.experiment import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the NTC-FinRep experiment")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-applicants", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.n_applicants, args.epochs)
