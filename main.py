"""Pipeline principal du projet QSVM pour classification audio.

Usage:
    python main.py --config config/paths.yaml --mode train
    python main.py --config config/paths.yaml --mode predict --audio path/to/audio.wav
    python main.py --config config/paths.yaml --mode evaluate
"""

import argparse
from src.pipeline import AudioQSVMpipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QSVM Audio Classification")
    parser.add_argument("--config", required=True, help="Path to paths config file (YAML)")
    parser.add_argument("--mode", choices=["train", "predict", "evaluate"], required=True)
    parser.add_argument("--audio", help="Audio file for prediction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = AudioQSVMpipeline(config_path=args.config)

    if args.mode == "train":
        pipeline.train()
    elif args.mode == "predict" and args.audio:
        result = pipeline.predict(args.audio)
        print(f"Prediction: {result}")
    elif args.mode == "evaluate":
        pipeline.evaluate()
    else:
        raise SystemExit("Invalid argument combination: for predict mode, --audio is required.")


if __name__ == "__main__":
    main()
