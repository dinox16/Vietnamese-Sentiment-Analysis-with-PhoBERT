"""Dự đoán từ dòng lệnh.

Ví dụ:
    python scripts/predict.py --model YOUR_HF_USERNAME/phobert-uit-vsfc "Thầy dạy rất hay"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsfc.predictor import SentimentPredictor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("texts", nargs="+", help="Câu cần dự đoán")
    parser.add_argument("--model", required=True, help="HF Hub id hoặc thư mục model")
    args = parser.parse_args()

    predictor = SentimentPredictor(args.model)
    for result in predictor.predict(args.texts):
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
