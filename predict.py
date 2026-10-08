"""Predict an insurance charge from the command line.

Example:
    python predict.py --age 45 --sex male --bmi 30 --children 2 --smoker yes --region southeast
"""

import argparse
import sys

from predictor import predict_charge


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate a yearly medical insurance charge (USD).")
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--sex", choices=["female", "male"], required=True)
    parser.add_argument("--bmi", type=float, required=True)
    parser.add_argument("--children", type=int, required=True)
    parser.add_argument("--smoker", choices=["no", "yes"], required=True)
    parser.add_argument("--region", choices=["northeast", "northwest", "southeast", "southwest"], required=True)
    args = parser.parse_args()

    try:
        charge = predict_charge(args.age, args.sex, args.bmi, args.children, args.smoker, args.region)
    except ValueError as err:
        sys.exit(f"Invalid input: {err}")

    print(f"Estimated yearly charge: ${charge:,.2f} (USD, model estimate)")


if __name__ == "__main__":
    main()
