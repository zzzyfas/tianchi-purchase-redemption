from pathlib import Path

import pandas as pd

from src.high_score_calendar import forecast_final_ensemble
from src.high_score_periodic import load_daily_segments, write_submission


def main() -> None:
    segments = load_daily_segments(Path("data/raw"))
    pred = forecast_final_ensemble(segments, "2014-09-01")
    output_path = Path("outputs/final/tc_comp_predict_table.csv")
    write_submission(pred, output_path)

    submission = pd.read_csv(output_path, header=None)
    if submission.shape != (30, 3):
        raise ValueError(f"Unexpected submission shape: {submission.shape}")
    if submission.iloc[0, 0] != 20140901 or submission.iloc[-1, 0] != 20140930:
        raise ValueError("Submission dates must cover every day in September 2014.")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values.")
    print(output_path.resolve())


if __name__ == "__main__":
    main()
