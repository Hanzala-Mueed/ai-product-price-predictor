from pathlib import Path

from pricer.fine_tune_data import FineTuneDataBuilder


BASE_DIR = Path(__file__).resolve().parent.parent

input_csv = BASE_DIR / "data" / "raw" / "human_out.csv"
output_dir = BASE_DIR / "data" / "processed"

builder = FineTuneDataBuilder(
    input_csv=input_csv,
    output_dir=output_dir,
)

paths = builder.build()

print("Fine-tuning files created:")
for name, path in paths.items():
    print(f"{name}: {path}")