import argparse
import os

from lit_pcba_utils import prepare_target_inputs


def main():
    parser = argparse.ArgumentParser(description="Prepare refs.csv and library.csv for one LIT-PCBA target")
    parser.add_argument("--target_dir", required=True, help="Path to one target folder under LIT-PCBA")
    parser.add_argument("--out_dir", required=True, help="Output folder for prepared CSV files")
    args = parser.parse_args()

    refs_csv, library_csv, meta_json = prepare_target_inputs(
        target_dir=os.path.abspath(args.target_dir),
        out_dir=os.path.abspath(args.out_dir),
    )
    print(f"Saved refs csv: {refs_csv}")
    print(f"Saved library csv: {library_csv}")
    print(f"Saved metadata: {meta_json}")


if __name__ == "__main__":
    main()
