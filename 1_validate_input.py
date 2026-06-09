"""
Step 1 — Validate that the local 10-K file exists in the data folder.

Does not download from EDGAR. The file must already exist as one of:
- data/sample_10k.txt  (when USE_SAMPLE=True)
- data/<company>_10k.txt  (preferred, when USE_SAMPLE=False)
- data/<company>_10k.pdf  (fallback, when USE_SAMPLE=False)

Usage:
    python 1_validate_input.py
    python 1_validate_input.py --company Microsoft
"""
import argparse
import os

import config


def get_local_file_path(company_name: str) -> str:
    """Return the path of the first matching file: sample → .txt → .pdf."""
    if config.USE_SAMPLE:
        return config.SAMPLE_DOC_PATH
    base = company_name.lower()
    txt_path = os.path.join(config.DATA_DIR, f"{base}_10k.txt")
    pdf_path = os.path.join(config.DATA_DIR, f"{base}_10k.pdf")
    if os.path.exists(txt_path):
        return txt_path
    if os.path.exists(pdf_path):
        return pdf_path
    # Neither found — return the txt path so the error message is actionable
    return txt_path


def fetch_and_save(company_name: str) -> str:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    path = get_local_file_path(company_name)

    if not os.path.exists(path):
        base = company_name.lower()
        raise FileNotFoundError(
            f"No 10-K file found for {company_name}. "
            f"Place either data/{base}_10k.txt or data/{base}_10k.pdf in the data folder and rerun."
        )

    print(f"Found local 10-K file: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--company", default=config.TARGET_COMPANY,
                        choices=list(config.COMPANY_CIKS.keys()))
    args = parser.parse_args()

    path = fetch_and_save(args.company)
    print(f"\nDone. Run next: python 2_chunk_and_embed.py --company {args.company}")


if __name__ == "__main__":
    main()
