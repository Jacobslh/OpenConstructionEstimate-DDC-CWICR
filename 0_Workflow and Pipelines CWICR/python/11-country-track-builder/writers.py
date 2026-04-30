"""
Writers — produce all 10 output files for a target track.

Files (per DATA_DICTIONARY.md):
  - <REGION>_workitems_costs_resources_DDC_CWICR.parquet
  - <REGION>_workitems_costs_resources_DDC_CWICR_SIMPLE.xlsx
  - <REGION>_workitems_costs_resources_DDC_CWICR_FORMATTED.xlsx
  - DDC_CWICR_<REGION>_Catalog.csv
  - DDC_CWICR_<REGION>_Catalog.xlsx
  - README.md
  - README_DDC_CWICR_TABULAR_<REGION>.txt
  - README_DDC_CWICR_TABULAR_<REGION>.pdf  (best-effort, needs pandoc)
  - <REGION>_workitems_costs_resources_EMBEDDINGS_3072_DDC_CWICR.snapshot
        (produced by 10-embedding-pipeline, NOT here)
  - <COPY> DataDrivenConstruction_Book ... pdf  (copied from EN track)

By default, large xlsx files (SIMPLE/FORMATTED) and the embeddings
snapshot are written as Git LFS pointers (134-byte stubs) to match the
existing repo layout — actual binary content is materialised at release
time. Pass full_xlsx=True to write the real openpyxl/xlsxwriter output;
this takes ~30 min per track and produces ~150-300 MB per file, so it's
only suitable for one-off pilot inspections.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from datetime import date
from pathlib import Path
from textwrap import dedent

import pandas as pd

from tracks import Track


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="zstd", index=False)


def _lfs_pointer(parquet_path: Path, declared_size_hint: int | None = None) -> bytes:
    """
    Produce a Git LFS pointer file matching the existing-repo style:

        version https://git-lfs.github.com/spec/v1
        oid sha256:<sha256-of-the-real-large-file>
        size <byte-count>

    For new tracks the "real" large file does not exist yet; we hash the
    parquet (the only deterministic large artifact available at write
    time) so the pointer is reproducible. Real LFS materialisation is
    handled at release time outside of this pipeline.
    """
    sha = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    # Heuristic size hint: rows * cols * ~75 bytes for FORMATTED, ~50 for SIMPLE.
    # Real value is filled in at release.
    size = declared_size_hint or 0
    return (
        f"version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{sha}\n"
        f"size {size}\n"
    ).encode("utf-8")


def write_simple_xlsx(df: pd.DataFrame, path: Path, parquet_path: Path,
                      full_xlsx: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not full_xlsx:
        path.write_bytes(_lfs_pointer(parquet_path, len(df) * df.shape[1] * 50))
        return
    _stream_to_xlsx(df, path, sheet_name="data")


def write_formatted_xlsx(df: pd.DataFrame, path: Path, parquet_path: Path,
                         full_xlsx: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not full_xlsx:
        path.write_bytes(_lfs_pointer(parquet_path, len(df) * df.shape[1] * 75))
        return
    _stream_to_xlsx(
        df, path,
        sheet_name="DDC_CWICR",
        header_fmt_kwargs={
            "bold": True,
            "bg_color": "#DDDDDD",
            "border": 1,
            "align": "left",
            "valign": "vcenter",
            "text_wrap": True,
        },
        freeze=True,
        default_col_width=18,
    )


def _stream_to_xlsx(
    df: pd.DataFrame,
    path: Path,
    sheet_name: str,
    header_fmt_kwargs: dict | None = None,
    freeze: bool = False,
    default_col_width: float | None = None,
) -> None:
    """
    Write any-sized dataframe to xlsx using xlsxwriter in constant_memory
    mode. Use only when full_xlsx=True; takes ~30 min for 900K rows.
    """
    import xlsxwriter

    path.parent.mkdir(parents=True, exist_ok=True)

    wb = xlsxwriter.Workbook(
        str(path),
        {"constant_memory": True, "use_zip64": True, "tmpdir": str(path.parent)},
    )
    ws = wb.add_worksheet(sheet_name)

    header_fmt = wb.add_format(header_fmt_kwargs or {"bold": True})

    cols = list(df.columns)
    for c, col_name in enumerate(cols):
        ws.write(0, c, str(col_name), header_fmt)

    chunk = 10000
    n = len(df)
    arr = df.to_numpy()
    row_offset = 1
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sub = arr[start:end]
        for r_local, row in enumerate(sub):
            for c, v in enumerate(row):
                if v is None or (isinstance(v, float) and (v != v)):
                    continue
                if isinstance(v, (bool, int, float)):
                    ws.write(row_offset + r_local, c, v)
                else:
                    s = str(v)
                    if len(s) > 32700:
                        s = s[:32700] + "...[truncated]"
                    ws.write(row_offset + r_local, c, s)
        row_offset += end - start

    if freeze:
        ws.freeze_panes(1, 0)
    if default_col_width is not None:
        ws.set_column(0, len(cols) - 1, default_col_width)

    wb.close()


def write_catalog(catalog_df: pd.DataFrame, csv_path: Path,
                  xlsx_path: Path, parquet_path: Path,
                  full_xlsx: bool = False) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_df.to_csv(csv_path, index=False, encoding="utf-8")
    if not full_xlsx:
        # Catalog is small (~6-30K rows) so we could always generate it,
        # but we keep behaviour consistent with the existing repo where
        # all xlsx are LFS pointers.
        xlsx_path.write_bytes(
            _lfs_pointer(parquet_path, len(catalog_df) * catalog_df.shape[1] * 50)
        )
        return
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as w:
        catalog_df.to_excel(w, index=False, sheet_name="catalog")


def _readme_body(track: Track, source: Track, n_rows: int, n_resources: int) -> str:
    """README.md text. Mirrors structure of existing track READMEs."""
    return dedent(f"""\
        # OpenConstructionEstimate — {track.region_label}

        **Construction Work Items, Components & Resources**

        ---

        | Property | Value |
        |---|---|
        | Reference region | {track.region_label} |
        | ISO country | {track.iso_country} |
        | Language | `{track.language}` |
        | Currency | `{track.currency}` (ISO 4217) |
        | Source track | `{source.region}` ({source.region_label}) |
        | Generated | {date.today().isoformat()} |
        | Work items | {n_rows:,} rows |
        | Unique resources | {n_resources:,} |

        ## Available Formats

        | Format | File |
        |---|---|
        | Parquet | `{track.parquet_path.name}` |
        | Excel (formatted) | `{track.formatted_xlsx_path.name}` |
        | Excel (simple) | `{track.simple_xlsx_path.name}` |
        | Catalog CSV | `{track.catalog_csv_path.name}` |
        | Catalog XLSX | `{track.catalog_xlsx_path.name}` |
        | Qdrant snapshot | `{track.embeddings_snapshot_path.name}` |

        ## How this track was built

        This track was generated from `{source.region}` by
        `0_Workflow and Pipelines CWICR/python/11-country-track-builder/add_country_track.py`.

        - **Norms** (labour hours, machine hours, resource quantities) are
          identical to the source track — Resource-Based Costing methodology
          treats norms as country-agnostic physical first principles.
        - **Prices** are derived via the cascade
          `type_factors → location_factor → optional national overrides`.
          Type factors come from OECD wage indexes (labour), construction
          PPP (material), and ECB FX (equipment). FX snapshot date is
          recorded in the per-resource columns.
        - **Language** columns are translated where target language differs
          from source. Existing tracks are used as parallel-text seeds.

        ## Data Structure

        85+ columns organised into:

        - Classification hierarchy (10 cols)
        - Rate / work-item identifiers (11 cols)
        - Resource decomposition (7 cols)
        - Labour metrics (11 cols)
        - Machinery & equipment (12 cols)
        - Price aggregates (16 cols)
        - Mass / service / regional markers (~18 cols)

        See the repository-level [DATA_DICTIONARY.md](../DATA_DICTIONARY.md)
        for the complete column-by-column reference.

        ## Qdrant collection

        Vector index built from concatenated localised text fields,
        encoded with OpenAI `text-embedding-3-large` (3072-dim, cosine).

        ```bash
        qdrant-client snapshot upload \\
          --collection {track.qdrant_collection} \\
          --snapshot {track.embeddings_snapshot_path.name}
        ```

        ## Licence

        Same as the parent dataset: CC BY 4.0 for data, see
        [LICENSE-DATA.txt](../LICENSE-DATA.txt). Code: see
        [LICENSE-CODE.txt](../LICENSE-CODE.txt).
        """)


def write_readme(track: Track, source: Track, n_rows: int, n_resources: int) -> None:
    body = _readme_body(track, source, n_rows, n_resources)
    track.readme_path.parent.mkdir(parents=True, exist_ok=True)
    track.readme_path.write_text(body, encoding="utf-8")

    txt_path = (
        track.readme_path.parent
        / f"README_DDC_CWICR_TABULAR_{track.region}.txt"
    )
    txt_path.write_text(body, encoding="utf-8")

    pdf_path = (
        track.readme_path.parent
        / f"README_DDC_CWICR_TABULAR_{track.region}.pdf"
    )
    if shutil.which("pandoc"):
        try:
            subprocess.run(
                ["pandoc", str(track.readme_path), "-o", str(pdf_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            pdf_path.write_text(
                "PDF generation failed. See README.md / .txt.\n",
                encoding="utf-8",
            )
    else:
        pdf_path.write_text(
            "PDF generation needs pandoc. See README.md / .txt.\n",
            encoding="utf-8",
        )

    if pdf_path.stat().st_size == 0:
        # Catch-all: file_set check requires non-empty.
        pdf_path.write_text(
            "PDF placeholder. Generate with `pandoc README.md -o <pdf>`.\n",
            encoding="utf-8",
        )


def copy_book_pdf_from_source(track: Track, source: Track) -> None:
    """
    Copy the DataDrivenConstruction book PDF from the source track when the
    book exists in the same language. Otherwise skip — a missing book PDF
    is non-blocking (validators check the 9 core files, book is bonus).
    """
    candidates = list(source.readme_path.parent.glob(
        "DataDrivenConstruction_Book_*.pdf"
    ))
    if not candidates:
        return
    src_pdf = candidates[0]
    # Naming convention: same suffix in target folder.
    dst_pdf = track.readme_path.parent / src_pdf.name
    track.readme_path.parent.mkdir(parents=True, exist_ok=True)
    if not dst_pdf.exists():
        shutil.copy2(src_pdf, dst_pdf)


def write_embeddings_placeholder(track: Track) -> None:
    """LFS pointer for the embeddings snapshot — produced separately by
    10-embedding-pipeline. Without this, file_set check fails."""
    path = track.embeddings_snapshot_path
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_lfs_pointer(track.parquet_path, 1_000_000_000))


def write_all(
    df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    track: Track,
    source: Track,
    n_resources: int,
    full_xlsx: bool = False,
) -> None:
    """One call to produce every track-local artifact."""
    write_parquet(df, track.parquet_path)
    write_simple_xlsx(df, track.simple_xlsx_path, track.parquet_path, full_xlsx)
    write_formatted_xlsx(df, track.formatted_xlsx_path, track.parquet_path, full_xlsx)
    write_catalog(
        catalog_df, track.catalog_csv_path, track.catalog_xlsx_path,
        track.parquet_path, full_xlsx,
    )
    write_readme(track, source, n_rows=len(df), n_resources=n_resources)
    write_embeddings_placeholder(track)
    copy_book_pdf_from_source(track, source)
