#!/usr/bin/env python3
"""Download vendor documentation PDFs into organized folders under docs/."""
from __future__ import annotations

import scripts._bootstrap  # noqa: F401 — project root on sys.path for direct script runs

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from tqdm import tqdm

DOCS_ROOT = Path("docs")
from providers.config_paths import LOGS_DIR, VENDORS_JSON

LOG_FILE = LOGS_DIR / "download_log.txt"

DELAY_BETWEEN_DOWNLOADS = 2
MAX_RETRIES = 3
RETRY_DELAY = 5
REQUEST_TIMEOUT = 120

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

VENDORS: dict[str, list[str]] = {
    "siemens": [
        "https://support.industry.siemens.com/cs/document/109745954",
        "https://support.industry.siemens.com/cs/document/109478084",
    ],
    "rockwell": [
        "https://literature.rockwellautomation.com/idc/groups/literature/documents/um/1756-um001_-en-p.pdf",
        "https://literature.rockwellautomation.com/idc/groups/literature/documents/um/2080-um002_-en-e.pdf",
    ],
    "schneider": [
        "https://www.se.com/ww/en/download/document/EIO0000001462",
        "https://www.se.com/ww/en/download/document/EIO0000000067",
    ],
    "sick": [
        "https://www.sick.com/media/docs/7/07/507/Operating_instructions_S3000_en_IM0001507.PDF",
        "https://www.sick.com/media/pdf/3/03/903/dataSheet_WL12G-3B2531_1042903_en.pdf",
    ],
    "festo": [
        "https://www.festo.com/net/SupportPortal/Files/43484/P_BE-SPC200_EN.pdf",
        "https://www.festo.com/net/SupportPortal/Files/391048/8076869e2.pdf",
    ],
    "conveylinx": [
        "https://www.pulseroller.com/wp-content/uploads/2019/09/ConveyLinx-Ai2-User-Manual.pdf",
        "https://www.pulseroller.com/wp-content/uploads/2020/01/ConveyLinx-ERSC-User-Manual.pdf",
    ],
    "abb": [
        "https://library.abb.com/d/9AKK107991A9465",
        "https://library.abb.com/d/3AUA0000094478",
    ],
    "kuka": [
        "https://www.kuka.com/en/services/downloads/robotics/robot-programming/kuka-system-software",
        "https://xpert.kuka.com/service-express/portal/api/public/documents/download/KST_AUT_2_en.pdf",
    ],
    "ur": [
        "https://s3-eu-west-1.amazonaws.com/ur-support-site/105198/scriptManual_en_Global.pdf",
        "https://s3-eu-west-1.amazonaws.com/ur-support-site/105197/userManual_en_Global.pdf",
    ],
    "sew": [
        "https://download.sew-eurodrive.com/download/pdf/16752626.pdf",
        "https://download.sew-eurodrive.com/download/pdf/28982899.pdf",
    ],
}

CUSTOM_FOLDERS = [
    "mitsubishi",
    "mechmind",
    "pekat",
    "zivid",
    "lmi",
    "general",
]

CONTENT_TYPE_EXTENSIONS = {
    "application/pdf": ".pdf",
    "application/octet-stream": ".pdf",
    "text/html": ".html",
    "application/zip": ".zip",
}


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("download_docs")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
    return logger


def load_vendors_json() -> tuple[dict[str, list[str]], list[str]]:
    if not VENDORS_JSON.exists():
        return {}, []
    try:
        data = json.loads(VENDORS_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}, []
    vendors = data.get("vendors", {})
    folders = data.get("custom_folders", [])
    return vendors, folders


def save_vendors_json(vendors: dict[str, list[str]], custom_folders: list[str]) -> None:
    payload = {
        "vendors": {k: list(v) for k, v in sorted(vendors.items())},
        "custom_folders": sorted(set(custom_folders)),
    }
    VENDORS_JSON.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def merge_vendors(
    base: dict[str, list[str]], extra: dict[str, list[str]]
) -> dict[str, list[str]]:
    merged = {vendor: list(urls) for vendor, urls in base.items()}
    for vendor, urls in extra.items():
        bucket = merged.setdefault(vendor, [])
        for url in urls:
            if url not in bucket:
                bucket.append(url)
    return merged


def sanitize_filename(name: str) -> str:
    cleaned = unquote(name).strip().strip('"').strip("'")
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip("._")
    return cleaned or "download"


def url_hash(url: str) -> str:
    import hashlib

    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]


def guess_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    segment = unquote(parsed.path.rstrip("/").split("/")[-1])
    if segment and "." in segment:
        return sanitize_filename(segment)
    if segment:
        return sanitize_filename(f"{segment}.pdf")
    return sanitize_filename(f"download_{url_hash(url)}.pdf")


def filename_from_content_disposition(header: str) -> str | None:
    if not header:
        return None
    match = re.search(
        r"filename\*=UTF-8''([^;]+)",
        header,
        flags=re.IGNORECASE,
    )
    if match:
        return sanitize_filename(match.group(1))
    match = re.search(
        r'filename="?([^";]+)"?',
        header,
        flags=re.IGNORECASE,
    )
    if match:
        return sanitize_filename(match.group(1))
    return None


def extension_from_content_type(content_type: str) -> str:
    mime = content_type.split(";")[0].strip().lower()
    return CONTENT_TYPE_EXTENSIONS.get(mime, "")


def resolve_filename(url: str, response: requests.Response) -> str:
    cd_name = filename_from_content_disposition(
        response.headers.get("Content-Disposition", "")
    )
    if cd_name:
        return cd_name

    url_name = guess_filename_from_url(url)
    if "." in Path(url_name).suffix:
        return url_name

    ext = extension_from_content_type(response.headers.get("Content-Type", ""))
    if ext and not url_name.lower().endswith(ext):
        return sanitize_filename(f"{Path(url_name).stem}{ext}")
    return url_name


def unique_dest_path(folder: Path, filename: str, url: str) -> Path:
    dest = folder / filename
    if not dest.exists():
        return dest
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".bin"
    return folder / sanitize_filename(f"{stem}_{url_hash(url)}{suffix}")


def ensure_folders(vendors: dict[str, list[str]], custom_folders: list[str]) -> None:
    all_folders = sorted(set(vendors.keys()) | set(custom_folders))
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    for folder_name in all_folders:
        (DOCS_ROOT / folder_name).mkdir(parents=True, exist_ok=True)


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def download_one(
    vendor: str,
    url: str,
    folder: Path,
    session: requests.Session,
    logger: logging.Logger,
) -> tuple[str, str | None, str | None]:
    """Return status: downloaded | skipped | failed, plus filename and error."""
    primary_name = guess_filename_from_url(url)
    primary_path = folder / primary_name
    if primary_path.exists():
        logger.info("SKIP %s %s -> %s", vendor, url, primary_path)
        return "skipped", primary_name, None

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with session.get(
                url,
                stream=True,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                filename = resolve_filename(url, response)
                dest = unique_dest_path(folder, filename, url)

                if dest.exists():
                    logger.info("SKIP %s %s -> %s", vendor, url, dest)
                    return "skipped", dest.name, None

                total = int(response.headers.get("Content-Length", 0) or 0)
                desc = f"{vendor}: {dest.name[:48]}"
                with open(dest, "wb") as handle, tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=desc,
                    leave=False,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1024 * 64):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        bar.update(len(chunk))

            logger.info("OK %s %s -> %s", vendor, url, dest)
            return "downloaded", dest.name, None
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "FAIL attempt %s/%s %s %s: %s",
                attempt,
                MAX_RETRIES,
                vendor,
                url,
                exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    logger.error("FAIL %s %s: %s", vendor, url, last_error)
    return "failed", None, last_error


def run_downloads(
    vendors: dict[str, list[str]],
    custom_folders: list[str],
    vendor_filter: str | None = None,
) -> dict[str, list[tuple[str, str]]]:
    logger = setup_logging()
    ensure_folders(vendors, custom_folders)

    selected = vendors
    if vendor_filter:
        if vendor_filter not in vendors:
            print(f"Unknown vendor: {vendor_filter}")
            print(f"Available: {', '.join(sorted(vendors))}")
            sys.exit(1)
        selected = {vendor_filter: vendors[vendor_filter]}

    stats = {"downloaded": 0, "failed": 0, "skipped": 0}
    failures: list[tuple[str, str]] = []

    session = create_session()
    jobs = [
        (vendor, url)
        for vendor, urls in sorted(selected.items())
        for url in urls
    ]

    for index, (vendor, url) in enumerate(jobs):
        if index > 0:
            time.sleep(DELAY_BETWEEN_DOWNLOADS)

        folder = DOCS_ROOT / vendor
        status, _, error = download_one(vendor, url, folder, session, logger)
        stats[status] += 1
        if status == "failed":
            failures.append((vendor, url))

    print("\nDownload complete:")
    print(f"  ✓ {stats['downloaded']} downloaded")
    print(f"  ✗ {stats['failed']} failed (see {LOG_FILE})")
    print(f"  ~ {stats['skipped']} skipped (already exist)")

    if failures:
        print("\nFailed:")
        for vendor, url in failures:
            print(f"  - {vendor}: {url}")

    return stats


def list_vendors(vendors: dict[str, list[str]], custom_folders: list[str]) -> None:
    ensure_folders(vendors, custom_folders)
    print("Configured vendors:")
    for vendor in sorted(vendors):
        print(f"  {vendor}: {len(vendors[vendor])} URL(s)")
    extra_folders = sorted(set(custom_folders) - set(vendors.keys()))
    if extra_folders:
        print("\nCustom folders (no URLs configured):")
        for folder in extra_folders:
            print(f"  {folder}")


def add_folder(folder_name: str, custom_folders: list[str]) -> list[str]:
    normalized = folder_name.strip().lower()
    if not normalized:
        raise ValueError("Folder name cannot be empty")
    updated = sorted(set(custom_folders) | {normalized})
    (DOCS_ROOT / normalized).mkdir(parents=True, exist_ok=True)
    return updated


def add_vendor(
    vendor_name: str,
    urls: list[str],
    vendors: dict[str, list[str]],
) -> dict[str, list[str]]:
    normalized = vendor_name.strip().lower()
    if not normalized:
        raise ValueError("Vendor name cannot be empty")
    bucket = list(vendors.get(normalized, []))
    for url in urls:
        if url not in bucket:
            bucket.append(url)
    vendors[normalized] = bucket
    return vendors


def build_config() -> tuple[dict[str, list[str]], list[str]]:
    json_vendors, json_folders = load_vendors_json()
    vendors = merge_vendors(VENDORS, json_vendors)
    custom_folders = sorted(set(CUSTOM_FOLDERS) | set(json_folders))
    return vendors, custom_folders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download vendor documentation into docs/<vendor>/ folders."
    )
    parser.add_argument(
        "--vendor",
        help="Download only the specified vendor",
    )
    parser.add_argument(
        "--add-folder",
        metavar="NAME",
        help="Create a vendor folder without downloading",
    )
    parser.add_argument(
        "--add-vendor",
        metavar="NAME",
        help="Add a vendor with URLs and download immediately",
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        default=[],
        help="URLs to associate with --add-vendor",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured vendors and URL counts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vendors, custom_folders = build_config()
    json_vendors, json_folders = load_vendors_json()

    if args.list:
        list_vendors(vendors, custom_folders)
        return

    if args.add_folder:
        custom_folders = add_folder(args.add_folder, custom_folders)
        save_vendors_json(json_vendors, custom_folders)
        print(f"Added folder: {args.add_folder.strip().lower()}")
        return

    if args.add_vendor:
        if not args.urls:
            print("--add-vendor requires at least one --urls value")
            sys.exit(1)
        vendors = add_vendor(args.add_vendor, args.urls, vendors)
        json_vendors = merge_vendors(json_vendors, {args.add_vendor.strip().lower(): args.urls})
        save_vendors_json(json_vendors, custom_folders)
        run_downloads(vendors, custom_folders, vendor_filter=args.add_vendor.strip().lower())
        return

    run_downloads(vendors, custom_folders, vendor_filter=args.vendor)


if __name__ == "__main__":
    main()
