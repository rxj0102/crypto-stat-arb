"""Rebuild data/snapshot from the public Binance archive (https://data.binance.vision).

    python scripts/fetch_data.py /path/to/empty/folder 2026-10-31

Lists every spot pair quoted in USDT, downloads its monthly files of daily candles (about
29,000 small files, a few minutes) and writes the two snapshot files. The second argument is
the last day to keep, so a rebuild can reproduce the committed snapshot exactly.
"""
from __future__ import annotations

import concurrent.futures as cf
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from statarb import data  # noqa: E402

BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILES = "https://data.binance.vision/"
PREFIX = "data/spot/monthly/klines/"


def listing(params: dict) -> str:
    pages, marker = [], ""
    while True:
        q = dict(params, **({"marker": marker} if marker else {}))
        page = urllib.request.urlopen(BUCKET + "?" + urllib.parse.urlencode(q), timeout=60).read().decode()
        pages.append(page)
        if "<IsTruncated>true</IsTruncated>" not in page:
            return "".join(pages)
        nxt = re.search(r"<NextMarker>(.*?)</NextMarker>", page)
        marker = nxt.group(1) if nxt else re.findall(r"<Key>(.*?)</Key>", page)[-1]


def files_of(symbol: str) -> list[str]:
    return [k for k in re.findall(r"<Key>(.*?)</Key>", listing({"prefix": f"{PREFIX}{symbol}/1d/"}))
            if k.endswith(".zip")]


def download(key: str, folder: Path) -> None:
    target = folder / key.rsplit("/", 1)[1]
    if target.exists() and target.stat().st_size > 0:
        return
    for attempt in range(4):
        try:
            target.write_bytes(urllib.request.urlopen(FILES + key, timeout=60).read())
            return
        except OSError:
            if attempt == 3:
                raise


def main() -> None:
    folder, last_day = Path(sys.argv[1]), sys.argv[2]
    folder.mkdir(parents=True, exist_ok=True)
    symbols = sorted(s for s in re.findall(r"klines/([A-Z0-9]+)/</Prefix>", listing({"delimiter": "/", "prefix": PREFIX}))
                     if s.endswith("USDT") and not data.excluded(s))
    with cf.ThreadPoolExecutor(32) as pool:
        keys = [k for ks in pool.map(files_of, symbols) for k in ks]
        list(pool.map(lambda k: download(k, folder), keys))
    close, vol = data.build(folder)
    data.write_snapshot(close, vol, last_day)
    print(f"{len(symbols)} symbols, {len(keys)} files, snapshot written to {data.SNAPSHOT}")


if __name__ == "__main__":
    main()
