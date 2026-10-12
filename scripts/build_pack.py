"""Preparation-time builder for one offline OSM trail pack (ONLINE step).

Fetches named places inside an explicitly supplied bounding box from the
OpenStreetMap Overpass API and saves a validated local pack for offline
runtime use. One request, descriptive User-Agent, server+client timeouts.

Usage:
  python3 scripts/build_pack.py --area "My Trail Area" \\
      --bbox MIN_LON,MIN_LAT,MAX_LON,MAX_LAT [--out data/packs/area.json]

The area/bbox MUST be supplied: the tool never invents a location and
never starts a download without a valid configured area. Output packs are
local files; runtime lookup never contacts Overpass (see README).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.places.overpass import OverpassError, build_query, extract_places, fetch_elements  # noqa: E402
from src.places.pack import (  # noqa: E402
    ATTRIBUTION,
    LICENSE,
    LICENSE_URL,
    SCHEMA_VERSION,
    SOURCE,
    is_valid_bbox,
)


def parse_bbox(text: str) -> dict[str, float]:
    try:
        min_lon, min_lat, max_lon, max_lat = (float(v) for v in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"bbox must be MIN_LON,MIN_LAT,MAX_LON,MAX_LAT (got {text!r})"
        ) from exc
    bbox = {"min_lat": min_lat, "min_lon": min_lon, "max_lat": max_lat, "max_lon": max_lon}
    if not is_valid_bbox(bbox):
        raise argparse.ArgumentTypeError(f"invalid bbox (ranges or min<=max): {text!r}")
    return bbox


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one offline OSM trail pack.")
    parser.add_argument("--area", required=True, help="explicit area name (never invented)")
    parser.add_argument("--bbox", required=True, type=parse_bbox, help="MIN_LON,MIN_LAT,MAX_LON,MAX_LAT")
    parser.add_argument("--out", default=None, help="output pack path (default data/packs/<slug>.json)")
    parser.add_argument(
        "--from-file",
        default=None,
        help="saved Overpass JSON response: rebuild the pack without re-querying "
        "(rate-limit friendly; still requires --area/--bbox)",
    )
    return parser.parse_args(argv)


def slugify(area: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in area.strip())
    return "-".join(filter(None, slug.split("-"))) or "area"


def build_pack(area: str, bbox: dict[str, float], out: Path, from_file: str | None = None) -> dict:
    if from_file:
        try:
            payload = json.loads(Path(from_file).read_text(encoding="utf-8"))
            elements = payload["elements"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"Cannot read Overpass response from {from_file}: {exc}")
            raise SystemExit(2) from exc
    else:
        query = build_query(bbox["min_lat"], bbox["min_lon"], bbox["max_lat"], bbox["max_lon"])
        try:
            elements = fetch_elements(query)
        except OverpassError as exc:
            print(f"Overpass fetch failed: {exc}")
            raise SystemExit(2) from exc
    places = extract_places(elements)
    if not places:
        print("Overpass returned no usable named places in this bbox; pack NOT written.")
        print("(Missing data is reported, never silently replaced with fictional data.)")
        raise SystemExit(3)
    pack = {
        "schema_version": SCHEMA_VERSION,
        "area_name": area,
        "bbox": bbox,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": SOURCE,
        "attribution": ATTRIBUTION,
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "places": places,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return pack


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out) if args.out else REPO_ROOT / "data" / "packs" / f"{slugify(args.area)}.json"
    print(f"Building pack for {args.area!r} bbox={args.bbox} -> {out}")
    pack = build_pack(args.area, args.bbox, out, args.from_file)
    print(f"Wrote {len(pack['places'])} places to {out}")
    print(f"Attribution: {ATTRIBUTION} ({LICENSE_URL}), {LICENSE}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
