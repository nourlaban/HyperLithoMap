"""
Download Copernicus GLO-30 DEM tiles that cover a given Sentinel-2 scene.

Workflow (Copernicus Data Space Ecosystem, CDSE):
  1. Look up the S2 scene footprint via OData (by product Name).
  2. Search COP-DEM_GLO-30-DGED tiles that intersect that footprint.
  3. Download the tile ZIPs with a Keycloak access token.

Requirements
------------
  pip install requests
  Register a free account at https://dataspace.copernicus.eu
  Then export:
      export CDSE_USER="you@example.com"
      export CDSE_PASS="your_password"

Usage
-----
  python download_dem_for_s2.py
  # or with a custom scene / release:
  python download_dem_for_s2.py --scene S2B_MSIL1C_...SAFE --release 2024_1

Notes
-----
- One COP-DEM GLO-30 tile = 1° x 1°. An S2 100 km tile at ~28-30 N will be
  covered by ~4 (2x2) DEM tiles.
- If a chosen release has no coverage for the area, try an older one:
  release options in CDSE are 2019_1, 2020_1, 2021_1, 2022_1, 2023_1, 2024_1.
- After download, unzip each product; the DEM raster is under
  <PRODUCT>/DEM/*.tif (DGED = GeoTIFF, EPSG:4326).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
IDENTITY  = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
DOWNLOAD_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products({pid})/$value"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def get_token(user: str, password: str) -> str:
    """Get a CDSE Keycloak access token (valid ~10 minutes)."""
    r = requests.post(
        IDENTITY,
        data={
            "client_id": "cdse-public",
            "username": user,
            "password": password,
            "grant_type": "password",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
def get_scene_footprint(scene_name: str) -> list[tuple[float, float]]:
    """Return the outer ring of the S2 scene as [(lon, lat), ...]."""
    name = scene_name.replace(".zip", "")
    params = {"$filter": f"Name eq '{name}'", "$top": 1}
    r = requests.get(CATALOGUE, params=params, timeout=60)
    r.raise_for_status()
    products = r.json().get("value", [])
    if not products:
        raise RuntimeError(f"Scene not found on CDSE: {name}")
    ring = products[0]["GeoFootprint"]["coordinates"][0]
    return [(float(lon), float(lat)) for lon, lat in ring]


def coords_to_wkt(coords: list[tuple[float, float]]) -> str:
    return "POLYGON((" + ", ".join(f"{lon} {lat}" for lon, lat in coords) + "))"


def find_dem_tiles(wkt: str, dataset_full: str) -> list[dict]:
    """
    Find COP-DEM tiles intersecting a WKT polygon.

    dataset_full example: 'COP-DEM_GLO-30-DGED/2024_1'
    """
    dataset_filter = (
        "Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'datasetFull' and "
        f"att/OData.CSC.StringAttribute/Value eq '{dataset_full}')"
    )
    spatial_filter = f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')"
    params = {
        "$filter": f"{dataset_filter} and {spatial_filter}",
        "$top": 100,
        "$expand": "Attributes",
    }
    r = requests.get(CATALOGUE, params=params, timeout=90)
    r.raise_for_status()
    return r.json().get("value", [])


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_product(product_id: str, out_path: Path, token: str) -> None:
    """Stream a product ZIP to disk. Handles the CDSE redirect chain."""
    headers = {"Authorization": f"Bearer {token}"}
    url = DOWNLOAD_URL.format(pid=product_id)

    with requests.Session() as s:
        s.headers.update(headers)
        # Session headers persist across redirects to other CDSE hosts
        # (e.g. zipper.dataspace.copernicus.eu).
        with s.get(url, stream=True, timeout=(30, 600), allow_redirects=True) as r:
            r.raise_for_status()
            tmp = out_path.with_suffix(out_path.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
            tmp.rename(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scene",
        default="S2B_MSIL1C_20260412T082559_N0512_R021_T36RVT_20260412T120451.SAFE",
        help="Sentinel-2 product name (with or without .SAFE / .zip)",
    )
    ap.add_argument(
        "--release", default="2024_1",
        help="COP-DEM release: 2019_1, 2020_1, 2021_1, 2022_1, 2023_1, 2024_1",
    )
    ap.add_argument(
        "--product", default="COP-DEM_GLO-30-DGED",
        help="COP-DEM product family (GLO-30-DGED, GLO-30-DTED, GLO-90-DGED, ...)",
    )
    ap.add_argument("--out", default="./dem_tiles", type=Path)
    args = ap.parse_args()

    user = os.environ.get("CDSE_USER")
    pw   = os.environ.get("CDSE_PASS")
    if not user or not pw:
        print("Set CDSE_USER and CDSE_PASS env vars first "
              "(register at https://dataspace.copernicus.eu).", file=sys.stderr)
        return 1

    dataset_full = f"{args.product}/{args.release}"
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Fetching footprint of {args.scene} ...")
    coords = get_scene_footprint(args.scene)
    wkt = coords_to_wkt(coords)
    lons = [c[0] for c in coords]; lats = [c[1] for c in coords]
    print(f"      bbox lon {min(lons):.4f} .. {max(lons):.4f}, "
          f"lat {min(lats):.4f} .. {max(lats):.4f}")

    print(f"[2/4] Searching {dataset_full} tiles intersecting the footprint ...")
    tiles = find_dem_tiles(wkt, dataset_full)
    if not tiles:
        print(f"No DEM tiles found for {dataset_full}. "
              "Try an older release, e.g. --release 2023_1", file=sys.stderr)
        return 2

    print(f"      {len(tiles)} tile(s):")
    for t in tiles:
        size_mb = int(t.get("ContentLength", 0)) / 1e6
        print(f"        - {t['Name']}  ({size_mb:.1f} MB)")

    print("[3/4] Getting access token ...")
    token = get_token(user, pw)
    token_time = time.time()

    print(f"[4/4] Downloading into {args.out.resolve()} ...")
    for t in tiles:
        # Refresh the token if it is close to expiry (10 min lifetime).
        if time.time() - token_time > 8 * 60:
            token = get_token(user, pw)
            token_time = time.time()

        name = t["Name"]
        out = args.out / (name if name.endswith(".zip") else name + ".zip")
        if out.exists() and out.stat().st_size > 0:
            print(f"      skip {name} (already present)")
            continue
        print(f"      downloading {name} ...")
        t0 = time.time()
        try:
            download_product(t["Id"], out, token)
        except requests.HTTPError as e:
            print(f"        FAILED: {e}", file=sys.stderr)
            continue
        print(f"          done in {time.time()-t0:.1f}s "
              f"({out.stat().st_size/1e6:.1f} MB)")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
