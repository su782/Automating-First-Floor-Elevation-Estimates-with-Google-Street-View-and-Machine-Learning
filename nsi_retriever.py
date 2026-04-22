import os
import json
import time
import httpx
import pandas as pd

PIPELINE_ROOT = os.environ.get("PIPELINE_ROOT")
TRACT_ID = os.environ.get("TRACT_ID")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if PIPELINE_ROOT is None or TRACT_ID is None:
    raise RuntimeError("PIPELINE_ROOT or TRACT_ID not set.")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set.")

NSI_STRUCTURES_URL = "https://nsi.sec.usace.army.mil/nsiapi/structures"

TIGER_TRACT_URL = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/"
    "TIGERweb/tigerWMS_Census2020/MapServer/6/query"
)

GOOGLE_REVERSE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

REQUEST_TIMEOUT = 60
REVERSE_GEOCODE_SLEEP = 0.05

PREFERRED_OCCTYPE = {"RES1"}
RESIDENTIAL_PREFIXES = ("RES",)
DEDUP_BY_ADDRESS = True


def fetch_tract_geometry(geoid: str) -> dict:
    params = {
        "where": f"GEOID='{geoid}'",
        "outFields": "GEOID,NAME",
        "outSR": "4326",
        "f": "geojson",
    }
    r = httpx.get(TIGER_TRACT_URL, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    features = r.json().get("features", [])
    if not features:
        raise ValueError(f"No tract found for {geoid}")
    return features[0]


def fetch_nsi_structures(geojson_feature: dict) -> list[dict]:
    body = {"type": "FeatureCollection", "features": [geojson_feature]}

    features = []
    with httpx.stream(
        "POST",
        NSI_STRUCTURES_URL,
        params={"fmt": "fs"},
        json=body,
        timeout=300,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                features.append(json.loads(line))
    return features


def feature_to_row(feat: dict) -> dict:
    props = feat.get("properties", {}) or {}
    geom = feat.get("geometry", {}) or {}
    coords = geom.get("coordinates", None)

    lng, lat = None, None
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        lng, lat = coords[0], coords[1]

    return {
        "address_nsi": props.get("address"),
        "occtype": props.get("occtype"),
        "lat": lat,
        "lng": lng,
    }


def reverse_geocode_google(lat: float, lng: float, api_key: str) -> str | None:
    params = {
        "latlng": f"{lat},{lng}",
        "key": api_key,
    }
    r = httpx.get(GOOGLE_REVERSE_GEOCODE_URL, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    if data.get("status") != "OK":
        return None

    results = data.get("results", [])
    if not results:
        return None

    return results[0].get("formatted_address")


def normalize_address_for_filename(addr: str) -> str:
    bad = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', ',']
    out = addr.strip()
    for ch in bad:
        out = out.replace(ch, "")
    out = out.replace(" ", "_")
    return out


def main():
    print("Retrieving NSI addresses for tract:", TRACT_ID)

    tract_feature = fetch_tract_geometry(TRACT_ID)
    nsi_features = fetch_nsi_structures(tract_feature)

    if not nsi_features:
        print("No NSI structures found.")
        out_csv = os.path.join(PIPELINE_ROOT, "addresses.csv")
        pd.DataFrame(columns=["address", "lat", "lng"]).to_csv(out_csv, index=False)
        print(f"Saved 0 addresses to {out_csv}")
        return

    rows = [feature_to_row(f) for f in nsi_features]
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["lat", "lng"]).copy()

    print(f"[INFO] NSI structures total = {len(df)}")

    if "occtype" in df.columns:
        print("[INFO] top occtype counts:")
        print(df["occtype"].value_counts(dropna=False).head(20))

    occtype_series = df["occtype"].fillna("").astype(str)

    # Strict single-family: keep only RES1*
    df_sf = df[occtype_series.str.startswith("RES1")].copy()
    print(f"[INFO] strict single-family (RES1*) count = {len(df_sf)}")

    if len(df_sf) == 0:
        print("[WARN] No strict single-family (RES1*) structures found in this tract.")
        out_csv = os.path.join(PIPELINE_ROOT, "addresses.csv")
        pd.DataFrame(columns=["address", "lat", "lng"]).to_csv(out_csv, index=False)

        debug_csv = os.path.join(PIPELINE_ROOT, "addresses_debug.csv")
        pd.DataFrame(columns=["address_raw", "address", "lat", "lng", "occtype"]).to_csv(debug_csv, index=False)

        print(f"Saved 0 addresses to {out_csv}")
        print(f"Debug file saved to {debug_csv}")
        return

    df_use = df_sf
    print("[INFO] Using strict single-family RES1* only.")
    addresses = []
    total = len(df_use)

    for i, row in enumerate(df_use.itertuples(index=False), 1):
        lat = float(row.lat)
        lng = float(row.lng)

        google_addr = reverse_geocode_google(lat, lng, GOOGLE_API_KEY)
        time.sleep(REVERSE_GEOCODE_SLEEP)

        if google_addr:
            addr = google_addr
        elif pd.notna(row.address_nsi) and str(row.address_nsi).strip():
            addr = str(row.address_nsi).strip()
        else:
            addr = f"{lat:.6f}_{lng:.6f}"

        addresses.append({
            "address_raw": addr,
            "address": normalize_address_for_filename(addr),
            "lat": lat,
            "lng": lng,
            "occtype": row.occtype,
        })

        if i % 50 == 0 or i == total:
            print(f"[INFO] reverse geocoded {i}/{total}")

    out_df = pd.DataFrame(addresses)

    if DEDUP_BY_ADDRESS and len(out_df) > 0:
        before = len(out_df)
        out_df = out_df.drop_duplicates(subset=["address"]).copy()
        print(f"[INFO] dedup by address: {before} -> {len(out_df)}")

    out_csv = os.path.join(PIPELINE_ROOT, "addresses.csv")
    out_df[["address", "lat", "lng"]].to_csv(out_csv, index=False)

    debug_csv = os.path.join(PIPELINE_ROOT, "addresses_debug.csv")
    out_df.to_csv(debug_csv, index=False)

    print(f"Saved {len(out_df)} addresses to {out_csv}")
    print(f"Debug file saved to {debug_csv}")


if __name__ == "__main__":
    main()