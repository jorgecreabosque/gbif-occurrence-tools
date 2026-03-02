from pygbif import occurrences
import pandas as pd
import simplekml
import geopandas as gpd
from shapely.geometry import Point
from pyogrio import write_dataframe

# ============================================================
# gbif_sp_occurrence_vii.py
# Descarga ocurrencias GBIF (Chile) para especies objetivo,
# filtra registros asociados a Región del Maule (VII) usando
# stateProvince, y exporta CSV + resumen + KMZ + GPKG.
# ============================================================

# =========================
# Config
# =========================
SPECIES_LIST = [
    "Crinodendron patagua",
    "Citronella mucronata",
    "Nothofagus alessandrii",
    "Neoporteria castanea"
]

COUNTRY = "CL"
LIMIT = 300
REGION_TEXT = "maule"  # filtro por texto en stateProvince (case-insensitive)

# Outputs (con nombre consistente al script)
OUT_CSV = "gbif_sp_occurrence_vii_maule.csv"
OUT_KMZ = "gbif_sp_occurrence_vii_maule.kmz"
OUT_SUMMARY_CSV = "gbif_sp_occurrence_vii_resumen.csv"
OUT_SUMMARY_TXT = "gbif_sp_occurrence_vii_resumen.txt"
OUT_GPKG = "gbif_sp_occurrence_vii_maule.gpkg"
GPKG_LAYER = "gbif_vii_maule"

# Iconos KML (distintos por especie)
ICON_BY_SPECIES = {
    "Crinodendron patagua": "http://maps.google.com/mapfiles/kml/paddle/grn-circle.png",
    "Citronella mucronata": "http://maps.google.com/mapfiles/kml/paddle/red-circle.png",
    "Nothofagus alessandrii": "http://maps.google.com/mapfiles/kml/paddle/blu-circle.png",
    "Neoporteria castanea": "http://maps.google.com/mapfiles/kml/paddle/pink-circle.png",
}


# =========================
# GBIF pagination
# =========================
def fetch_all_occurrences(scientific_name: str) -> list[dict]:
    offset = 0
    out = []

    while True:
        res = occurrences.search(
            scientificName=scientific_name,
            country=COUNTRY,
            hasCoordinate=True,
            limit=LIMIT,
            offset=offset,
        )

        batch = res.get("results", [])
        print(f"{scientific_name} | offset={offset} | nuevos={len(batch)}")
        out.extend(batch)

        if res.get("endOfRecords", True) or len(batch) == 0:
            break

        offset += LIMIT

    return out


# =========================
# Build table
# =========================
def build_dataframe(records: list[dict], species_name: str) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "species": species_name,
                "gbifID": r.get("gbifID"),
                "occurrenceID": r.get("occurrenceID"),
                "eventDate": r.get("eventDate"),
                "year": r.get("year"),
                "month": r.get("month"),
                "day": r.get("day"),
                "lat": r.get("decimalLatitude"),
                "lon": r.get("decimalLongitude"),
                "locality": r.get("locality"),
                "municipality": r.get("municipality"),
                "stateProvince": r.get("stateProvince"),
                "country": r.get("country"),
                "basisOfRecord": r.get("basisOfRecord"),
                "institutionCode": r.get("institutionCode"),
                "collectionCode": r.get("collectionCode"),
                "datasetKey": r.get("datasetKey"),
                "publisher": r.get("publisher"),
                "recordedBy": r.get("recordedBy"),
                "identifiedBy": r.get("identifiedBy"),
                "license": r.get("license"),
                "coordinateUncertaintyInMeters": r.get("coordinateUncertaintyInMeters"),
                "elevation": r.get("elevation"),
                "habitat": r.get("habitat"),
                "occurrenceStatus": r.get("occurrenceStatus"),
                "issues": ";".join(r.get("issues", [])) if isinstance(r.get("issues"), list) else r.get("issues"),
            }
        )
    return pd.DataFrame(rows)


def filter_region(df: pd.DataFrame) -> pd.DataFrame:
    # Filtro por texto en stateProvince (metadato administrativo)
    return df[df["stateProvince"].astype(str).str.lower().str.contains(REGION_TEXT, na=False)].copy()


# =========================
# Summary
# =========================
def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            [{
                "species": "",
                "n_records": 0,
                "year_min": None,
                "year_max": None,
                "n_unique_localities": 0,
                "n_unique_municipalities": 0,
                "top_municipality": None,
                "top_municipality_n": 0,
                "n_missing_eventDate": 0,
                "n_missing_locality": 0,
            }]
        )

    def top_value(series: pd.Series):
        s = series.dropna().astype(str).str.strip()
        s = s[s != ""]
        if s.empty:
            return (None, 0)
        vc = s.value_counts()
        return (vc.index[0], int(vc.iloc[0]))

    out = []
    for sp, g in df.groupby("species", dropna=False):
        year_min = pd.to_numeric(g["year"], errors="coerce").min()
        year_max = pd.to_numeric(g["year"], errors="coerce").max()

        top_muni, top_muni_n = top_value(g["municipality"])

        out.append(
            {
                "species": sp,
                "n_records": int(len(g)),
                "year_min": int(year_min) if pd.notna(year_min) else None,
                "year_max": int(year_max) if pd.notna(year_max) else None,
                "n_unique_localities": int(g["locality"].dropna().nunique()),
                "n_unique_municipalities": int(g["municipality"].dropna().nunique()),
                "top_municipality": top_muni,
                "top_municipality_n": top_muni_n,
                "n_missing_eventDate": int(g["eventDate"].isna().sum()),
                "n_missing_locality": int(g["locality"].isna().sum()),
            }
        )

    return pd.DataFrame(out).sort_values(by=["n_records", "species"], ascending=[False, True])


def write_summary_txt(summary_df: pd.DataFrame, path: str) -> None:
    lines = []
    lines.append("RESUMEN GBIF — Región del Maule (VII) (filtro por stateProvince contiene 'Maule')\n")
    if summary_df.empty or (len(summary_df) == 1 and summary_df.iloc[0].get("n_records", 0) == 0):
        lines.append("No se encontraron registros con el filtro aplicado.\n")
    else:
        total = int(summary_df["n_records"].sum())
        lines.append(f"Total registros (Maule): {total}\n")
        lines.append("Por especie:\n")
        for _, row in summary_df.iterrows():
            lines.append(
                f"- {row['species']}: {row['n_records']} registros | "
                f"años {row.get('year_min', 'NA')}–{row.get('year_max', 'NA')} | "
                f"municipios únicos {row['n_unique_municipalities']} | "
                f"top municipio: {row.get('top_municipality', 'NA')} ({row.get('top_municipality_n', 0)})"
            )
        lines.append("\nArchivos generados:\n")
        lines.append(f"- {OUT_CSV}\n- {OUT_SUMMARY_CSV}\n- {OUT_KMZ}\n- {OUT_GPKG}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# KMZ with per-species icons
# =========================
def build_kmz(df: pd.DataFrame, kmz_path: str) -> None:
    kml = simplekml.Kml()

    style_by_species = {}
    for sp in df["species"].dropna().unique():
        st = simplekml.Style()
        st.iconstyle.icon.href = ICON_BY_SPECIES.get(
            sp,
            "http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png"
        )
        st.iconstyle.scale = 1.1
        style_by_species[sp] = st

    df_sorted = df.sort_values(["species", "year", "month", "day"], na_position="last")
    for sp, g in df_sorted.groupby("species"):
        folder = kml.newfolder(name=sp)

        for _, row in g.dropna(subset=["lat", "lon"]).iterrows():
            name = f"{sp} | {row.get('year','') or ''} | gbifID={row.get('gbifID','') or ''}"
            pnt = folder.newpoint(name=name, coords=[(float(row["lon"]), float(row["lat"]))])

            desc_fields = [
                "eventDate", "locality", "municipality", "stateProvince",
                "basisOfRecord", "gbifID", "datasetKey",
                "coordinateUncertaintyInMeters", "elevation", "habitat",
                "license", "issues"
            ]
            desc = []
            for col in desc_fields:
                val = row.get(col, "")
                if pd.notna(val) and str(val).strip() != "":
                    desc.append(f"<b>{col}</b>: {val}")
            pnt.description = "<br>".join(desc)

            pnt.style = style_by_species.get(sp)

    kml.savekmz(kmz_path)


# =========================
# GeoPackage export (robusto para QGIS)
# =========================
def export_gpkg(df: pd.DataFrame, gpkg_path: str, layer: str) -> None:
    if df.empty:
        print("No se creó GPKG porque no hay registros para Maule con el filtro aplicado.")
        return

    df_clean = df.copy()

    # Forzar numéricos en lat/lon
    df_clean["lat"] = pd.to_numeric(df_clean["lat"], errors="coerce")
    df_clean["lon"] = pd.to_numeric(df_clean["lon"], errors="coerce")

    # Convertir objetos a texto (evita tipos mixtos que rompen QGIS)
    for col in df_clean.columns:
        if col not in ("lat", "lon") and df_clean[col].dtype == "object":
            df_clean[col] = df_clean[col].astype(str)

    # Geometría
    df_clean = df_clean.dropna(subset=["lon", "lat"]).copy()
    geometry = [Point(xy) for xy in zip(df_clean["lon"], df_clean["lat"])]
    gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:4326")

    # Escritura robusta para QGIS
    write_dataframe(gdf, gpkg_path, layer=layer, driver="GPKG")
    print("GeoPackage creado:", gpkg_path)


# =========================
# Main
# =========================
def main():
    all_frames = []

    for sp in SPECIES_LIST:
        print(f"\nBuscando: {sp}")
        recs = fetch_all_occurrences(sp)
        df_sp = build_dataframe(recs, sp)
        all_frames.append(df_sp)

    df_all = pd.concat(all_frames, ignore_index=True)

    # Filtrar Región del Maule (VII) por texto en stateProvince
    df_vii = filter_region(df_all)

    # CSV principal
    df_vii.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nCSV creado: {OUT_CSV}")
    print(f"Registros Maule (VII): {len(df_vii)}")

    # Resumen
    summary = make_summary(df_vii)
    summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8")
    write_summary_txt(summary, OUT_SUMMARY_TXT)
    print(f"Resumen CSV creado: {OUT_SUMMARY_CSV}")
    print(f"Resumen TXT creado: {OUT_SUMMARY_TXT}")

    # KMZ
    if len(df_vii) > 0:
        build_kmz(df_vii, OUT_KMZ)
        print(f"KMZ creado: {OUT_KMZ}")

    # GPKG
    export_gpkg(df_vii, OUT_GPKG, GPKG_LAYER)


if __name__ == "__main__":
    main()