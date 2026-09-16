import io
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import PurePosixPath

import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st

st.set_page_config(page_title="Reporte NoK V2", page_icon="📊", layout="wide")
st.title("📊 Reporte NoK — V2 Optimizada")
st.caption("Procesamiento masivo de PDF con extracción paralela y generación de Excel una sola vez")

# -----------------------------------------------------------------------------
# Constantes y regex compiladas una sola vez
# -----------------------------------------------------------------------------
CHARACTERISTICS = [
    "Diametro", "Roundness", "Runout", "Concentricity",
    "Parallelism", "Taper", "Cylindricity"
]
APOYO_IDENTIFIERS = ["Aux:G", "1:A L", "1:A R", "2:C", "3:D", "4:E", "5:B"]
APOYO_MAPPING = {"1:": "1:A L", "2:": "1:A R", "3:": "2:C", "4:": "3:D", "5:": "4:E", "6:": "5:B"}
CHATTER_APOYOS_CHARS = [
    "(1) 5 - 8 UPR", "(2) 9 - 15 UPR", "(3) 16 - 23 UPR", "(4) 24 - 28 UPR",
    "(5) 29- 45 UPR", "(6) 46-70 UPR", "(7) 71-140 UPR", "(8) 141-215 UPR"
]
CHATTER_CHARS = [
    "(1) 40- 80 UPR", "(2) 81-140 UPR", "(3) 141-190 UPR",
    "(4) 191-300 UPR", "(5) 301-400 UPR"
]
LOBES_HEADERS = [
    "AngleErr", "BC-Rad'sErr", "BC-Runout", "BC-Vel./10°", "Ramp-MaxLift",
    "Nose-MaxLift", "Ramp+9°Vel/1°", "Nose-Vel./1°", "Taper", "Center-Dev"
]
THRESHOLD = 0.00008
MAX_ROWS_PER_SHEET = 1048576 - 1

RE_MAIN = re.compile(
    r"(" + "|".join(re.escape(x) for x in APOYO_IDENTIFIERS) +
    r")\s+([\s\d.\-#]+(?:\s+J[\d\-]+:\s+[\d.\-]+)?(?:\s+L[\d\-]+:\s+[\d.\-]+)?(?:\s+Tol:\s+[\d.\-]+)?)"
)
RE_LOBES = re.compile(r"^(\d+:\s[A-Z]+-\d+)\s+(.*)")
RE_NUMBERED = re.compile(r"^(\d+):\s+(.*)")
RE_NUMBERED_ONLY = re.compile(r"^\s*\d+:")


def get_resultado(value, characteristic_name=None):
    s = str(value)
    if "#" in s:
        return "Nok"
    if characteristic_name in ("(5) 301-400 UPR", "(9) 301-400 UPR"):
        try:
            if float(s.replace(",", ".")) > THRESHOLD:
                return "Nok"
        except (ValueError, TypeError):
            pass
    return "Ok"


def row(filename, piece, leva, apoyo, characteristic, measurement, area, result):
    return {
        "Nombre del archivo": filename,
        "Pieza": piece,
        "Leva": leva,
        "Apoyo": apoyo,
        "Caracteristica": characteristic,
        "Medicion": str(measurement).replace("#", ""),
        "Area": area,
        "Resultado": result,
    }


def extract_pdfs_from_zip(zip_bytes, prefix="", max_depth=5):
    result = {}

    def walk(data, current_prefix, depth):
        if depth > max_depth:
            return
        with zipfile.ZipFile(io.BytesIO(data), "r") as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = PurePosixPath(info.filename.replace("\\", "/")).name
                if not name:
                    continue
                lower = name.lower()
                if lower.endswith(".pdf"):
                    result[f"{current_prefix}{name}"] = z.read(info)
                elif lower.endswith(".zip"):
                    try:
                        walk(z.read(info), f"{current_prefix}{os.path.splitext(name)[0]}__", depth + 1)
                    except (zipfile.BadZipFile, OSError):
                        pass

    try:
        walk(zip_bytes, prefix, 0)
    except zipfile.BadZipFile as exc:
        raise ValueError("Uno de los archivos ZIP no es válido.") from exc
    return result


def normalize_uploads(files):
    uploaded = {}
    used = set()

    def unique_name(name):
        if name not in used:
            used.add(name)
            return name
        stem, ext = os.path.splitext(name)
        i = 2
        while f"{stem}_{i}{ext}" in used:
            i += 1
        candidate = f"{stem}_{i}{ext}"
        used.add(candidate)
        return candidate

    for f in files:
        data = f.getvalue()
        base = os.path.basename(f.name)
        if base.lower().endswith(".pdf"):
            uploaded[unique_name(base)] = data
        elif base.lower().endswith(".zip"):
            extracted = extract_pdfs_from_zip(data, prefix=f"{os.path.splitext(base)[0]}__")
            for name, pdf_data in extracted.items():
                uploaded[unique_name(name)] = pdf_data
    return uploaded


def parse_pdf(item):
    """Parsea un PDF. Función independiente para ejecutarse en paralelo."""
    piece, filename, pdf_bytes = item
    apoyos, levas, chatter_lobes, chatter_apoyos = [], [], [], []
    error = None

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if not pdf.pages:
                return piece, filename, apoyos, levas, chatter_lobes, chatter_apoyos, "PDF sin páginas"
            first = pdf.pages[0].extract_text() or ""
            second = pdf.pages[1].extract_text() or "" if len(pdf.pages) > 1 else ""
    except Exception as exc:
        return piece, filename, apoyos, levas, chatter_lobes, chatter_apoyos, f"{type(exc).__name__}: {exc}"

    main_start = first.find("MAIN JOURNALS:")
    lobes_start = first.find("LOBES:")
    chatter_start = first.find("CHATTER:")

    # MAIN JOURNALS
    if main_start != -1 and lobes_start != -1:
        text = first[main_start:lobes_start]
        for line in text.splitlines():
            match = RE_MAIN.match(line.strip())
            if not match:
                continue
            apoyo = match.group(1).strip()
            raw = [v for v in re.split(r"\s+", match.group(2).strip())
                   if v and not re.match(r"J\d-\d:|J\d:", v) and v != "Tol:"]
            vals = {c: None for c in CHARACTERISTICS}
            if apoyo == "1:A L" and len(raw) >= 6:
                vals.update(Diametro=raw[1], Roundness=raw[2], Runout=raw[3], Concentricity=raw[4], Taper=raw[5])
            elif apoyo == "5:B" and len(raw) >= 7:
                vals.update(Diametro=raw[1], Roundness=raw[2], Runout=raw[3], Concentricity=raw[4], Taper=raw[5], Cylindricity=raw[6])
            else:
                idx = 0
                # Equivalent to original notebook's header/index progression.
                for header in ["Measured Diameter", "Error", "Roundness", "Runout", "Concentricity", "Parallelism", "Taper", "Cylindricity"]:
                    if idx >= len(raw):
                        break
                    if header == "Measured Diameter":
                        idx += 1
                    else:
                        char = {"Error": "Diametro", "Roundness": "Roundness", "Runout": "Runout",
                                "Concentricity": "Concentricity", "Parallelism": "Parallelism",
                                "Taper": "Taper", "Cylindricity": "Cylindricity"}.get(header)
                        if char:
                            vals[char] = raw[idx]
                        idx += 1
            for char in CHARACTERISTICS:
                value = vals.get(char)
                if value is not None:
                    apoyos.append(row(filename, piece, None, apoyo, char, value, "Apoyos", get_resultado(value)))

    # LOBES
    if lobes_start != -1:
        text = first[lobes_start:chatter_start if chatter_start != -1 else len(first)]
        data_started = False
        for line in text.splitlines():
            stripped = line.strip()
            if not data_started:
                if stripped.startswith("1: EXH-1"):
                    data_started = True
                else:
                    continue
            match = RE_LOBES.match(stripped)
            if not match:
                continue
            leva, value_text = match.groups()
            raw = value_text.split()
            if len(raw) == len(LOBES_HEADERS):
                processed = raw
            elif len(raw) == 2 * len(LOBES_HEADERS):
                processed = raw[::2]
            else:
                continue
            for char, value in zip(LOBES_HEADERS, processed):
                levas.append(row(filename, piece, leva, None, char, value, "Levas", get_resultado(value)))

    # CHATTER LEVAS
    if chatter_start != -1:
        data_started = False
        for line in first[chatter_start:].splitlines():
            stripped = line.strip()
            if not data_started:
                if RE_NUMBERED_ONLY.match(stripped):
                    data_started = True
                else:
                    continue
            match = RE_NUMBERED.match(stripped)
            if not match:
                continue
            leva, value_text = match.groups()
            raw = value_text.split()
            if len(raw) != len(CHATTER_CHARS) * 4:
                continue
            n = len(CHATTER_CHARS)
            bc = raw[: n * 2][::2]
            la = raw[n * 2:][::2]
            for char, value in zip(CHATTER_CHARS, bc):
                chatter_lobes.append(row(filename, piece, leva, None, char, value, "Base Circle", get_resultado(value, char)))
            for char, value in zip(CHATTER_CHARS, la):
                chatter_lobes.append(row(filename, piece, leva, None, char, value, "Lift Area", get_resultado(value, char)))

    # CHATTER APoyos / Journals
    marker = "CHATTER: ------------------------- Journals --------------------------"
    journals_start = second.find(marker)
    if journals_start != -1:
        data_started = False
        for line in second[journals_start:].splitlines():
            stripped = line.strip()
            if not data_started:
                if RE_NUMBERED_ONLY.match(stripped):
                    data_started = True
                else:
                    continue
            match = RE_NUMBERED.match(stripped)
            if not match:
                continue
            apoyo_original, value_text = match.groups()
            raw = value_text.split()
            if len(raw) != len(CHATTER_APOYOS_CHARS) * 2:
                continue
            apoyo = APOYO_MAPPING.get(apoyo_original, apoyo_original)
            for char, value in zip(CHATTER_APOYOS_CHARS, raw[::2]):
                chatter_apoyos.append(row(filename, piece, None, apoyo, char, value, "Apoyos", get_resultado(value, char)))

    return piece, filename, apoyos, levas, chatter_lobes, chatter_apoyos, error


def build_df(rows):
    columns = ["Nombre del archivo", "Pieza", "Leva", "Apoyo", "Caracteristica", "Medicion", "Area", "Resultado"]
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df["Medicion"] = pd.to_numeric(df["Medicion"], errors="coerce")
    return df


def process_reports_parallel(uploaded, progress_callback=None, workers=None):
    """Procesamiento paralelo. El orden final conserva la numeración original de Pieza."""
    names = list(uploaded.keys())
    items = [(i, name, uploaded[name]) for i, name in enumerate(names, start=1)]
    if workers is None:
        workers = min(8, max(2, (os.cpu_count() or 4)))
        workers = min(workers, len(items))

    all_apoyos, all_levas, all_chatter_lobes, all_chatter_apoyos = [], [], [], []
    errors = []
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(parse_pdf, item) for item in items]
        for future in as_completed(futures):
            piece, filename, apoyos, levas, chatter_lobes, chatter_apoyos, error = future.result()
            all_apoyos.extend(apoyos)
            all_levas.extend(levas)
            all_chatter_lobes.extend(chatter_lobes)
            all_chatter_apoyos.extend(chatter_apoyos)
            if error:
                errors.append((piece, filename, error))
            completed += 1
            if progress_callback:
                progress_callback(completed, len(items), filename)

    # Orden estable y creación de DataFrames sólo una vez.
    key = lambda r: (r["Pieza"], r["Nombre del archivo"])
    all_apoyos.sort(key=key); all_levas.sort(key=key)
    all_chatter_lobes.sort(key=key); all_chatter_apoyos.sort(key=key)

    df_apoyos = build_df(all_apoyos)
    df_levas = build_df(all_levas)
    df_chatter_lobes = build_df(all_chatter_lobes)
    df_chatter_apoyos = build_df(all_chatter_apoyos)

    frames = [df for df in (df_apoyos, df_levas, df_chatter_lobes, df_chatter_apoyos) if not df.empty]
    df_master = pd.concat(frames, ignore_index=True) if frames else build_df([])
    df_nok = df_master.loc[df_master["Resultado"].eq("Nok")].copy() if not df_master.empty else build_df([])

    total_pieces = df_master["Pieza"].nunique() if not df_master.empty else 0
    nok_piece_ids = set(df_nok["Pieza"].unique())
    nok_pieces = len(nok_piece_ids)
    ok_pieces = total_pieces - nok_pieces
    total_chars = len(df_master)
    nok_chars = len(df_nok)
    ok_chars = total_chars - nok_chars

    rejection = nok_pieces / total_pieces * 100 if total_pieces else 0
    ftq = ok_pieces / total_pieces * 100 if total_pieces else 0
    pct_ok = ok_chars / total_chars * 100 if total_chars else 0
    pct_nok = nok_chars / total_chars * 100 if total_chars else 0

    specific = "(5) 301-400 UPR"
    spec_nok = df_nok.loc[df_nok["Caracteristica"].eq(specific)] if not df_nok.empty else df_nok
    spec_by_piece = spec_nok.groupby("Pieza")["Caracteristica"].nunique() if not spec_nok.empty else pd.Series(dtype=int)
    nok_unique_by_piece = df_nok.groupby("Pieza")["Caracteristica"].nunique() if not df_nok.empty else pd.Series(dtype=int)
    spec_pieces = set(spec_nok["Pieza"].unique()) if not spec_nok.empty else set()
    only_specific = sum(nok_unique_by_piece.get(p, 0) == 1 and p in spec_pieces for p in spec_pieces)
    specific_and_others = sum(nok_unique_by_piece.get(p, 0) > 1 for p in spec_pieces)

    df_analysis = pd.DataFrame({
        "Metrica": [
            "Total de piezas analizadas",
            "Piezas con resultado OK (todas las caracteristicas OK)",
            "Piezas con resultado NOK (al menos una caracteristica NOK)",
            "Porcentaje de Rechazo", "FTQ (First Time Quality)",
            "Total de características analizadas", "Características con resultado OK",
            "Características con resultado NOK", "% Características OK", "% Características NOK",
            f'Cantidad piezas rechazadas SOLO por "{specific}"',
            f'Cantidad piezas rechazadas por "{specific}" Y alguna otra característica'
        ],
        "Valor": [
            total_pieces, ok_pieces, nok_pieces, f"{rejection:.2f}%", f"{ftq:.2f}%",
            total_chars, ok_chars, nok_chars, f"{pct_ok:.2f}%", f"{pct_nok:.2f}%",
            int(only_specific), int(specific_and_others)
        ]
    })

    # Análisis detallado usando groupby, evitando filtros repetidos por característica.
    if df_nok.empty:
        df_detailed = pd.DataFrame()
    else:
        top_chars = df_nok["Caracteristica"].value_counts().head(15).index.tolist()
        master_top = df_master[df_master["Caracteristica"].isin(top_chars)]
        nok_top = df_nok[df_nok["Caracteristica"].isin(top_chars)]
        stats_all = master_top.groupby("Caracteristica")["Medicion"].agg(["mean", "std"])
        stats_nok = nok_top.groupby("Caracteristica")["Medicion"].agg(["mean", "std", "max", "min"])
        counts = nok_top.groupby("Caracteristica")["Pieza"].nunique()
        detailed = []
        for char in top_chars:
            if char not in stats_all.index:
                continue
            a = stats_all.loc[char]; n = stats_nok.loc[char] if char in stats_nok.index else None
            detailed.append({
                "Caracteristica": char,
                "Cantidad Piezas NOK": int(counts.get(char, 0)),
                "Promedio Total Med.": f"{a['mean']:.4f}" if pd.notna(a['mean']) else "N/A",
                "Std Total Med.": f"{a['std']:.4f}" if pd.notna(a['std']) else "N/A",
                "Promedio NOK Med.": f"{n['mean']:.4f}" if n is not None and pd.notna(n['mean']) else "N/A",
                "Std NOK Med.": f"{n['std']:.4f}" if n is not None and pd.notna(n['std']) else "N/A",
                "Max NOK Med.": f"{n['max']:.4f}" if n is not None and pd.notna(n['max']) else "N/A",
                "Min NOK Med.": f"{n['min']:.4f}" if n is not None and pd.notna(n['min']) else "N/A",
            })
        df_detailed = pd.DataFrame(detailed)

    return {
        "df_master": df_master,
        "df_nok": df_nok,
        "df_analysis": df_analysis,
        "df_detailed": df_detailed,
        "df_apoyos": df_apoyos,
        "df_levas": df_levas,
        "df_chatter_lobes": df_chatter_lobes,
        "df_chatter_apoyos": df_chatter_apoyos,
        "errors": sorted(errors),
    }


def create_excel(results):
    """Genera el Excel una sola vez. Divide Levas si excede el límite de Excel."""
    output = io.BytesIO()
    sheets = [
        ("Apoyos", results["df_apoyos"]),
        ("Levas", results["df_levas"]),
        ("chatter Levas", results["df_chatter_lobes"]),
        ("Chatter apoyos", results["df_chatter_apoyos"]),
        ("Nok", results["df_nok"]),
        ("Analisis_Resumen", results["df_analysis"]),
        ("Analisis_Detallado_NOK", results["df_detailed"]),
    ]
    part2 = None
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet, df in sheets:
            if df is None or df.empty:
                continue
            if sheet == "Levas" and len(df) > MAX_ROWS_PER_SHEET:
                df.iloc[:MAX_ROWS_PER_SHEET].to_excel(writer, index=False, sheet_name=sheet)
                part2 = io.BytesIO()
                with pd.ExcelWriter(part2, engine="openpyxl") as writer2:
                    df.iloc[MAX_ROWS_PER_SHEET:].to_excel(writer2, index=False, sheet_name="Levas_Parte_2")
            else:
                df.to_excel(writer, index=False, sheet_name=sheet)
    return output.getvalue(), part2.getvalue() if part2 else None


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Selecciona PDF, ZIP o combinación de ambos",
    type=["pdf", "zip"],
    accept_multiple_files=True,
    help="Se aceptan ZIP anidados. V2 procesa los PDF en paralelo."
)

if uploaded_files:
    t0 = time.perf_counter()
    with st.spinner("Preparando archivos…"):
        try:
            uploaded = normalize_uploads(uploaded_files)
        except Exception as exc:
            st.error(f"❌ Error preparando archivos: {exc}")
            st.stop()
    prep_time = time.perf_counter() - t0

    st.info(f"📄 **{len(uploaded):,} PDFs** listos para procesar · preparación: **{prep_time:.2f} s**")

    workers_default = min(8, max(2, os.cpu_count() or 4), len(uploaded))
    workers = st.slider("Procesadores simultáneos", min_value=1, max_value=max(1, min(12, len(uploaded))), value=max(1, workers_default),
                        help="Empieza con 4–8. Si el servidor tiene pocos recursos, reduce este valor.")

    if st.button("🚀 Procesar lote completo", type="primary", use_container_width=True):
        if not uploaded:
            st.error("No se encontró ningún PDF.")
            st.stop()

        progress = st.progress(0)
        status = st.empty()
        start = time.perf_counter()

        def update_progress(done, total, filename):
            progress.progress(done / total)
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed else 0
            remaining = (total - done) / rate if rate else 0
            status.write(f"Procesados **{done:,}/{total:,}** · {rate:.1f} PDF/s · restante estimado: {remaining:.0f} s · `{filename}`")

        try:
            results = process_reports_parallel(uploaded, update_progress, workers)
            excel_bytes, excel_part2 = create_excel(results)
            elapsed = time.perf_counter() - start
            progress.progress(1.0)
            status.write(f"✅ Lote terminado en **{elapsed:.2f} s** · velocidad promedio **{len(uploaded)/elapsed:.1f} PDF/s**")

            analysis = results["df_analysis"]
            ftq = analysis.loc[analysis["Metrica"].eq("FTQ (First Time Quality)"), "Valor"].iloc[0] if not analysis.empty else "N/A"
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("PDF", f"{len(uploaded):,}")
            c2.metric("Piezas", f"{results['df_master']['Pieza'].nunique():,}")
            c3.metric("Piezas NOK", f"{results['df_nok']['Pieza'].nunique():,}")
            c4.metric("FTQ", ftq)
            c5.metric("Características NOK", f"{len(results['df_nok']):,}")

            st.download_button("📥 Descargar data.xlsx", excel_bytes,
                               "data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
            if excel_part2:
                st.download_button("📥 Descargar data_part2.xlsx", excel_part2,
                                   "data_part2.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)

            if results["errors"]:
                st.warning(f"⚠️ {len(results['errors'])} PDF(s) no pudieron procesarse.")
                st.dataframe(pd.DataFrame(results["errors"], columns=["Pieza", "Archivo", "Error"]), use_container_width=True)

            with st.expander("📊 Ver resumen"):
                st.dataframe(results["df_analysis"], use_container_width=True)
            with st.expander("🔎 Ver piezas/características NOK"):
                st.dataframe(results["df_nok"], use_container_width=True, height=500)
            with st.expander("📈 Análisis detallado NOK"):
                st.dataframe(results["df_detailed"], use_container_width=True)

        except Exception as exc:
            st.error(f"❌ Error durante el procesamiento: {exc}")
            st.exception(exc)
else:
    st.write("Carga tus reportes para comenzar.")
