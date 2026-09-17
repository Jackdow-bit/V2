# -*- coding: utf-8 -*-
"""Reporte NoK V3 - Streamlit + Colab compatible
Procesamiento optimizado de reportes PDF (MAIN JOURNALS, LOBES, CHATTER).
"""
import io
import os
import re
import zipfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pdfplumber

# ---------------- CONFIGURACION ----------------
CHATTER_NOK_THRESHOLD = 0.0001
MAX_ROWS_PER_SHEET = 1_048_575

CHARACTERISTICS = ['Diametro', 'Roundness', 'Runout', 'Concentricity', 'Parallelism', 'Taper', 'Cylindricity']
APOYO_IDENTIFIERS = ['Aux:G', '1:A L', '1:A R', '2:C', '3:D', '4:E', '5:B']
CHAR_MAPPING = {
    'Error': 'Diametro', 'Roundness': 'Roundness', 'Runout': 'Runout',
    'Concentricity': 'Concentricity', 'Parallelism': 'Parallelism',
    'Taper': 'Taper', 'Cylindricity': 'Cylindricity'
}
CHATTER_APOYOS_CHARS = [
    '(1) 5 - 8 UPR', '(2) 9 - 15 UPR', '(3) 16 - 23 UPR', '(4) 24 - 28 UPR',
    '(5) 29- 45 UPR', '(6) 46-70 UPR', '(7) 71-140 UPR', '(8) 141-215 UPR'
]
CHATTER_CHARS = ['(1) 40- 80 UPR', '(2) 81-140 UPR', '(3) 141-190 UPR', '(4) 191-300 UPR', '(5) 301-400 UPR']
APOYO_RENAME = {'1:': '1:A L', '2:': '1:A R', '3:': '2:C', '4:': '3:D', '5:': '4:E', '6:': '5:B'}
LOBES_HEADERS = ['AngleErr', "BC-Rad'sErr", 'BC-Runout', 'BC-Vel./10', 'Ramp-MaxLift', 'Nose-MaxLift', 'Ramp+9Vel/1', 'Nose-Vel./1', 'Taper', 'Center-Dev']

# Regex compiladas una sola vez (antes se recompilaban dentro del procesamiento)
RE_SPLIT = re.compile(r'\s+')
RE_MAIN = re.compile(r'(' + '|'.join(re.escape(x) for x in APOYO_IDENTIFIERS) + r')\s+([\s\d.\-#]+(?:\s+J[\d\-]+:\s+[\d.\-]+)?(?:\s+L[\d\-]+:\s+[\d.\-]+)?(?:\s+Tol:\s+[\d.\-]+)?)')
RE_LOBES = re.compile(r'^(\d+:\s[A-Z]+-\d+)\s+(.*)')
RE_NUMBERED = re.compile(r'^(\d+):\s+(.*)')
RE_J_MARKER = re.compile(r'J\d-\d:|J\d:')


def resultado(value, characteristic=None):
    s = str(value)
    if '#' in s:
        return 'Nok'
    if characteristic in ('(5) 301-400 UPR', '(9) 301-400 UPR'):
        try:
            if float(s.replace(',', '.')) > CHATTER_NOK_THRESHOLD:
                return 'Nok'
        except (ValueError, TypeError):
            pass
    return 'Ok'


def add_row(rows, filename, pieza, leva, apoyo, char, medicion, area):
    s = str(medicion)
    rows.append((filename, pieza, leva, apoyo, char, s.replace('#', ''), area,
                 resultado(s, char if area in ('Base Circle', 'Lift Area', 'Apoyos') and char in CHATTER_CHARS + CHATTER_APOYOS_CHARS else None)))


def rows_to_df(rows):
    cols = ['Nombre del archivo', 'Pieza', 'Leva', 'Apoyo', 'Caracteristica', 'Medicion', 'Area', 'Resultado']
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame.from_records(rows, columns=cols)
    # Conversión vectorizada una sola vez.
    df['Medicion'] = pd.to_numeric(df['Medicion'], errors='coerce')
    return df


def parse_pdf(pdf_filename, pdf_bytes, pieza):
    """Procesa únicamente las dos primeras páginas. Devuelve 4 listas de filas + error."""
    apoyos, levas, chatter_levas, chatter_apoyos = [], [], [], []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if not pdf.pages:
                return pieza, pdf_filename, apoyos, levas, chatter_levas, chatter_apoyos, 'PDF sin páginas'
            first = pdf.pages[0].extract_text() or ''
            second = pdf.pages[1].extract_text() or '' if len(pdf.pages) > 1 else ''

        main_start = first.find('MAIN JOURNALS:')
        lobes_start = first.find('LOBES:')
        chatter_start = first.find('CHATTER:')

        # MAIN JOURNALS
        if main_start >= 0 and lobes_start >= 0:
            block = first[main_start:lobes_start]
            for line in block.splitlines():
                m = RE_MAIN.match(line.strip())
                if not m:
                    continue
                apoyo = m.group(1)
                raw = [v for v in RE_SPLIT.split(m.group(2).strip()) if v and not RE_J_MARKER.match(v) and v != 'Tol:']
                vals = {c: None for c in CHARACTERISTICS}
                if apoyo == '1:A L' and len(raw) >= 6:
                    vals.update(dict(zip(['Diametro','Roundness','Runout','Concentricity','Taper'], raw[1:6])))
                elif apoyo == '5:B' and len(raw) >= 7:
                    vals.update(dict(zip(['Diametro','Roundness','Runout','Concentricity','Taper','Cylindricity'], raw[1:7])))
                else:
                    # En las demás líneas, Error corresponde a Diametro y se omite Measured Diameter.
                    idx = 1
                    for header in ['Error','Roundness','Runout','Concentricity','Parallelism','Taper','Cylindricity']:
                        if idx < len(raw):
                            vals[CHAR_MAPPING[header]] = raw[idx]
                        idx += 1
                for char, value in vals.items():
                    if value is not None:
                        add_row(apoyos, pdf_filename, pieza, None, apoyo, char, value, 'Apoyos')

        # LOBES
        if lobes_start >= 0:
            block = first[lobes_start:chatter_start if chatter_start >= 0 else len(first)]
            started = False
            for line in block.splitlines():
                st = line.strip()
                if not started and st.startswith('1: EXH-1'):
                    started = True
                if not started:
                    continue
                m = RE_LOBES.match(st)
                if not m:
                    continue
                raw = RE_SPLIT.split(m.group(2).strip())
                if len(raw) == 2 * len(LOBES_HEADERS):
                    raw = raw[::2]
                if len(raw) != len(LOBES_HEADERS):
                    continue
                for char, value in zip(LOBES_HEADERS, raw):
                    add_row(levas, pdf_filename, pieza, m.group(1), None, char, value, 'Levas')

        # CHATTER LEVAS
        if chatter_start >= 0:
            block = first[chatter_start:]
            started = False
            n = len(CHATTER_CHARS)
            for line in block.splitlines():
                st = line.strip()
                if not started and re.match(r'^\d+:', st):
                    started = True
                if not started:
                    continue
                m = RE_NUMBERED.match(st)
                if not m:
                    continue
                raw = RE_SPLIT.split(m.group(2).strip())
                if len(raw) != n * 4:
                    continue
                bc = raw[0:n*2:2]
                la = raw[n*2::2]
                for char, value in zip(CHATTER_CHARS, bc):
                    add_row(chatter_levas, pdf_filename, pieza, m.group(1), None, char, value, 'Base Circle')
                for char, value in zip(CHATTER_CHARS, la):
                    add_row(chatter_levas, pdf_filename, pieza, m.group(1), None, char, value, 'Lift Area')

        # CHATTER APoyos (página 2)
        marker = 'CHATTER: ------------------------- Journals --------------------------'
        start = second.find(marker) if second else -1
        if start >= 0:
            started = False
            n = len(CHATTER_APOYOS_CHARS)
            for line in second[start:].splitlines():
                st = line.strip()
                if not started and re.match(r'^\d+:', st):
                    started = True
                if not started:
                    continue
                m = RE_NUMBERED.match(st)
                if not m:
                    continue
                raw = RE_SPLIT.split(m.group(2).strip())
                if len(raw) != n * 2:
                    continue
                apoyo = APOYO_RENAME.get(m.group(1), m.group(1))
                for char, value in zip(CHATTER_APOYOS_CHARS, raw[::2]):
                    add_row(chatter_apoyos, pdf_filename, pieza, None, apoyo, char, value, 'Apoyos')

        return pieza, pdf_filename, apoyos, levas, chatter_levas, chatter_apoyos, None
    except Exception as e:
        return pieza, pdf_filename, [], [], [], [], f'{type(e).__name__}: {e}'


def extract_zip_files(uploaded_files):
    """Extrae PDFs de ZIPs, incluyendo ZIP anidado, sin colisiones silenciosas de nombres."""
    pdfs = {}
    errors = []
    queue = list(uploaded_files.items())
    while queue:
        name, data = queue.pop(0)
        low = name.lower()
        if low.endswith('.pdf'):
            base = os.path.basename(name)
            if base.startswith('.') or base == '':
                continue
            candidate = base
            if candidate in pdfs:
                stem, ext = os.path.splitext(base)
                i = 2
                while f'{stem}_{i}{ext}' in pdfs:
                    i += 1
                candidate = f'{stem}_{i}{ext}'
            pdfs[candidate] = data
        elif low.endswith('.zip'):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for info in zf.infolist():
                        if info.is_dir() or info.filename.startswith('__MACOSX/'):
                            continue
                        queue.append((info.filename, zf.read(info)))
            except zipfile.BadZipFile:
                errors.append(f'ZIP inválido: {name}')
            except Exception as e:
                errors.append(f'Error ZIP {name}: {e}')
    return pdfs, errors


def build_analysis(master, nok):
    if master.empty:
        return pd.DataFrame(columns=['Metrica','Valor']), pd.DataFrame()
    pieces_nok = set(nok['Pieza'].unique())
    total_pieces = master['Pieza'].nunique()
    nok_pieces = len(pieces_nok)
    ok_pieces = total_pieces - nok_pieces
    total_chars = len(master)
    nok_chars = len(nok)
    ok_chars = total_chars - nok_chars
    specific = '(5) 301-400 UPR'
    specific_pieces = nok.loc[nok['Caracteristica'].eq(specific), 'Pieza'].unique()
    if len(specific_pieces):
        counts = nok[nok['Pieza'].isin(specific_pieces)].groupby('Pieza')['Caracteristica'].nunique()
        only_specific = int((counts == 1).sum())
        with_others = int((counts > 1).sum())
    else:
        only_specific = with_others = 0
    analysis = pd.DataFrame({
        'Metrica': ['Total de piezas analizadas','Piezas con resultado OK (todas las caracteristicas OK)',
                    'Piezas con resultado NOK (al menos una caracteristica NOK)','Porcentaje de Rechazo',
                    'FTQ (First Time Quality)','Total de características analizadas','Características con resultado OK',
                    'Características con resultado NOK','% Características OK','% Características NOK',
                    f'Cantidad piezas rechazadas SOLO por "{specific}"',
                    f'Cantidad piezas rechazadas por "{specific}" Y alguna otra característica'],
        'Valor': [total_pieces,ok_pieces,nok_pieces,f'{100*nok_pieces/total_pieces:.2f}%',
                  f'{100*ok_pieces/total_pieces:.2f}%',total_chars,ok_chars,nok_chars,
                  f'{100*ok_chars/total_chars:.2f}%',f'{100*nok_chars/total_chars:.2f}%',only_specific,with_others]
    })

    top = nok['Caracteristica'].value_counts().head(15).index.tolist()
    a = master[master['Caracteristica'].isin(top)].groupby('Caracteristica')['Medicion'].agg(['mean','std'])
    b = nok[nok['Caracteristica'].isin(top)].groupby('Caracteristica')['Medicion'].agg(['count','mean','std','max','min'])
    detail = a.join(b, lsuffix='_total', rsuffix='_nok').reindex(top)
    detail = detail.reset_index().rename(columns={
        'count':'Cantidad NOK','mean_total':'Promedio Total Med.','std_total':'Std Total Med.',
        'mean_nok':'Promedio NOK Med.','std_nok':'Std NOK Med.','max':'Max NOK Med.','min':'Min NOK Med.'
    })
    if not detail.empty:
        detail['Cantidad Piezas NOK'] = nok[nok['Caracteristica'].isin(top)].groupby('Caracteristica')['Pieza'].nunique().reindex(top).fillna(0).astype(int).values
        detail = detail[['Caracteristica','Cantidad Piezas NOK','Promedio Total Med.','Std Total Med.',
                         'Promedio NOK Med.','Std NOK Med.','Max NOK Med.','Min NOK Med.']]
        for c in detail.columns[2:]:
            detail[c] = detail[c].map(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
    return analysis, detail


def process_reports(uploaded_files, workers=None, progress_callback=None):
    pdfs, zip_errors = extract_zip_files(uploaded_files)
    items = sorted(pdfs.items(), key=lambda x: x[0].lower())
    if not items:
        raise ValueError('No se encontraron archivos PDF.')
    workers = workers or min(8, max(2, (os.cpu_count() or 4)))
    results = [None] * len(items)
    errors = list(zip_errors)
    done = 0
    # Threads evitan problemas de spawn en Streamlit/Windows y permiten concurrencia durante I/O + parsing.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(parse_pdf, name, data, i+1): i for i, (name, data) in enumerate(items)}
        for fut in as_completed(futures):
            idx = futures[fut]
            res = fut.result()
            results[idx] = res
            done += 1
            if res[-1]:
                errors.append(f'{res[1]}: {res[-1]}')
            if progress_callback:
                progress_callback(done, len(items), res[1])

    apoyos, levas, chatter_l, chatter_a = [], [], [], []
    for res in results:
        _, _, a, l, cl, ca, _ = res
        apoyos.extend(a); levas.extend(l); chatter_l.extend(cl); chatter_a.extend(ca)
    dfs = [rows_to_df(x) for x in (apoyos, levas, chatter_l, chatter_a)]
    df_apoyos, df_levas, df_chatter_lobes, df_chatter_apoyos = dfs
    master = pd.concat(dfs, ignore_index=True) if any(not d.empty for d in dfs) else pd.DataFrame(columns=df_apoyos.columns)
    nok = master[master['Resultado'].eq('Nok')].copy() if not master.empty else master.copy()
    analysis, detail = build_analysis(master, nok)
    return {
        'Apoyos': df_apoyos, 'Levas': df_levas, 'chatter Levas': df_chatter_lobes,
        'Chatter apoyos': df_chatter_apoyos, 'Nok': nok,
        'Analisis_Resumen': analysis, 'Analisis_Detallado_NOK': detail,
        '_master': master, '_errors': errors, '_files': len(items)
    }


def create_excel(results):
    """Escribe Excel una sola vez. Master se conserva en memoria pero no se exporta, como en el código original."""
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        for name in ['Apoyos','Levas','chatter Levas','Chatter apoyos','Nok','Analisis_Resumen','Analisis_Detallado_NOK']:
            df = results[name]
            if name == 'Levas' and len(df) > MAX_ROWS_PER_SHEET:
                df.iloc[:MAX_ROWS_PER_SHEET].to_excel(writer, index=False, sheet_name='Levas')
                # Parte 2 se genera después en otro archivo.
            else:
                df.to_excel(writer, index=False, sheet_name=name)
    out.seek(0)
    part2 = None
    levas = results['Levas']
    if len(levas) > MAX_ROWS_PER_SHEET:
        part2 = io.BytesIO()
        with pd.ExcelWriter(part2, engine='openpyxl') as writer:
            levas.iloc[MAX_ROWS_PER_SHEET:].to_excel(writer, index=False, sheet_name='Levas_Parte_2')
        part2.seek(0)
    return out.getvalue(), part2.getvalue() if part2 else None


# ---------------- STREAMLIT ----------------
def run_streamlit():
    import streamlit as st
    st.set_page_config(page_title='Reporte NoK', page_icon='📊', layout='wide')
    st.title('📊 Reporte NoK — V3 Optimizada')
    st.caption('Procesamiento paralelo de PDF + ZIP. Solo se leen las primeras 2 páginas de cada reporte.')
    uploads = st.file_uploader('Arrastra tus PDF o ZIP aquí', type=['pdf','zip'], accept_multiple_files=True)
    workers = st.slider('Procesos concurrentes', 1, min(16, max(1, (os.cpu_count() or 4))), min(8, max(2, (os.cpu_count() or 4))))
    if uploads and st.button('🚀 Procesar reportes', type='primary'):
        raw = {f.name: f.getvalue() for f in uploads}
        bar = st.progress(0, text='Preparando archivos...')
        status = st.empty()
        start = time.perf_counter()
        def cb(done, total, name):
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed else 0
            eta = (total-done)/rate if rate else 0
            bar.progress(done/total, text=f'{done}/{total} PDF · {rate:.2f} PDF/s · ETA {eta:.1f}s')
            status.write(f'Procesando: **{name}**')
        try:
            r = process_reports(raw, workers=workers, progress_callback=cb)
            elapsed = time.perf_counter() - start
            xlsx, part2 = create_excel(r)
            st.session_state['report_results'] = r
            st.session_state['report_xlsx'] = xlsx
            st.session_state['report_part2'] = part2
            st.session_state['report_elapsed'] = elapsed
            bar.progress(1.0, text=f'Completado en {elapsed:.2f} s')
        except Exception as e:
            st.error(f'Error: {e}')
            st.exception(e)
    if 'report_results' in st.session_state:
        r = st.session_state['report_results']; master = r['_master']; nok = r['Nok']
        total = master['Pieza'].nunique() if not master.empty else 0
        nok_p = nok['Pieza'].nunique() if not nok.empty else 0
        ftq = 100*(total-nok_p)/total if total else 0
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Piezas', total); c2.metric('Piezas NOK', nok_p); c3.metric('FTQ', f'{ftq:.2f}%'); c4.metric('Características NOK', len(nok))
        st.download_button('📥 Descargar data.xlsx', st.session_state['report_xlsx'], 'data.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        if st.session_state.get('report_part2'):
            st.download_button('📥 Descargar data_part2.xlsx', st.session_state['report_part2'], 'data_part2.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        if r['_errors']:
            st.warning('Algunos archivos tuvieron problemas:')
            st.write(r['_errors'])
        st.subheader('Resumen')
        st.dataframe(r['Analisis_Resumen'], use_container_width=True, hide_index=True)
        st.subheader('Características NOK')
        st.dataframe(r['Analisis_Detallado_NOK'], use_container_width=True, hide_index=True)
        if not nok.empty:
            st.subheader('NOK detectados')
            st.dataframe(nok.head(5000), use_container_width=True, hide_index=True)


if __name__ == '__main__':
    run_streamlit()
