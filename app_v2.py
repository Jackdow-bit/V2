# -*- coding: utf-8 -*-
"""Reporte NoK - Streamlit V6. Adaptado desde v5_nok_con_zip.py."""

import io
import os
import re
import zipfile
import hashlib
import time
from pathlib import PurePosixPath
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import pdfplumber
import streamlit as st


# ============================================================
# 1. Global Configurations
# ============================================================

# Maximum number of worker processes for parallel processing
# Using os.cpu_count() or a reasonable fixed number like 4-8 for Colab
MAX_WORKERS = 8
MAX_ZIP_DEPTH = 8 # Maximum depth for nested ZIP files

# Specific threshold for chatter in '(5) 301-400 UPR' and '(9) 301-400 UPR' (from original get_resultado)
CHATTER_NOK_THRESHOLD = 0.00008

# Define the lists for 'Caracteristica' and 'Apoyo'
CHARACTERISTICS_FOR_APOYOS_COLUMNS = ['Diametro', 'Roundness', 'Runout', 'Concentricity', 'Parallelism', 'Taper', 'Cylindricity']
APOYO_IDENTIFIERS = ['Aux:G', '1:A L', '1:A R', '2:C', '3:D', '4:E', '5:B']
LOBES_HEADERS = ['AngleErr', "BC-Rad'sErr", 'BC-Runout', 'BC-Vel./10°', 'Ramp-MaxLift', 'Nose-MaxLift', 'Ramp+9°Vel/1°', 'Nose-Vel./1°', 'Taper', 'Center-Dev']
CHATTER_LOBES_CHARACTERISTICS = ['(1) 40- 80 UPR', '(2) 81-140 UPR', '(3) 141-190 UPR', '(4) 191-300 UPR', '(5) 301-400 UPR']
CHATTER_APOYOS_CHARACTERISTICS = [
    '(1) 5 - 8 UPR', '(2) 9 - 15 UPR', '(3) 16 - 23 UPR', '(4) 24 - 28 UPR',
    '(5) 29- 45 UPR', '(6) 46-70 UPR', '(7) 71-140 UPR', '(8) 141-215 UPR'
]
APOYO_RENAME_MAPPING = {
    '1:': '1:A L', '2:': '1:A R', '3:': '2:C', '4:': '3:D', '5:': '4:E', '6:': '5:B'
}
APOYOS_CHAR_MAPPING = { # Mapping PDF headers to desired DataFrame column names for MAIN JOURNALS
    'Error': 'Diametro', 'Roundness': 'Roundness', 'Runout': 'Runout',
    'Concentricity': 'Concentricity', 'Parallelism': 'Parallelism', 'Taper': 'Taper', 'Cylindricity': 'Cylindricity'
}

# Compile Regex Patterns for efficiency
MAIN_JOURNALS_LINE_PATTERN = re.compile(r'(' + '|'.join(re.escape(i) for i in APOYO_IDENTIFIERS) + r')\s+([\s\d.\-#]+(?:(?:\s+J[\d\-]+:\s+[\d.\-]+)?\s+L[\d\-]+:\s+[\d.\-]+)?(?:\s+Tol:\s+[\d.\-]+)?)')
LOBES_LINE_PATTERN = re.compile(r'^(\d+:\s[A-Z]+-\d+)\s+(.*)')
CHATTER_LINE_START_PATTERN = re.compile(r'^\s*\d+:') # Used for finding start of chatter data
CHATTER_DATA_LINE_PATTERN = re.compile(r'^(\d+):\s+(.*)')

# Excel row limit
MAX_ROWS_PER_SHEET = 1048576 - 1 # 1 header row

# ============================================================
# Helper Functions (adapted from original notebook)
# ============================================================

def get_resultado(value_str, characteristic_name=None):
    '''Determines 'Resultado' based on value string and characteristic name.'''
    # First, check for '#' which always indicates 'Nok'
    if '#' in str(value_str):
        return 'Nok'

    # Specific threshold for chatter in certain characteristics
    if characteristic_name in ['(5) 301-400 UPR', '(9) 301-400 UPR']:
        try:
            # Convert to float for comparison. Handle potential comma decimal separator.
            numeric_val = float(str(value_str).replace(',', '.'))
            if numeric_val > CHATTER_NOK_THRESHOLD:
                return 'Nok'
        except ValueError:
            # If conversion to float fails, it's not a numeric value for comparison.
            pass # Keep it 'Ok' if it's not a numeric value or doesn't exceed threshold.
    return 'Ok'


def calculate_sha256(data_bytes):
    '''Calculates the SHA256 hash of bytes data.'''
    return hashlib.sha256(data_bytes).hexdigest()


# ============================================================
# 2. Robust ZIP and PDF Collection
# ============================================================

def collect_pdf_inputs(uploaded_files):
    """Collect PDFs directly uploaded to Streamlit plus PDFs inside nested ZIPs.

    Returns: (pdf_list, summary, errors)
    """
    pdf_collection = {}  # sha256 -> (display_name, bytes)
    zip_files_found = 0
    ignored_files = []
    collection_errors = []

    def add_pdf(name, content):
        digest = calculate_sha256(content)
        if digest in pdf_collection:
            return
        final_name = name
        existing_names = {item[0] for item in pdf_collection.values()}
        if final_name in existing_names:
            stem = PurePosixPath(name).stem
            suffix = PurePosixPath(name).suffix or '.pdf'
            parent = str(PurePosixPath(name).parent)
            counter = 2
            while final_name in existing_names:
                candidate = f"{stem}_{counter}{suffix}"
                final_name = candidate if parent == '.' else f"{parent}/{candidate}"
                counter += 1
        pdf_collection[digest] = (final_name, content)

    def add_zip_contents_recursive(zip_bytes, source_name='archivo.zip', depth=0):
        nonlocal zip_files_found
        if depth > MAX_ZIP_DEPTH:
            collection_errors.append((source_name, f"Se alcanzó el máximo de ZIP anidados ({MAX_ZIP_DEPTH})."))
            return
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as z:
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    safe_name = str(PurePosixPath(info.filename))
                    if safe_name.startswith('__MACOSX/') or safe_name.endswith('/'):
                        continue
                    suffix = PurePosixPath(safe_name).suffix.lower()
                    full_name = f"{source_name}::{safe_name}"
                    try:
                        content = z.read(info)
                    except Exception as exc:
                        collection_errors.append((full_name, f"No se pudo leer: {exc}"))
                        continue
                    if suffix == '.pdf':
                        add_pdf(full_name, content)
                    elif suffix == '.zip':
                        zip_files_found += 1
                        add_zip_contents_recursive(content, full_name, depth + 1)
        except zipfile.BadZipFile:
            collection_errors.append((source_name, 'ZIP no válido o corrupto.'))
        except Exception as exc:
            collection_errors.append((source_name, f'Error procesando ZIP: {exc}'))

    for uploaded in uploaded_files:
        name = uploaded.name
        content = uploaded.getvalue()
        suffix = PurePosixPath(name).suffix.lower()
        if suffix == '.pdf':
            add_pdf(name, content)
        elif suffix == '.zip':
            zip_files_found += 1
            add_zip_contents_recursive(content, name, 0)
        else:
            ignored_files.append(name)

    pdfs = sorted(pdf_collection.values(), key=lambda item: item[0].lower())
    summary = {
        'pdfs_loaded': len(pdfs),
        'zips_found': zip_files_found,
        'ignored_files_count': len(ignored_files),
        'ignored_files_list': ignored_files,
    }
    return pdfs, summary, collection_errors


# ============================================================
# 3. Single PDF Processing Function
# ============================================================

def process_single_pdf(pdf_tuple, file_consecutive_number):
    '''
    Processes a single PDF file (bytes content) to extract relevant data.

    Args:
        pdf_tuple (tuple): A tuple (pdf_filename, pdf_bytes).
        file_consecutive_number (int): A unique identifier for the piece.

    Returns:
        dict: A dictionary containing extracted data for each section
              (apoyos, levas, chatter_lobes, chatter_apoyos) and any errors.
    '''
    pdf_filename, pdf_bytes = pdf_tuple
    extracted_data = {
        'apoyos_data_rows': [],
        'levas_data_rows': [],
        'chatter_lobes_data_rows': [],
        'chatter_apoyos_data_rows': [],
        'errors': []
    }

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if not pdf.pages:
                extracted_data['errors'].append(f"No pages found in PDF.")
                return extracted_data

            first_page_text = pdf.pages[0].extract_text()
            second_page_text = ""
            if len(pdf.pages) > 1:
                second_page_text = pdf.pages[1].extract_text()

            # --- Extraction logic for MAIN JOURNALS (for df_apoyos_final) ---
            main_journals_start = first_page_text.find('MAIN JOURNALS:')
            lobes_start = first_page_text.find('LOBES:')
            chatter_start = first_page_text.find('CHATTER:')

            if main_journals_start != -1 and lobes_start != -1:
                main_journals_text = first_page_text[main_journals_start:lobes_start]
                main_journals_headers_pdf = ['Measured Diameter', 'Error', 'Roundness', 'Runout', 'Concentricity', 'Parallelism', 'Taper', 'Cylindricity']

                lines = main_journals_text.split('\n')
                for line in lines:
                    match = MAIN_JOURNALS_LINE_PATTERN.match(line.strip())
                    if match:
                        apoyo_val = match.group(1).strip()
                        values_str = match.group(2).strip()
                        raw_values = [v for v in re.split(r'\s+', values_str) if v and not re.match(r'J\d-\d:|J\d:', v) and v != 'Tol:']
                        extracted_data_for_row = {char: None for char in CHARACTERISTICS_FOR_APOYOS_COLUMNS}

                        # Specific parsing logic for '1:A L' and '5:B'
                        if apoyo_val == '1:A L':
                            if len(raw_values) >= 6:
                                extracted_data_for_row['Diametro'] = raw_values[1]
                                extracted_data_for_row['Roundness'] = raw_values[2]
                                extracted_data_for_row['Runout'] = raw_values[3]
                                extracted_data_for_row['Concentricity'] = raw_values[4]
                                extracted_data_for_row['Taper'] = raw_values[5]
                        elif apoyo_val == '5:B':
                            if len(raw_values) >= 7:
                                extracted_data_for_row['Diametro'] = raw_values[1]
                                extracted_data_for_row['Roundness'] = raw_values[2]
                                extracted_data_for_row['Runout'] = raw_values[3]
                                extracted_data_for_row['Concentricity'] = raw_values[4]
                                extracted_data_for_row['Taper'] = raw_values[5]
                                extracted_data_for_row['Cylindricity'] = raw_values[6]
                        else:
                            current_raw_value_index = 0
                            for pdf_header_name in main_journals_headers_pdf:
                                if pdf_header_name == 'Measured Diameter':
                                    if current_raw_value_index < len(raw_values):
                                        current_raw_value_index += 1
                                    continue
                                if pdf_header_name in APOYOS_CHAR_MAPPING:
                                    char_name = APOYOS_CHAR_MAPPING[pdf_header_name]
                                    if char_name in CHARACTERISTICS_FOR_APOYOS_COLUMNS:
                                        if current_raw_value_index < len(raw_values):
                                            extracted_data_for_row[char_name] = raw_values[current_raw_value_index]
                                        current_raw_value_index += 1
                                    else:
                                        if current_raw_value_index < len(raw_values):
                                            current_raw_value_index += 1
                                else:
                                    if current_raw_value_index < len(raw_values):
                                        current_raw_value_index += 1

                        for char_name_in_df in CHARACTERISTICS_FOR_APOYOS_COLUMNS:
                            medicion_val = extracted_data_for_row.get(char_name_in_df)
                            if medicion_val is not None:
                                resultado_val = get_resultado(str(medicion_val))
                                extracted_data['apoyos_data_rows'].append({
                                    'Nombre del archivo': pdf_filename,
                                    'Pieza': file_consecutive_number,
                                    'Leva': None,
                                    'Apoyo': apoyo_val,
                                    'Caracteristica': char_name_in_df,
                                    'Medicion': str(medicion_val).replace('#', ''),
                                    'Area': 'Apoyos',
                                    'Resultado': resultado_val
                                })

            # --- Extraction logic for LOBES (for df_levas_final) ---
            if lobes_start != -1:
                lobes_text = first_page_text[lobes_start:chatter_start if chatter_start != -1 else len(first_page_text)]
                lines = lobes_text.split('\n')
                data_lines_start_index = -1
                for i, line in enumerate(lines):
                    if line.strip().startswith('1: EXH-1'):
                        data_lines_start_index = i
                        break

                if data_lines_start_index != -1:
                    for line in lines[data_lines_start_index:]:
                        stripped_line = line.strip()
                        match = LOBES_LINE_PATTERN.match(stripped_line)
                        if match:
                            leva_val = match.group(1).strip()
                            values_str = match.group(2).strip()
                            raw_values = [v for v in re.split(r'\s+', values_str) if v]

                            processed_values = []
                            if len(raw_values) == len(LOBES_HEADERS):
                                processed_values = raw_values
                            elif len(raw_values) == 2 * len(LOBES_HEADERS):
                                processed_values = [raw_values[j] for j in range(0, len(raw_values), 2)]
                            else:
                                continue

                            for i, char_name in enumerate(LOBES_HEADERS):
                                if i < len(processed_values):
                                    medicion_val = processed_values[i]
                                    resultado_val = get_resultado(medicion_val)
                                    extracted_data['levas_data_rows'].append({
                                        'Nombre del archivo': pdf_filename,
                                        'Pieza': file_consecutive_number,
                                        'Leva': leva_val,
                                        'Apoyo': None,
                                        'Caracteristica': char_name,
                                        'Medicion': medicion_val.replace('#', ''),
                                        'Area': 'Levas',
                                        'Resultado': resultado_val
                                    })

            # --- Extraction logic for CHATTER LEVAS ---
            if chatter_start != -1:
                chatter_text = first_page_text[chatter_start:]
                chatter_lines = chatter_text.split('\n')
                chatter_data_start_idx = -1
                for i, line in enumerate(chatter_lines):
                    if CHATTER_LINE_START_PATTERN.match(line.strip()):
                        chatter_data_start_idx = i
                        break

                if chatter_data_start_idx != -1:
                    for line in chatter_lines[chatter_data_start_idx:]:
                        stripped_line = line.strip()
                        match = CHATTER_DATA_LINE_PATTERN.match(stripped_line)
                        if match:
                            leva_val = match.group(1).strip()
                            values_str = match.group(2).strip()
                            raw_values = [v for v in re.split(r'\s+', values_str) if v]

                            if len(raw_values) == len(CHATTER_LOBES_CHARACTERISTICS) * 2 * 2:
                                num_chatter_chars = len(CHATTER_LOBES_CHARACTERISTICS)
                                amplitudes_bc = [raw_values[i * 2] for i in range(num_chatter_chars)]
                                amplitudes_la = [raw_values[num_chatter_chars * 2 + i * 2] for i in range(num_chatter_chars)]

                                for i, char_name in enumerate(CHATTER_LOBES_CHARACTERISTICS):
                                    medicion_val = amplitudes_bc[i]
                                    resultado_val = get_resultado(medicion_val, characteristic_name=char_name)
                                    extracted_data['chatter_lobes_data_rows'].append({
                                        'Nombre del archivo': pdf_filename,
                                        'Pieza': file_consecutive_number,
                                        'Leva': leva_val,
                                        'Apoyo': None,
                                        'Caracteristica': char_name,
                                        'Medicion': medicion_val.replace('#', ''),
                                        'Area': 'Base Circle',
                                        'Resultado': resultado_val
                                    })
                                for i, char_name in enumerate(CHATTER_LOBES_CHARACTERISTICS):
                                    medicion_val = amplitudes_la[i]
                                    resultado_val = get_resultado(medicion_val, characteristic_name=char_name)
                                    extracted_data['chatter_lobes_data_rows'].append({
                                        'Nombre del archivo': pdf_filename,
                                        'Pieza': file_consecutive_number,
                                        'Leva': leva_val,
                                        'Apoyo': None,
                                        'Caracteristica': char_name,
                                        'Medicion': medicion_val.replace('#', ''),
                                        'Area': 'Lift Area',
                                        'Resultado': resultado_val
                                    })

            # --- Extraction logic for CHATTER APYOS ---
            if second_page_text:
                chatter_journals_start = second_page_text.find('CHATTER: ------------------------- Journals --------------------------')
                if chatter_journals_start != -1:
                    chatter_journals_text = second_page_text[chatter_journals_start:]
                    chatter_journals_lines = chatter_journals_text.split('\n')
                    chatter_journals_data_start_idx = -1
                    for i, line in enumerate(chatter_journals_lines):
                        if CHATTER_LINE_START_PATTERN.match(line.strip()):
                            chatter_journals_data_start_idx = i
                            break

                    if chatter_journals_data_start_idx != -1:
                        for line in chatter_journals_lines[chatter_journals_data_start_idx:]:
                            stripped_line = line.strip()
                            match = CHATTER_DATA_LINE_PATTERN.match(stripped_line)

                            if match:
                                apoyo_val_original = match.group(1).strip()
                                apoyo_val_mapped = APOYO_RENAME_MAPPING.get(apoyo_val_original, apoyo_val_original)
                                values_str = match.group(2).strip()

                                raw_values = [v for v in re.split(r'\s+', values_str) if v]

                                if len(raw_values) == len(CHATTER_APOYOS_CHARACTERISTICS) * 2:
                                    amplitudes = [raw_values[j] for j in range(0, len(raw_values), 2)]
                                    for i, char_name in enumerate(CHATTER_APOYOS_CHARACTERISTICS):
                                        medicion_val = amplitudes[i]
                                        resultado_val = get_resultado(medicion_val, characteristic_name=char_name)
                                        extracted_data['chatter_apoyos_data_rows'].append({
                                            'Nombre del archivo': pdf_filename,
                                            'Pieza': file_consecutive_number,
                                            'Leva': None,
                                            'Apoyo': apoyo_val_mapped,
                                            'Caracteristica': char_name,
                                            'Medicion': medicion_val.replace('#', ''),
                                            'Area': 'Apoyos',
                                            'Resultado': resultado_val
                                        })

    except pdfplumber.pdf.PdfminerException as e:
        extracted_data['errors'].append(f"PDFMiner error: {e}")
    except Exception as e:
        extracted_data['errors'].append(f"Unexpected error during PDF processing: {e}")

    return extracted_data

# ============================================================
# Main Orchestration Logic
# ============================================================


def run_processing_pipeline(pdfs_to_process, summary_collection, collection_errors, progress_bar, status_box, current_file_box, max_workers=8):
    start_time_total = time.perf_counter()
    num_pdfs_total = len(pdfs_to_process)
    all_apoyos_dfs_raw = []
    all_levas_dfs_raw = []
    all_chatter_lobes_dfs_raw = []
    all_chatter_apoyos_dfs_raw = []
    processed_pdf_errors = []
    successful_pdfs_count = 0

    print("\nIniciando procesamiento paralelo de PDFs...")
    # 4. Parallel Processing Orchestration
    # Use max_workers, but not more than the number of PDFs
    workers_to_use = min(MAX_WORKERS, num_pdfs_total)
    print(f"Utilizando {workers_to_use} trabajadores para el procesamiento paralelo.")

    future_to_meta = {}
    with ThreadPoolExecutor(max_workers=workers_to_use) as executor:
        for i, pdf_tuple in enumerate(pdfs_to_process):
            future = executor.submit(process_single_pdf, pdf_tuple, i + 1)
            future_to_meta[future] = (i + 1, pdf_tuple[0])

        for completed_count, future in enumerate(as_completed(future_to_meta), start=1):
            try:
                result = future.result()
                if result['errors']:
                    pdf_filename = future_to_meta[future][1]
                    processed_pdf_errors.append((pdf_filename, "; ".join(result['errors'])))
                else:
                    all_apoyos_dfs_raw.extend(result['apoyos_data_rows'])
                    all_levas_dfs_raw.extend(result['levas_data_rows'])
                    all_chatter_lobes_dfs_raw.extend(result['chatter_lobes_data_rows'])
                    all_chatter_apoyos_dfs_raw.extend(result['chatter_apoyos_data_rows'])
                    successful_pdfs_count += 1

                progress = completed_count / num_pdfs_total if num_pdfs_total else 1.0
                elapsed = time.perf_counter() - start_time_total
                rate = completed_count / elapsed if elapsed > 0 else 0
                remaining = (num_pdfs_total - completed_count) / rate if rate > 0 else 0
                progress_bar.progress(progress, text=f"Procesando PDF {completed_count}/{num_pdfs_total} • {progress:.0%}")
                status_box.info(f"⏱️ Tiempo: {elapsed:.1f} s  |  Velocidad: {rate:.2f} PDF/s  |  ETA: {remaining:.1f} s")
                current_file_box.write(f"📄 Último PDF: {future_to_meta[future][1]}")
            except Exception as e:
                # This catches errors in the future.result() call itself, e.g., process crash
                processed_pdf_errors.append((future_to_meta[future][1], f"Critical error: {e}"))

    # 6. Consolidate Results
    print("\nConsolidando resultados...")
    df_apoyos_final = pd.DataFrame(all_apoyos_dfs_raw)
    if not df_apoyos_final.empty:
        df_apoyos_final['Medicion'] = pd.to_numeric(df_apoyos_final['Medicion'], errors='coerce')

    df_levas_final = pd.DataFrame(all_levas_dfs_raw)
    if not df_levas_final.empty:
        df_levas_final['Medicion'] = pd.to_numeric(df_levas_final['Medicion'], errors='coerce')

    df_chatter_lobes_final = pd.DataFrame(all_chatter_lobes_dfs_raw)
    if not df_chatter_lobes_final.empty:
        df_chatter_lobes_final['Medicion'] = pd.to_numeric(df_chatter_lobes_final['Medicion'], errors='coerce')

    df_chatter_apoyos_final = pd.DataFrame(all_chatter_apoyos_dfs_raw)
    if not df_chatter_apoyos_final.empty:
        df_chatter_apoyos_final['Medicion'] = pd.to_numeric(df_chatter_apoyos_final['Medicion'], errors='coerce')

    # Create df_master by concatenating all final DataFrames
    df_master = pd.concat([
        df_apoyos_final,
        df_levas_final,
        df_chatter_lobes_final,
        df_chatter_apoyos_final
    ], ignore_index=True)

    # Create df_nok by filtering df_master
    df_nok = df_master[df_master['Resultado'] == 'Nok'].copy()

    # --- Calculate summary statistics for 'Analisis_Resumen' ---
    total_pieces_analyzed = df_master['Pieza'].nunique()
    pieces_with_nok = df_nok['Pieza'].unique()
    num_nok_pieces = len(pieces_with_nok)
    # Correct calculation for num_ok_pieces
    all_pieces_ids = df_master['Pieza'].unique()
    ok_pieces_ids = [pid for pid in all_pieces_ids if pid not in pieces_with_nok]
    num_ok_pieces = len(ok_pieces_ids)

    total_characteristics = len(df_master)
    num_ok_characteristics = len(df_master[df_master['Resultado'] == 'Ok'])
    num_nok_characteristics = len(df_nok)
    rejection_percentage = (num_nok_pieces / total_pieces_analyzed) * 100 if total_pieces_analyzed > 0 else 0
    ftq = (num_ok_pieces / total_pieces_analyzed) * 100 if total_pieces_analyzed > 0 else 0
    pct_ok_characteristics = (num_ok_characteristics / total_characteristics) * 100 if total_characteristics > 0 else 0
    pct_nok_characteristics = (num_nok_characteristics / total_characteristics) * 100 if total_characteristics > 0 else 0

    specific_nok_char_to_check = '(5) 301-400 UPR'
    pieces_rejected_by_specific_nok_char = df_nok[df_nok['Caracteristica'] == specific_nok_char_to_check]['Pieza'].unique()
    count_pieces_only_this_nok_char = 0
    for piece_id in pieces_rejected_by_specific_nok_char:
        nok_chars_for_this_piece = df_nok[df_nok['Pieza'] == piece_id]['Caracteristica'].unique()
        if len(nok_chars_for_this_piece) == 1 and nok_chars_for_this_piece[0] == specific_nok_char_to_check:
            count_pieces_only_this_nok_char += 1

    count_pieces_with_specific_nok_and_others = 0
    for piece_id in pieces_rejected_by_specific_nok_char:
        nok_chars_for_this_piece = df_nok[df_nok['Pieza'] == piece_id]['Caracteristica'].unique()
        if specific_nok_char_to_check in nok_chars_for_this_piece and len(nok_chars_for_this_piece) > 1:
            count_pieces_with_specific_nok_and_others += 1

    analysis_data = {
        'Metrica': [
            'Total de piezas analizadas',
            'Piezas con resultado OK (todas las caracteristicas OK)',
            'Piezas con resultado NOK (al menos una caracteristica NOK)',
            'Porcentaje de Rechazo',
            'FTQ (First Time Quality)',
            'Total de características analizadas',
            'Características con resultado OK',
            'Características con resultado NOK',
            '% Características OK',
            '% Características NOK',
            f'Cantidad piezas rechazadas SOLO por "{specific_nok_char_to_check}"',
            f'Cantidad piezas rechazadas por "{specific_nok_char_to_check}" Y alguna otra característica'
        ],
        'Valor': [
            total_pieces_analyzed,
            num_ok_pieces,
            num_nok_pieces,
            f'{rejection_percentage:.2f}%',
            f'{ftq:.2f}%',
            total_characteristics,
            num_ok_characteristics,
            num_nok_characteristics,
            f'{pct_ok_characteristics:.2f}%',
            f'{pct_nok_characteristics:.2f}%',
            count_pieces_only_this_nok_char,
            count_pieces_with_specific_nok_and_others
        ]
    }
    df_analysis = pd.DataFrame(analysis_data)

    # --- Calculate detailed NOK characteristics for 'Analisis_Detallado_NOK' ---
    nok_characteristics_counts = df_nok['Caracteristica'].value_counts().reset_index()
    nok_characteristics_counts.columns = ['Caracteristica', 'Cantidad_NOK']
    top_15_nok_characteristics = nok_characteristics_counts['Caracteristica'].head(15).tolist() # Use top 15 as in plan

    detailed_analysis_results = []
    for char in top_15_nok_characteristics:
        df_char_all = df_master[df_master['Caracteristica'] == char].copy()
        df_char_nok = df_nok[df_nok['Caracteristica'] == char].copy()

        if df_char_all.empty:
            continue

        num_pieces_with_nok = df_char_nok['Pieza'].nunique() if not df_char_nok.empty else 0
        avg_total_medicion = df_char_all['Medicion'].mean()
        std_total_medicion = df_char_all['Medicion'].std()
        avg_nok_medicion = df_char_nok['Medicion'].mean() if not df_char_nok.empty else float('nan')
        std_nok_medicion = df_char_nok['Medicion'].std() if not df_char_nok.empty else float('nan')
        max_nok_medicion = df_char_nok['Medicion'].max() if not df_char_nok.empty else float('nan')
        min_nok_medicion = df_char_nok['Medicion'].min() if not df_char_nok.empty else float('nan')

        detailed_analysis_results.append({
            'Caracteristica': char,
            'Cantidad Piezas NOK': num_pieces_with_nok,
            'Promedio Total Med.': f'{avg_total_medicion:.4f}' if pd.notna(avg_total_medicion) else 'N/A',
            'Std Total Med.': f'{std_total_medicion:.4f}' if pd.notna(std_total_medicion) else 'N/A',
            'Promedio NOK Med.': f'{avg_nok_medicion:.4f}' if pd.notna(avg_nok_medicion) else 'N/A',
            'Std NOK Med.': f'{std_nok_medicion:.4f}' if pd.notna(std_nok_medicion) else 'N/A',
            'Max NOK Med.': f'{max_nok_medicion:.4f}' if pd.notna(max_nok_medicion) else 'N/A',
            'Min NOK Med.': f'{min_nok_medicion:.4f}' if pd.notna(min_nok_medicion) else 'N/A'
        })
    df_detailed_analysis_summary = pd.DataFrame(detailed_analysis_results)



    # 7. Generate Excel in memory for Streamlit download
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_excel_filename = f"Reporte_NoK_V6_{timestamp}.xlsx"
    output_excel_filename_part2 = f"Reporte_NoK_V6_Levas_Parte2_{timestamp}.xlsx"

    dfs_to_write_to_main_excel = []
    df_levas_second_file = None

    if not df_apoyos_final.empty:
        dfs_to_write_to_main_excel.append((df_apoyos_final, 'Apoyos'))
    if not df_levas_final.empty:
        if len(df_levas_final) > MAX_ROWS_PER_SHEET:
            df_levas_part1 = df_levas_final.iloc[:MAX_ROWS_PER_SHEET].copy()
            df_levas_second_file = df_levas_final.iloc[MAX_ROWS_PER_SHEET:].copy()
            dfs_to_write_to_main_excel.append((df_levas_part1, 'Levas'))
        else:
            dfs_to_write_to_main_excel.append((df_levas_final, 'Levas'))
    if not df_chatter_lobes_final.empty:
        dfs_to_write_to_main_excel.append((df_chatter_lobes_final, 'chatter Levas'))
    if not df_chatter_apoyos_final.empty:
        dfs_to_write_to_main_excel.append((df_chatter_apoyos_final, 'Chatter apoyos'))
    if not df_nok.empty:
        dfs_to_write_to_main_excel.append((df_nok, 'Nok'))
    if not df_analysis.empty:
        dfs_to_write_to_main_excel.append((df_analysis, 'Analisis_Resumen'))
    if not df_detailed_analysis_summary.empty:
        dfs_to_write_to_main_excel.append((df_detailed_analysis_summary, 'Analisis_Detallado_NOK'))

    excel_bytes = None
    if dfs_to_write_to_main_excel:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for df_to_write, sheet_name in dfs_to_write_to_main_excel:
                df_to_write.to_excel(writer, index=False, sheet_name=sheet_name)
        excel_bytes = output.getvalue()

    excel_part2_bytes = None
    if df_levas_second_file is not None and not df_levas_second_file.empty:
        output2 = io.BytesIO()
        with pd.ExcelWriter(output2, engine='openpyxl') as writer2:
            df_levas_second_file.to_excel(writer2, index=False, sheet_name='Levas_Parte_2')
        excel_part2_bytes = output2.getvalue()

    end_time_total = time.perf_counter()
    total_processing_time = end_time_total - start_time_total

    return {
        'excel_bytes': excel_bytes,
        'excel_filename': output_excel_filename,
        'excel_part2_bytes': excel_part2_bytes,
        'excel_part2_filename': output_excel_filename_part2,
        'df_apoyos': df_apoyos_final,
        'df_levas': df_levas_final,
        'df_chatter_lobes': df_chatter_lobes_final,
        'df_chatter_apoyos': df_chatter_apoyos_final,
        'df_master': df_master,
        'df_nok': df_nok,
        'df_analysis': df_analysis,
        'df_detailed': df_detailed_analysis_summary,
        'num_pdfs': num_pdfs_total,
        'successful': successful_pdfs_count,
        'errors': processed_pdf_errors,
        'collection_errors': collection_errors,
        'summary_collection': summary_collection,
        'elapsed': total_processing_time,
    }


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Reporte NoK",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📊 Reporte NoK – PDF / ZIP")
st.caption("Carga reportes PDF o ZIP desde tu celular. El procesamiento ocurre en el servidor y el Excel se descarga al finalizar.")

with st.sidebar:
    st.header("⚙️ Configuración")
    workers = st.slider("Trabajadores simultáneos", min_value=1, max_value=15, value=min(8, os.cpu_count() or 1), step=1)
    st.caption("En Streamlit Cloud recomiendo 4–8 para estabilidad. Puedes subirlo si el servidor lo soporta.")
    st.info(f"Umbral chatter NOK: {CHATTER_NOK_THRESHOLD}")
    st.info(f"ZIP anidados: hasta {MAX_ZIP_DEPTH} niveles")

uploaded_files = st.file_uploader(
    "📁 Selecciona PDF o ZIP",
    type=["pdf", "zip"],
    accept_multiple_files=True,
    help="Puedes seleccionar varios PDF y ZIP. Los ZIP pueden contener otros ZIP hasta 8 niveles.",
)

if uploaded_files:
    with st.spinner("Analizando archivos y buscando PDFs..."):
        pdfs, collection_summary, collection_errors = collect_pdf_inputs(uploaded_files)

    c1, c2, c3 = st.columns(3)
    c1.metric("PDF únicos", collection_summary['pdfs_loaded'])
    c2.metric("ZIP detectados", collection_summary['zips_found'])
    c3.metric("Errores de carga", len(collection_errors))

    if collection_errors:
        with st.expander("⚠️ Ver errores de carga"):
            for filename, error in collection_errors:
                st.warning(f"**{filename}** — {error}")

    if collection_summary['ignored_files_list']:
        with st.expander("ℹ️ Archivos ignorados"):
            st.write(collection_summary['ignored_files_list'])

    if pdfs:
        with st.expander(f"📄 PDFs encontrados ({len(pdfs)})"):
            st.dataframe(
                pd.DataFrame({'#': range(1, len(pdfs)+1), 'Archivo': [p[0] for p in pdfs]}),
                use_container_width=True,
                hide_index=True,
            )

        if st.button("▶️ Procesar reportes", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Preparando procesamiento...")
            status_box = st.empty()
            current_file_box = st.empty()
            started = time.perf_counter()
            try:
                result = run_processing_pipeline(
                    pdfs,
                    collection_summary,
                    collection_errors,
                    progress_bar,
                    status_box,
                    current_file_box,
                    max_workers=workers,
                )
                result['elapsed'] = time.perf_counter() - started
                st.session_state['reporte_nok_result'] = result
                progress_bar.progress(1.0, text="✅ Procesamiento terminado")
                status_box.success(f"Proceso terminado en {result['elapsed']:.1f} segundos.")
                current_file_box.empty()
            except Exception as exc:
                st.error(f"❌ Error general durante el procesamiento: {exc}")
                st.exception(exc)

result = st.session_state.get('reporte_nok_result')
if result:
    st.divider()
    st.subheader("📈 Resultado")
    a, b, c, d = st.columns(4)
    a.metric("PDF identificados", result['num_pdfs'])
    b.metric("Procesados OK", result['successful'])
    c.metric("Con error", len(result['errors']))
    d.metric("Tiempo", f"{result['elapsed']:.1f} s")

    if result['excel_bytes']:
        st.download_button(
            "📥 Descargar reporte Excel",
            data=result['excel_bytes'],
            file_name=result['excel_filename'],
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            use_container_width=True,
            type='primary',
        )
    if result['excel_part2_bytes']:
        st.download_button(
            "📥 Descargar Levas – Parte 2",
            data=result['excel_part2_bytes'],
            file_name=result['excel_part2_filename'],
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            use_container_width=True,
        )

    if result['errors']:
        with st.expander("⚠️ PDFs con errores"):
            for filename, error in result['errors']:
                st.error(f"**{filename}** — {error}")

    st.subheader("Resumen de análisis")
    st.dataframe(result['df_analysis'], use_container_width=True, hide_index=True)

    tabs = st.tabs(["NOK", "Apoyos", "Levas", "Chatter Levas", "Chatter apoyos", "Detalle NOK"])
    frames = [result['df_nok'], result['df_apoyos'], result['df_levas'], result['df_chatter_lobes'], result['df_chatter_apoyos'], result['df_detailed']]
    for tab, frame in zip(tabs, frames):
        with tab:
            st.dataframe(frame, use_container_width=True, hide_index=True, height=420)
else:
    st.info("👆 Selecciona tus PDF/ZIP y después pulsa **Procesar reportes**.")
