# parser.py

import numpy as np
import pandas as pd
from pds4_tools import pds4_read
from datetime import datetime, timedelta
import glob
import struct
import os
import re
import xml.etree.ElementTree as ET

def extract_bits_from_bytes(byte_data, start_bit, num_bits):
    if isinstance(byte_data, (bytes, np.bytes_)) and len(byte_data) > 0:
        value = int.from_bytes(byte_data, byteorder='big', signed=False)

        total_bits = len(byte_data) * 8
        if start_bit + num_bits - 1 > total_bits:
            return 0

        shift = total_bits - (start_bit + num_bits - 1)
        
        mask = (1 << num_bits) - 1
        return (value >> shift) & mask
    
    return 0

def extract_items(packed_data, items_type):  
    if items_type == '6_14':
        return {
            'format_id': extract_bits_from_bytes(packed_data, 1, 3),         
            'receiving_station_id': extract_bits_from_bytes(packed_data, 4, 7), 
            'transmitting_station_id': extract_bits_from_bytes(packed_data, 11, 7),
            'network_id': extract_bits_from_bytes(packed_data, 18, 2),         
            'data_type_id': extract_bits_from_bytes(packed_data, 20, 6),      
            'downlink_band_id': extract_bits_from_bytes(packed_data, 26, 2),  
            'uplink_band_id': extract_bits_from_bytes(packed_data, 28, 2),     
            'ref_freq_band_id': extract_bits_from_bytes(packed_data, 30, 2),   
            'data_validity': extract_bits_from_bytes(packed_data, 32, 1)       
        }
    elif items_type == '15_19':
        return {
            'item_15': extract_bits_from_bytes(packed_data, 1, 7),   
            'item_16': extract_bits_from_bytes(packed_data, 8, 10), 
            'item_17': extract_bits_from_bytes(packed_data, 18, 1),   
            'item_18': extract_bits_from_bytes(packed_data, 19, 22),  
            'item_19': extract_bits_from_bytes(packed_data, 41, 24)  
        }
    else:
        return {}

def parse_to_df(file_name):
    structures = pds4_read(file_name)
    orbit_data_table = structures[5]
    structured_array = orbit_data_table.data
    
    extracted_data = []
    spice_epoch = datetime(1950, 1, 1)
    
    for i, record in enumerate(structured_array):
        time_tag_int = record[0]
        record_time = spice_epoch + timedelta(seconds=int(time_tag_int))
        
        observable_int = record[2]
        observable_frac = record[3]
        
        items_6_14 = extract_items(record[4], '6_14')
        items_15_19 = extract_items(record[5], '15_19')
        
        full_observable = observable_int + observable_frac * 1e-9
        full_ref_freq = (items_15_19['item_18'] * (2 ** 24) + items_15_19['item_19']) / 1000
        
        # Добавляем проверку на нулевую частоту
        if full_ref_freq <= 0:
            continue
            
        if items_6_14['network_id'] == 0 and items_6_14['data_type_id'] == 37 and items_6_14['data_validity'] == 0 and items_6_14['format_id'] == 2:
            extracted_record = {
                'record_index': i,
                'time_tag_seconds': time_tag_int,
                'record_time': record_time,
                'full_observable': full_observable,
                'observable_int': observable_int,
                'observable_frac': observable_frac,
                'receiving_station_id': items_6_14['receiving_station_id'],
                'transmitting_station_id': items_6_14['transmitting_station_id'],
                'network_id': items_6_14['network_id'],
                'data_type_id': items_6_14['data_type_id'],
                'downlink_band_id': items_6_14['downlink_band_id'],
                'uplink_band_id': items_6_14['uplink_band_id'],
                'ref_freq_band_id': items_6_14['ref_freq_band_id'],
                'data_validity': items_6_14['data_validity'],
                'format_id': items_6_14['format_id'],
                'full_ref_freq': full_ref_freq,  # в МГц
                'ref_freq_int': items_15_19['item_18'],
                'ref_freq_frac': items_15_19['item_19'],
                'item_15': items_15_19['item_15'],  # lowest frequency ranging component
                'item_17': items_15_19['item_17'],  # ramp flag
            }
            
            extracted_data.append(extracted_record)
    
    df_extracted = pd.DataFrame(extracted_data)
    return df_extracted 

def read_ramp_offsets_from_xml(xml_file):
    ramps_offsets = {}
    
    try:
        namespaces = {
            'pds': 'http://pds.nasa.gov/pds4/pds/v1'
        }
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for table in root.findall('.//pds:Table_Binary', namespaces):
            name_elem = table.find('pds:name', namespaces)
            if name_elem is not None and name_elem.text and 'Ramp Group' in name_elem.text:
                name_text = name_elem.text
                
                match = re.search(r'Station\s+(\d+)', name_text)
                if match:
                    station_id = int(match.group(1))

                    offset_elem = table.find('pds:offset', namespaces)
                    if offset_elem is not None:
                        offset = int(offset_elem.text)

                        records_elem = table.find('pds:records', namespaces)
                        records = int(records_elem.text) if records_elem is not None else 0

                        if 'Header' in name_text:
                            key = 'header'
                        else:
                            key = 'data'
                        
                        if station_id not in ramps_offsets:
                            ramps_offsets[station_id] = {}
                        
                        ramps_offsets[station_id][key] = offset
                        ramps_offsets[station_id]['records'] = records
    
    except Exception as e:
        print(f"Ошибка чтения offsets из XML: {e}")
        import traceback
        traceback.print_exc()
    
    return ramps_offsets

def parse_ramp_record_binary_correct(record_bytes):
    if len(record_bytes) < 36:
        return None
    
    try:
        # 1. Время начала (0-7 байты)
        t_start_int = struct.unpack('>I', record_bytes[0:4])[0]
        t_start_frac = struct.unpack('>I', record_bytes[4:8])[0]
        t_start = t_start_int + t_start_frac / 1e9
        
        # 2. Скорость рампы (8-15 байты)
        ramp_rate_int = struct.unpack('>i', record_bytes[8:12])[0]    # целая часть
        ramp_rate_frac = struct.unpack('>i', record_bytes[12:16])[0]  # дробная часть (nanoHz/s)
        ramp_rate = ramp_rate_int + (ramp_rate_frac / 1e9)
        
        # 3. Упакованные данные: GHz часть + Station ID (16-19 байты)
        packed = struct.unpack('>I', record_bytes[16:20])[0]
        
        # Бит 1-22: GHz часть (22 бита)
        # Бит 23-32: Station ID (10 бит)
        ghz_part = (packed >> 10) & 0x3FFFFF  # 22 бита
        tx_station_id = packed & 0x3FF         # 10 бит
        
        # 4. Частота: modulo 10^9 (20-23 байты)
        freq_mod_1e9 = struct.unpack('>I', record_bytes[20:24])[0]
        
        # 5. Дробная часть частоты (24-27 байты)
        freq_frac = struct.unpack('>I', record_bytes[24:28])[0]
        
        # Полная частота в Гц
        f_start = ghz_part * 1e9 + freq_mod_1e9 + (freq_frac / 1e9)
        
        # 6. Время окончания рампы (28-35 байты)
        t_end_int = struct.unpack('>I', record_bytes[28:32])[0]
        t_end_frac = struct.unpack('>I', record_bytes[32:36])[0]
        t_end = t_end_int + t_end_frac / 1e9
        
        return {
            't_start': t_start,
            't_end': t_end,
            'f_start': f_start,
            'ramp_rate': ramp_rate,
            'tx_station_id': tx_station_id,
            'ghz_part': ghz_part,
            'freq_mod_1e9': freq_mod_1e9,
            'freq_frac': freq_frac
        }
        
    except struct.error:
        return None
    except Exception:
        return None

def read_ramp_table_from_binary(dat_file, offset, num_records, station_id):
    ramps = []
    
    try:
        file_size = os.path.getsize(dat_file)
        
        if offset >= file_size:
            print(f"ОШИБКА: offset {offset} вне файла!")
            return []
        
        expected_end = offset + num_records * 36
        if expected_end > file_size:
            num_records = (file_size - offset) // 36
        
        with open(dat_file, 'rb') as f:
            f.seek(offset)
            
            for i in range(num_records):
                record_bytes = f.read(36)
                if len(record_bytes) < 36:
                    break
                
                ramp = parse_ramp_record_binary_correct(record_bytes)
                if ramp:
                    ramp['station_id'] = station_id
                    ramp['record_index'] = i
                    ramps.append(ramp)
    
    except FileNotFoundError:
        print(f"Файл не найден: {dat_file}")
        return []
    except Exception as e:
        print(f"Ошибка чтения рамп-таблицы: {e}")
        return []
    
    # Сортируем по времени
    ramps.sort(key=lambda x: x['t_start'])
    
    return ramps

def get_ramp_tables_for_file(xml_file):
    ramps_offsets = read_ramp_offsets_from_xml(xml_file)
    dat_file = xml_file.replace('.xml', '.dat')
    
    all_ramps = {}
    
    for station_id, offsets in ramps_offsets.items():
        if 'data' in offsets and offsets['data'] is not None:
            offset = offsets['data']
            records = offsets.get('records', 0)
            
            if records > 0:
                ramps = read_ramp_table_from_binary(dat_file, offset, records, station_id)
                if ramps:
                    all_ramps[station_id] = ramps
                    print(f"Прочитана рамп-таблица станции {station_id}: {len(ramps)} записей")
    
    return all_ramps

def save_to_csv(df, output_filename="odf_data.csv"):
    if df.empty:
        print("Нет данных для сохранения!")
        return False
    
    try:
        df.to_csv(output_filename, index=False, encoding='utf-8')
        print(f"  Данные сохранены в CSV: {output_filename}")
        print(f"  Записей: {len(df)}")
        print(f"  Колонок: {len(df.columns)}")
        return True
    except Exception as e:
        print(f"Ошибка сохранения CSV: {e}")
        return False

def save_ramps_to_csv(all_ramps, output_filename="ramp_tables.csv"):
    if not all_ramps:
        print("Нет рамп-таблиц для сохранения!")
        return False
    
    all_ramp_records = []
    
    for station_id, ramps in all_ramps.items():
        for ramp in ramps:
            record = {
                'station_id': station_id,
                'ramp_start_time': ramp['t_start'],  # секунды от 1950
                'ramp_end_time': ramp['t_end'],      # секунды от 1950
                'ramp_rate': ramp['ramp_rate'],      # Гц/сек
                'ramp_start_freq': ramp['f_start'],  # Гц
                'tx_station_id': ramp.get('tx_station_id', station_id),
                'ghz_part': ramp.get('ghz_part', 0),
                'freq_mod_1e9': ramp.get('freq_mod_1e9', 0),
                'freq_frac': ramp.get('freq_frac', 0),
                'record_index': ramp.get('record_index', 0)
            }
            all_ramp_records.append(record)

    df_ramps = pd.DataFrame(all_ramp_records)
    df_ramps = df_ramps.sort_values(['station_id', 'ramp_start_time'])
    
    try:
        df_ramps.to_csv(output_filename, index=False, encoding='utf-8')
        print(f"Рамп-таблицы сохранены в CSV: {output_filename}")
        print(f"Записей: {len(df_ramps)}")
        return True
        
    except Exception as e:
        print(f"Ошибка сохранения рамп-таблиц: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    try:
        xml_file = 'messenger_pds_data/2015/mess_rs_15012_013_odf.xml'
        
        df_extracted = parse_to_df(xml_file)
        if not df_extracted.empty:
            success_odf = save_to_csv(df_extracted, "odf_data.csv")
        else:
            print("Нет данных ODF для сохранения")

        all_ramps = get_ramp_tables_for_file(xml_file)
        if all_ramps:
            success_ramps = save_ramps_to_csv(all_ramps, "ramp_tables.csv")
        else:
            print("Нет рамп-таблиц для сохранения")
        
    except FileNotFoundError as e:
        print(f"ОШИБКА: Файл не найден - {e}")
    except Exception as e:
        print(f"ОШИБКА: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()