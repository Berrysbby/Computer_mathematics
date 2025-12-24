# parser.py

import numpy as np
import pandas as pd
from pds4_tools import pds4_read
from datetime import datetime, timedelta
import glob

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
        if items_6_14['network_id'] == 0 and items_6_14['data_type_id'] == 37 and items_6_14['data_validity'] == 0 and items_6_14['format_id'] == 2:
            extracted_record = {
                'time_tag_seconds': time_tag_int,
                'record_time': record_time,
                'full_observable': full_observable,
                'observable_int': observable_int,
                'observable_frac': observable_frac,
                'receiving_station_id': items_6_14['receiving_station_id'],
                'transmitting_station_id': items_6_14['transmitting_station_id'],
                'ref_freq_band_id': items_6_14['ref_freq_band_id'],
                'full_ref_freq': full_ref_freq,
                'ref_freq_int': items_15_19['item_18'],
                'ref_freq_frac': items_15_19['item_19'],
                'data_type_id': items_6_14['data_type_id'],
                'data_validity': items_6_14['data_validity'],
                'format_id': items_6_14['format_id'],
            }
            
            extracted_data.append(extracted_record)
    
    df_extracted = pd.DataFrame(extracted_data)
    return df_extracted 

def save_to_csv(df, output_filename="odf_data.csv"):
    """Сохраняет DataFrame в CSV файл"""
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
    
if __name__ == "__main__":

    try:
        df_extracted = parse_to_df('messenger_pds_data/2015/mess_rs_15012_013_odf.xml')
        
        success = save_to_csv(df_extracted, "odf_data.csv")
        
        if success:
            print("  Обработка завершена успешно.")
        else:
            print("  Ошибка при сохранении данных")
            
    except FileNotFoundError as e:
        print(f"Ошибка: Файл не найден - {e}")
    except Exception as e:
        print(f"Ошибка при обработке: {e}")