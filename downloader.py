#downloader.py

"""
Установка ODF-файлов MESSENGER с сервера NASA PDS
pip install requests beautifulsoup4
"""

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

def download_messenger_odf_files(base_url, year, download_dir='./messenger_odf'):
    """
    Скачивает ODF-файлы MESSENGER с PDS сервера NASA.
    
    Аргументы:
        base_url (str): Базовый URL директории, например 
                       'https://pds-geosciences.wustl.edu/messenger/urn-nasa-pds-mess-rs-raw/data-odf/'
        year (int/str): Год для скачивания
        download_dir (str): Локальная директория для сохранения файлов
    """
    
    # 1. Формируем полный URL для указанного года
    target_url = urljoin(base_url.rstrip('/') + '/', f"{year}/")
    print(f"Директория для скачивания: {target_url}")
    
    # 2. Создаем локальную директорию для года
    year_dir = os.path.join(download_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)
    print(f"Файлы будут сохранены в: {os.path.abspath(year_dir)}")
    
    try:
        # 3. Получаем содержимое веб-страницы
        response = requests.get(target_url, timeout=30)
        response.raise_for_status()  # Проверяем успешность запроса
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 5. Находим все ссылки на файлы
        file_links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and ('_odf.xml' in href or '_odf.dat' in href):
                full_url = urljoin(target_url, href)
                file_links.append(full_url)
        
        print(f"Найдено {len(file_links)} файлов.")
        
        # 6. Скачиваем файлы
        downloaded_count = 0
        for file_url in file_links:
            filename = os.path.basename(file_url)
            if not filename:
                continue
            
            local_path = os.path.join(year_dir, filename)
            
            # Проверяем, существует ли файл
            if os.path.exists(local_path):
                print(f"Файл уже существует: {filename}")
                continue
            
            try:
                file_response = requests.get(file_url, timeout=60)
                file_response.raise_for_status()
                
                # Сохраняем файл
                with open(local_path, 'wb') as f:
                    f.write(file_response.content)
                
                downloaded_count += 1
                print(f"Сохранен файл: {filename}")
                
                # Небольшая задержка, чтобы не перегружать сервер
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                print(f"Ошибка скачивания: {e}")
        
        print(f"\nЗагрузка завершена! Успешно скачано {downloaded_count} файлов в '{year_dir}'")
        
        # 7. Выводим список скачанных файлов
        if downloaded_count > 0:
            print("\nСписок скачанных файлов:")
            for f in os.listdir(year_dir):
                if os.path.isfile(os.path.join(year_dir, f)):
                    file_size = os.path.getsize(os.path.join(year_dir, f))
                    print(f"  - {f} ({file_size:,} байт)")
    
    except requests.exceptions.RequestException as e:
        print(f"Ошибка подключения к URL: {e}")
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")

if __name__ == "__main__":
    BASE_PDS_URL = "https://pds-geosciences.wustl.edu/messenger/urn-nasa-pds-mess-rs-raw/data-odf/" 
    
    # Локальная директория для сохранения
    DOWNLOAD_DIR = "./messenger_pds_data"
    
    print("=" * 60)
    print("УСТАНОВКА MESSENGER ODF ФАЙЛОВ")
    print("=" * 60)
    
    year_to_download=2015

    print(f"Задача: Скачать файлы за {year_to_download} год")
    print(f"Источник: {BASE_PDS_URL}")
    print("=" * 60)
    download_messenger_odf_files(BASE_PDS_URL, year_to_download, DOWNLOAD_DIR)