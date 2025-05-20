import os
import requests

def download_file(url, destination_folder="data", filename=None):
    """تحميل ملف من الإنترنت."""
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    if filename is None:
        filename = url.split("/")[-1]
    
    file_path = os.path.join(destination_folder, filename)
    
    if not os.path.exists(file_path):
        print(f"جارٍ تنزيل {filename}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"تم تنزيل {filename} بنجاح.")
        else:
            print(f"فشل تنزيل {filename}.")
    else:
        print(f"{filename} موجودة بالفعل.")

# قائمة بالملفات المطلوبة
required_files = [
    {"url": "https://example.com/stock_data.csv", "filename": "stock_data.csv"},
    {"url": "https://example.com/macro_data.json", "filename": "macro_data.json"}
]

# تحميل الملفات المطلوبة
for file_info in required_files:
    download_file(file_info["url"], filename=file_info["filename"])

print("تم تنزيل جميع الملفات المطلوبة.")