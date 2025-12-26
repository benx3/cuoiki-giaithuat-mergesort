# 📁 Hướng dẫn thiết lập dữ liệu

## File creditcard.csv

File `creditcard.csv` không được đưa vào repository vì kích thước quá lớn (143.84 MB > giới hạn 100MB của GitHub).

### Cách tải file:

**Option 1: Kaggle Dataset (Khuyến nghị)**
1. Truy cập: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click "Download" để tải file `creditcard.csv`
3. Giải nén và copy file `creditcard.csv` vào thư mục gốc của project (cùng cấp với `app.py`)

**Option 2: Google Drive (Nếu có)**
- Tải từ link được chia sẻ bởi giảng viên/nhóm

**Option 3: Dùng sample data nhỏ hơn**
- Tạo file CSV nhỏ hơn với cùng cấu trúc (cột Amount)
- Ứng dụng sẽ vẫn hoạt động bình thường

### Cấu trúc file yêu cầu:
```
creditcard.csv
├── Cột "Amount" (bắt buộc)
└── Các cột khác (tùy chọn, không sử dụng)
```

### Sau khi tải:
```
baicuoiki/
├── app.py
├── README.md
├── creditcard.csv      ← Đặt file vào đây
└── ...
```

Chạy lại `streamlit run app.py` để sử dụng dataset từ CSV!
