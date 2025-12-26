# Divide Smart & Conquer - Ứng dụng So sánh Thuật toán Sắp xếp

## 🎯 Mục đích
Ứng dụng giáo dục để trực quan hóa và so sánh 3 thuật toán sắp xếp từ bài báo "Divide Smart and Conquer":
- **Algorithm 1**: Consecutive Increasing Runs
- **Algorithm 2**: Consecutive Monotonic Runs  
- **Algorithm 3**: Non-consecutive L/R

## 🚀 Cài đặt & Chạy

```bash
# Cài đặt dependencies
pip install streamlit pandas

# Chạy ứng dụng
streamlit run app.py
```

Ứng dụng sẽ mở tại: http://localhost:8501

**Lưu ý**: File `creditcard.csv` cần được đặt cùng thư mục với `app.py` để sử dụng tính năng load dữ liệu CSV.

## 📊 Tính năng

### 1. Xem chi tiết 1 thuật toán
- Chọn dataset từ preset hoặc nhập thủ công
- Chọn 1 trong 3 thuật toán
- Xem từng bước thực thi với tree visualization
- Navigation: First/Prev/Next/Last + slider

### 2. So sánh 3 thuật toán
- Chạy cùng 1 dataset qua cả 3 thuật toán
- So sánh metrics:
  - ⏱️ Thời gian thực thi
  - 🔢 Số runs phát hiện được
  - 🔀 Số lần merge
  - 📝 Tổng số bước
- Biểu đồ so sánh trực quan
- Khuyến nghị thuật toán phù hợp nhất

## 📂 Preset Datasets (Ví dụ thực tế)

### IoT Sensors - Nhiệt độ từ nhiều cảm biến
```
[18, 20, 22, 25, 28, 30, 15, 17, 19, 21, 19, 21, 24, 27, 29]
```
**Đặc điểm**: Mỗi cảm biến đo nhiệt độ tăng dần, khi chuyển sensor thì reset  
**Thuật toán tốt nhất**: Algorithm 1 (phát hiện được nhiều dãy tăng liên tiếp)

### Banking - Lịch sử giao dịch số dư
```
[100, 150, 200, 250, 230, 210, 180, 150, 180, 200, 250, 300]
```
**Đặc điểm**: Nạp tiền (tăng) và rút tiền (giảm) xen kẽ  
**Thuật toán tốt nhất**: Algorithm 2 (tận dụng được cả dãy tăng & giảm)

### Stock Market - Giá cổ phiếu
```
[100, 105, 110, 115, 112, 108, 104, 100, 105, 110, 115, 120]
```
**Đặc điểm**: Xu hướng tăng → điều chỉnh giảm → phục hồi  
**Thuật toán tốt nhất**: Algorithm 2 (phát hiện xu hướng đảo chiều)

### Student Scores - Điểm thi xen kẽ
```
[7, 8, 9, 5, 4, 3, 8, 9, 10, 6, 5, 4]
```
**Đặc điểm**: Môn dễ (điểm cao) xen kẽ môn khó (điểm thấp)  
**Thuật toán tốt nhất**: Algorithm 3 (xử lý tốt dữ liệu xen kẽ phức tạp)

### E-commerce - Giá sản phẩm theo mùa
```
[50, 60, 70, 80, 75, 70, 65, 60, 70, 80, 90, 100]
```
**Đặc điểm**: Tăng giá đầu mùa → giảm giữa mùa → tăng lại cuối mùa  
**Thuật toán tốt nhất**: Algorithm 2 (theo dãy xu hướng)

### Credit Card - Số tiền giao dịch (từ CSV)
**File**: `creditcard.csv` (cột Amount)  
**Đặc điểm**: Dữ liệu giao dịch thẻ tín dụng thực tế, có thể chọn số lượng và vị trí records  
**Tùy chọn**:
- Số lượng giao dịch: 10-100 (khuyến nghị ≤ 30)
- Bỏ qua rows đầu: để lấy dữ liệu ở vị trí khác
- Preview dữ liệu trước khi load

**Thuật toán tốt nhất**: Tùy thuộc vào pattern của dãy được chọn

## 🎓 Hướng dẫn sử dụng cho giảng viên

### Demo trong lớp - Chế độ So sánh
1. **Mở sidebar** → Chọn "📊 So sánh 3 thuật toán"
2. **Chọn dataset** phù hợp với bài giảng (ví dụ: IoT Sensors)
3. **Click "🚀 Chạy so sánh 3 thuật toán"**
4. **Phân tích kết quả**:
   - Xem bảng metrics 3 cột
   - So sánh biểu đồ thời gian, runs, merges
   - Đọc phần "💡 Phân tích & Khuyến nghị"
5. **Thảo luận**: Tại sao Algorithm X tốt hơn với dataset này?

### Demo chi tiết 1 thuật toán
1. **Mở sidebar** → Chọn "🔍 Xem chi tiết 1 thuật toán"
2. **Chọn dataset** và thuật toán cụ thể
3. **Click "▶️ Chạy thuật toán"**
4. **Navigate từng bước**:
   - Dùng nút Next/Prev để xem từng bước
   - Quan sát tree visualization thay đổi
   - Đọc message giải thích từng bước
5. **Xem JSON details** để hiểu cấu trúc dữ liệu

## 📖 Giải thích Metrics

- **Số runs**: Số dãy con đơn điệu được phát hiện (càng ít càng tốt)
- **Số merges**: Tổng số lần ghép dãy (phản ánh độ phức tạp)
- **Thời gian**: Hiệu suất thực tế (milliseconds)
- **Độ dài run**: Thống kê về kích thước các dãy con

## 💡 Tips

- **Kích thước mảng**: Dùng n = 10-20 cho dễ quan sát
- **Algorithm 3**: Giới hạn n ≤ 20 để dễ theo dõi L/R building
- **Dataset thực tế**: Giúp sinh viên hiểu ứng dụng thực tiễn
- **So sánh nhiều lần**: Thử các dataset khác nhau để thấy sự khác biệt

## 🏗️ Cấu trúc Project

```
baicuoiki/
├── app.py                           # Ứng dụng Streamlit (single-file)
└── README.md                        # File này
```

## 📚 Tài liệu kỹ thuật

- Kiến trúc hệ thống
- Cấu trúc dữ liệu TreeNode
- Logic từng thuật toán
- Session state management
- Visualization strategy
