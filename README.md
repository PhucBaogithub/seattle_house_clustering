# Hệ Thống Dự Đoán Phân Cụm Giá Nhà Seattle

Dự án ứng dụng thuật toán DBSCAN để phân cụm và dự đoán giá nhà tại Seattle, Washington với giao diện web tương tác.

## Tổng Quan Dự Án

### Mục Tiêu
- Phân cụm thị trường bất động sản Seattle thành các nhóm có đặc điểm tương tự
- Dự đoán phân khúc thị trường cho bất động sản mới
- Cung cấp phân tích kinh doanh và tư vấn đầu tư

### Đặc Điểm Chính
- Sử dụng thuật toán DBSCAN với tham số tối ưu
- Giao diện web thân thiện với Streamlit
- Trực quan hóa dữ liệu 3D và biểu đồ so sánh
- Phân tích phân khúc kinh doanh chi tiết
- Đánh giá độ tin cậy và khoảng cách cluster

## Lý Thuyết Thuật Toán

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

**Nguyên lý hoạt động:**
- Thuật toán clustering dựa trên mật độ điểm dữ liệu
- Tự động xác định số cluster và phát hiện noise/outlier
- Phù hợp với dữ liệu có hình dạng không đều và nhiễu

**Tham số chính:**
- `eps`: Bán kính vùng lân cận (0.7)
- `min_samples`: Số điểm tối thiểu để tạo cluster (120)

**Ưu điểm:**
- Không cần biết trước số cluster
- Phát hiện được outlier
- Hiệu quả với cluster có hình dạng bất kỳ
- Ổn định với nhiễu

**Nhược điểm:**
- Nhạy cảm với tham số eps và min_samples
- Khó xử lý dữ liệu có mật độ khác nhau
- Độ phức tạp cao với dữ liệu lớn

### Tiền Xử Lý Dữ Liệu

**Log Transformation:**
- Áp dụng cho `price` và `sqft_living`
- Giảm độ lệch phân phối
- Công thức: `log(1 + x)`

**RobustScaler:**
- Chuẩn hóa dữ liệu dựa trên median và IQR
- Ít nhạy cảm với outlier
- Công thức: `(x - median) / IQR`

### Đánh Giá Mô Hình

**Phương pháp dự đoán:**
- Sử dụng k-nearest neighbors
- Tìm k=120 điểm gần nhất
- Chọn cluster phổ biến nhất

**Độ tin cậy:**
- Tỷ lệ neighbors cùng cluster
- Công thức: `confidence = count_same_cluster / k`

## Cấu Trúc Dự Án

```
seattle-house-clustering/
├── streamlit_app.py          # Giao diện web chính
├── requirements.txt          # Thư viện cần thiết
├── README.md                # Tài liệu hướng dẫn
├── kc_house_data.csv        # Dữ liệu gốc (không bắt buộc)
└── DataMining (2).ipynb     # Jupyter notebook phân tích
```

## Cài Đặt và Chạy

### Yêu Cầu Hệ Thống
- Python 3.7+
- Pip package manager

### Cài Đặt Thư Viện

```bash
# Clone repository
git clone https://github.com/PhucBaogithub/seattle_house_clustering.git
cd seattle-house-clustering

# Cài đặt dependencies
pip install -r requirements.txt
```

### Chạy Ứng Dụng Web

```bash
# Khởi động Streamlit
streamlit run streamlit_app.py

# Mở trình duyệt tại địa chỉ
# http://localhost:8501
```

### Chạy Jupyter Notebook

```bash
# Khởi động Jupyter
jupyter notebook

# Mở file DataMining.ipynb
# Chạy từng cell theo thứ tự
```

## Hướng Dẫn Sử Dụng

### Giao Diện Web

1. **Nhập thông tin nhà:**
   - Giá nhà: $50,000 - $10,000,000
   - Diện tích sống: 500 - 10,000 sqft
   - Chất lượng xây dựng: 1-13

2. **Dự đoán cluster:**
   - Nhấn "Dự Đoán Phân Cụm"
   - Xem kết quả và độ tin cậy
   - Phân tích so sánh với cluster

3. **Trực quan hóa:**
   - Biểu đồ so sánh 4 panel
   - Vị trí 3D trong không gian feature
   - Phân tích phân khúc kinh doanh

### Jupyter Notebook

1. **Cell 71 - Phân tích kinh doanh:**
   - Lý thuyết clustering
   - Phân tích từng cluster
   - Chiến lược marketing

2. **Cell 72 - Tổng kết lý thuyết:**
   - Chi tiết thuật toán DBSCAN
   - Công thức đánh giá
   - So sánh với thuật toán khác

3. **Cell 73 - Giao diện tương tác:**
   - Class HousePriceClustering
   - Demo dự đoán
   - Visualization

## Kết Quả Đạt Được

### Hiệu Suất Mô Hình
- **Số cluster**: 6 cluster chính
- **Silhouette Score**: 0.321
- **Tỷ lệ noise**: ~5%
- **Độ ổn định**: Cao với cross-validation

### Phân Khúc Thị Trường
- **Giá rẻ**: < $300,000 (30% thị trường)
- **Trung bình**: $300,000 - $600,000 (40% thị trường)
- **Cao cấp**: $600,000 - $1,000,000 (20% thị trường)
- **Siêu cao cấp**: > $1,000,000 (10% thị trường)

### Tính Năng Nổi Bật
- Dự đoán real-time với độ tin cậy cao
- Phân tích so sánh trực quan
- Tư vấn đầu tư thông minh
- Phát hiện nhà có đặc điểm đặc biệt

## Công Nghệ Sử Dụng

### Machine Learning
- **Scikit-learn**: Thuật toán DBSCAN, preprocessing
- **NumPy**: Xử lý mảng số học
- **Pandas**: Thao tác dữ liệu

### Visualization
- **Plotly**: Biểu đồ tương tác 3D
- **Matplotlib**: Biểu đồ cơ bản
- **Seaborn**: Biểu đồ thống kê

### Web Framework
- **Streamlit**: Giao diện web tương tác
- **HTML/CSS**: Tùy chỉnh giao diện

## Phát Triển Trong Tương Lai

### Cải Tiến Mô Hình
- Thêm feature địa lý (lat, long)
- Tích hợp deep learning
- Ensemble với nhiều thuật toán

### Tính Năng Mới
- API RESTful
- Database tích hợp
- Notification system
- Mobile app

### Mở Rộng Dữ Liệu
- Nhiều thành phố khác
- Dữ liệu real-time
- Tích hợp với MLS

## Đóng Góp

Chúng tôi hoan nghênh mọi đóng góp cho dự án:

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## Tác Giả

**Phuc Bao**
- Email: baominecraft12344@gmail.com
- GitHub: PhucBaogithub

## Giấy Phép

Dự án sử dụng giấy phép MIT. Xem file LICENSE để biết thêm chi tiết.

## Tài Liệu Tham Khảo

1. Ester, M., et al. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise"
2. Scikit-learn Documentation: DBSCAN
3. Seattle Housing Data Analysis
4. Streamlit Documentation 
