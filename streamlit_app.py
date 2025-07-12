import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class HouseClustering:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.X_scaled = None
        self.cluster_info = None
        self.features = ['price', 'sqft_living', 'grade']
        
    def train_model(self, df_data):
        """Huấn luyện mô hình DBSCAN"""
        X_raw = df_data[self.features].copy()
        
        # Log transformation
        X_transformed = X_raw.copy()
        X_transformed['price'] = np.log1p(X_raw['price'])
        X_transformed['sqft_living'] = np.log1p(X_raw['sqft_living'])
        
        # Scaling
        self.scaler = RobustScaler()
        self.X_scaled = self.scaler.fit_transform(X_transformed)
        
        # Training DBSCAN
        self.model = DBSCAN(eps=0.7, min_samples=120)
        cluster_labels = self.model.fit_predict(self.X_scaled)
        
        # Lưu thông tin clusters
        df_with_clusters = df_data[self.features].copy()
        df_with_clusters['cluster'] = cluster_labels
        
        # Tính toán thống kê cho từng cluster
        self.cluster_info = {}
        for cluster in sorted(set(cluster_labels)):
            if cluster != -1:
                cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster]
                self.cluster_info[cluster] = {
                    'count': len(cluster_data),
                    'price_mean': cluster_data['price'].mean(),
                    'price_min': cluster_data['price'].min(),
                    'price_max': cluster_data['price'].max(),
                    'sqft_mean': cluster_data['sqft_living'].mean(),
                    'grade_mean': cluster_data['grade'].mean(),
                    'percentage': len(cluster_data) / len(df_with_clusters[df_with_clusters['cluster'] != -1]) * 100
                }
        
        return len(self.cluster_info)
        
    def predict(self, price, sqft_living, grade):
        """Dự đoán cluster cho một mẫu mới"""
        if self.model is None:
            raise ValueError("Mô hình chưa được huấn luyện!")
            
        new_sample = pd.DataFrame({
            'price': [price],
            'sqft_living': [sqft_living],
            'grade': [grade]
        })
        
        # Log transformation
        new_sample_transformed = new_sample.copy()
        new_sample_transformed['price'] = np.log1p(new_sample['price'])
        new_sample_transformed['sqft_living'] = np.log1p(new_sample['sqft_living'])
        
        # Scaling
        new_sample_scaled = self.scaler.transform(new_sample_transformed)
        
        # Dự đoán bằng k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.model.min_samples)
        nbrs.fit(self.X_scaled)
        distances, indices = nbrs.kneighbors(new_sample_scaled)
        
        neighbor_labels = self.model.labels_[indices[0]]
        valid_labels = neighbor_labels[neighbor_labels != -1]
        
        if len(valid_labels) == 0:
            predicted_cluster = -1
            confidence = 0
        else:
            label_counts = Counter(valid_labels)
            predicted_cluster = label_counts.most_common(1)[0][0]
            confidence = label_counts.most_common(1)[0][1] / len(neighbor_labels)
            
        return predicted_cluster, confidence, distances[0].mean()
    
    def get_cluster_info(self, cluster):
        if cluster in self.cluster_info:
            return self.cluster_info[cluster]
        return None
    
    def classify_segment(self, cluster):
        if cluster == -1:
            return "Outlier", "Nhà có đặc điểm đặc biệt"
            
        cluster_data = self.cluster_info.get(cluster)
        if not cluster_data:
            return "Unknown", "Không xác định"
            
        avg_price = cluster_data['price_mean']
        
        if avg_price < 300000:
            return "Giá rẻ", "Phù hợp cho người mua lần đầu"
        elif avg_price < 600000:
            return "Trung bình", "Cân bằng giữa giá cả và chất lượng"
        elif avg_price < 1000000:
            return "Cao cấp", "Chất lượng tốt, thu nhập cao"
        else:
            return "Siêu cao cấp", "Phân khúc luxury"

@st.cache_data
def load_data():
    """Load và preprocess dữ liệu"""
    try:
        df = pd.read_csv('kc_house_data.csv')
        # Giả sử có preprocessing cơ bản
        df_cleaned = df.dropna()
        return df_cleaned
    except:
        # Tạo dữ liệu giả để demo
        np.random.seed(42)
        n_samples = 1000
        
        prices = np.random.lognormal(mean=12.5, sigma=0.8, size=n_samples)
        sqft = np.random.normal(2000, 800, n_samples)
        sqft = np.clip(sqft, 500, 8000)
        grades = np.random.choice(range(3, 14), n_samples, 
                                p=[0.05, 0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.05, 0.03, 0.01, 0.01])
        
        df_fake = pd.DataFrame({
            'price': prices,
            'sqft_living': sqft,
            'grade': grades
        })
        return df_fake

@st.cache_resource
def initialize_model():
    """Khởi tạo và huấn luyện mô hình"""
    df = load_data()
    clustering = HouseClustering()
    n_clusters = clustering.train_model(df)
    return clustering, n_clusters

def create_comparison_plots(clustering, price, sqft_living, grade, predicted_cluster):
    """Tạo biểu đồ so sánh"""
    clusters = list(clustering.cluster_info.keys())
    prices = [clustering.cluster_info[c]['price_mean'] for c in clusters]
    sqfts = [clustering.cluster_info[c]['sqft_mean'] for c in clusters]
    grades = [clustering.cluster_info[c]['grade_mean'] for c in clusters]
    counts = [clustering.cluster_info[c]['count'] for c in clusters]
    
    # Tạo subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('So sánh Giá', 'So sánh Diện tích', 'So sánh Chất lượng', 'Phân bố Clusters'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "pie"}]]
    )
    
    # Bar chart 1: Giá
    colors1 = ['red' if c == predicted_cluster else 'lightblue' for c in clusters]
    fig.add_trace(go.Bar(x=clusters, y=prices, marker_color=colors1, name="Giá TB", showlegend=False), row=1, col=1)
    fig.add_hline(y=price, line_dash="dash", line_color="red", row=1, col=1)
    
    # Bar chart 2: Diện tích
    colors2 = ['red' if c == predicted_cluster else 'lightgreen' for c in clusters]
    fig.add_trace(go.Bar(x=clusters, y=sqfts, marker_color=colors2, name="Diện tích TB", showlegend=False), row=1, col=2)
    fig.add_hline(y=sqft_living, line_dash="dash", line_color="red", row=1, col=2)
    
    # Bar chart 3: Chất lượng
    colors3 = ['red' if c == predicted_cluster else 'lightyellow' for c in clusters]
    fig.add_trace(go.Bar(x=clusters, y=grades, marker_color=colors3, name="Chất lượng TB", showlegend=False), row=2, col=1)
    fig.add_hline(y=grade, line_dash="dash", line_color="red", row=2, col=1)
    
    # Pie chart: Phân bố
    pie_colors = ['red' if c == predicted_cluster else 'lightcoral' for c in clusters]
    fig.add_trace(go.Pie(labels=[f'Cluster {c}' for c in clusters], values=counts, 
                        marker_colors=pie_colors, showlegend=False), row=2, col=2)
    
    fig.update_layout(height=600, title_text="Phân Tích So Sánh Với Các Clusters")
    return fig

def create_position_plot(clustering, price, sqft_living, grade):
    """Tạo biểu đồ vị trí 3D"""
    clusters = list(clustering.cluster_info.keys())
    prices = [clustering.cluster_info[c]['price_mean'] for c in clusters]
    sqfts = [clustering.cluster_info[c]['sqft_mean'] for c in clusters]
    grades = [clustering.cluster_info[c]['grade_mean'] for c in clusters]
    
    fig = go.Figure()
    
    # Thêm các clusters
    fig.add_trace(go.Scatter3d(
        x=sqfts, y=grades, z=prices,
        mode='markers',
        marker=dict(size=8, color=clusters, colorscale='viridis', opacity=0.7),
        text=[f'Cluster {c}' for c in clusters],
        name='Clusters'
    ))
    
    # Thêm nhà mới
    fig.add_trace(go.Scatter3d(
        x=[sqft_living], y=[grade], z=[price],
        mode='markers',
        marker=dict(size=15, color='red', symbol='diamond'),
        text=['Nhà mới'],
        name='Nhà mới'
    ))
    
    fig.update_layout(
        title='Vị Trí Của Nhà Mới Trong Không Gian 3D',
        scene=dict(
            xaxis_title='Diện Tích (sqft)',
            yaxis_title='Chất Lượng (Grade)',
            zaxis_title='Giá (USD)'
        ),
        height=600
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Dự Đoán Phân Cụm Giá Nhà Seattle",
        layout="wide"
    )
    
    st.title(" Hệ Thống Dự Đoán Phân Cụm Giá Nhà Seattle")
    st.markdown("### Sử dụng thuật toán DBSCAN để phân loại nhà theo giá, diện tích và chất lượng")
    
    # Khởi tạo mô hình
    with st.spinner("Đang khởi tạo mô hình..."):
        clustering, n_clusters = initialize_model()
    
    st.success(f"Mô hình đã sẵn sàng với {n_clusters} clusters")
    
    # Sidebar: Input
    st.sidebar.header("Thông Tin Nhà Cần Dự Đoán")
    
    price = st.sidebar.number_input(
        "Giá nhà (USD)",
        min_value=50000,
        max_value=10000000,
        value=500000,
        step=10000,
        format="%d"
    )
    
    sqft_living = st.sidebar.number_input(
        "Diện tích sống (sqft)",
        min_value=500,
        max_value=10000,
        value=2000,
        step=100,
        format="%d"
    )
    
    grade = st.sidebar.selectbox(
        "Chất lượng xây dựng (1-13)",
        options=list(range(1, 14)),
        index=7  # default = 8
    )
    
    # Prediction button
    if st.sidebar.button("Dự Đoán Phân Cụm", type="primary"):
        try:
            predicted_cluster, confidence, distance = clustering.predict(price, sqft_living, grade)
            
            # Main content area
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Kết Quả Dự Đoán")
                
                if predicted_cluster == -1:
                    st.error("**Cluster**: Noise/Outlier")
                    st.info("Nhà này có đặc điểm đặc biệt, khó phân loại vào nhóm nào")
                else:
                    st.success(f"**Cluster dự đoán**: {predicted_cluster}")
                    st.metric("Độ tin cậy", f"{confidence:.1%}")
                    st.metric("Khoảng cách", f"{distance:.3f}")
                    
                    # Thông tin cluster
                    cluster_data = clustering.get_cluster_info(predicted_cluster)
                    if cluster_data:
                        st.subheader(f"Thông Tin Cluster {predicted_cluster}")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Số nhà", f"{cluster_data['count']:,}")
                            st.metric("Giá trung bình", f"${cluster_data['price_mean']:,.0f}")
                            st.metric("Diện tích TB", f"{cluster_data['sqft_mean']:,.0f} sqft")
                        
                        with col_b:
                            st.metric("Thị phần", f"{cluster_data['percentage']:.1f}%")
                            st.metric("Khoảng giá", f"${cluster_data['price_min']:,.0f} - ${cluster_data['price_max']:,.0f}")
                            st.metric("Chất lượng TB", f"{cluster_data['grade_mean']:.1f}/13")
                        
                        # Phân loại phân khúc
                        segment, advice = clustering.classify_segment(predicted_cluster)
                        st.subheader("Phân Khúc Kinh Doanh")
                        st.info(f"**{segment}**: {advice}")
                        
                        # So sánh
                        price_diff = ((price - cluster_data['price_mean']) / cluster_data['price_mean']) * 100
                        sqft_diff = ((sqft_living - cluster_data['sqft_mean']) / cluster_data['sqft_mean']) * 100
                        grade_diff = ((grade - cluster_data['grade_mean']) / cluster_data['grade_mean']) * 100
                        
                        st.subheader("So Sánh Với Cluster")
                        st.write(f"Giá: {price_diff:+.1f}% so với TB cluster")
                        st.write(f"Diện tích: {sqft_diff:+.1f}% so với TB cluster")
                        st.write(f"Chất lượng: {grade_diff:+.1f}% so với TB cluster")
            
            with col2:
                st.subheader("Thông Tin Đầu Vào")
                st.write(f"**Giá**: ${price:,}")
                st.write(f"**Diện tích**: {sqft_living:,} sqft")
                st.write(f"**Chất lượng**: {grade}/13")
                
                if predicted_cluster != -1:
                    st.subheader("Gợi Ý Tương Tự")
                    cluster_data = clustering.get_cluster_info(predicted_cluster)
                    if cluster_data:
                        st.write(f"Nếu quan tâm đến cluster {predicted_cluster}:")
                        st.write(f"- Giá: ${cluster_data['price_min']:,.0f} - ${cluster_data['price_max']:,.0f}")
                        st.write(f"- Diện tích: {cluster_data['sqft_mean']*0.8:,.0f} - {cluster_data['sqft_mean']*1.2:,.0f} sqft")
                        st.write(f"- Chất lượng: {max(1, cluster_data['grade_mean']-1):.0f} - {min(13, cluster_data['grade_mean']+1):.0f}")
            
            # Visualizations
            st.subheader("Phân Tích Trực Quan")
            
            tab1, tab2 = st.tabs(["So Sánh Clusters", "Vị Trí 3D"])
            
            with tab1:
                fig1 = create_comparison_plots(clustering, price, sqft_living, grade, predicted_cluster)
                st.plotly_chart(fig1, use_container_width=True)
            
            with tab2:
                fig2 = create_position_plot(clustering, price, sqft_living, grade)
                st.plotly_chart(fig2, use_container_width=True)
                
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {str(e)}")
    
    # Thông tin về mô hình
    with st.expander("Thông Tin Về Mô Hình"):
        st.markdown("""
        **Thuật toán**: DBSCAN (Density-Based Spatial Clustering)
        
        **Tham số tối ưu**:
        - eps = 0.7
        - min_samples = 120
        
        **Đặc trưng sử dụng**:
        - price: Giá nhà (đã log transform)
        - sqft_living: Diện tích sống (đã log transform)
        - grade: Chất lượng xây dựng (1-13)
        
        **Tiền xử lý**:
        - Log transformation cho price và sqft_living
        - RobustScaler cho chuẩn hóa dữ liệu
        
        **Dự đoán**: Sử dụng k-nearest neighbors để tìm cluster gần nhất
        """)

if __name__ == "__main__":
    main() 