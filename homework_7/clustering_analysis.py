"""
Homework 7：聚类分析任务
对ESI数据进行聚类，发现与华东师范大学类似的学校
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import matplotlib
# 显式使用中文字体（优先使用系统已安装的 SimHei / Microsoft YaHei）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False # 解决负号变方块的问题

# 导入自定义工具函数
from utils import (
    load_esi_dataframe, clean_and_enrich, build_institution_features
)

def find_optimal_clusters(X, max_k=15):
    """
    寻找最优聚类数量
    使用轮廓系数和肘部法则
    """
    silhouette_scores = []
    inertias = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # 计算轮廓系数
        if len(set(labels)) > 1:  # 确保有多个簇
            silhouette_scores.append(silhouette_score(X, labels))
        else:
            silhouette_scores.append(0)
        
        inertias.append(kmeans.inertia_)
    
    # 找到最优k（轮廓系数最大）
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    return optimal_k, silhouette_scores, inertias, k_range

def perform_clustering(feature_matrix, n_clusters=None):
    """
    执行K-means聚类
    """
    if n_clusters is None:
        n_clusters, _, _, _ = find_optimal_clusters(feature_matrix.values)
        print(f"自动选择最优聚类数量: {n_clusters}")
    
    # 执行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(feature_matrix.values)
    
    # 创建聚类结果DataFrame
    clustering_results = pd.DataFrame({
        'institution': feature_matrix.index,
        'cluster': cluster_labels
    }).set_index('institution')
    
    return clustering_results, kmeans

def find_similar_institutions(target_institution, feature_matrix, clustering_results, n_similar=10):
    """
    寻找与目标学校相似的高校
    使用余弦相似度
    """
    if target_institution not in feature_matrix.index:
        print(f"未找到目标学校: {target_institution}")
        return None
    
    # 计算所有学校与目标学校的余弦相似度
    target_features = feature_matrix.loc[target_institution].values.reshape(1, -1)
    all_features = feature_matrix.values
    
    similarities = cosine_similarity(target_features, all_features)[0]
    
    # 创建相似度DataFrame
    similarity_df = pd.DataFrame({
        'institution': feature_matrix.index,
        'cosine_similarity': similarities,
        'cluster': clustering_results.loc[feature_matrix.index, 'cluster'].values
    })
    
    # 排除目标学校自身，按相似度排序
    similar_schools = similarity_df[similarity_df['institution'] != target_institution]
    similar_schools = similar_schools.sort_values('cosine_similarity', ascending=False).head(n_similar)
    
    return similar_schools

def analyze_cluster_characteristics(clustering_results, feature_matrix, df_clean):
    """
    分析每个簇的特征
    """
    # 合并聚类结果与原始数据
    cluster_analysis = clustering_results.copy()
    
    # 添加国家/地区信息
    country_info = df_clean.groupby('institution')['country_region'].first()
    cluster_analysis['country_region'] = country_info
    
    # 计算每个簇的平均特征
    cluster_means = feature_matrix.groupby(clustering_results['cluster']).mean()
    
    # 统计每个簇的学校数量和国家分布
    cluster_stats = cluster_analysis.groupby('cluster').agg({
        'country_region': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
    }).rename(columns={'country_region': 'most_common_country'})
    
    # 添加学校数量统计
    cluster_counts = cluster_analysis.groupby('cluster').size().rename('count')
    cluster_stats = pd.concat([cluster_stats, cluster_counts], axis=1)
    
    return cluster_analysis, cluster_means, cluster_stats

def visualize_clustering_results(feature_matrix, clustering_results, target_institution, similar_schools):
    """
    可视化聚类结果
    """
    # 使用PCA降维进行可视化
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(feature_matrix.values)
    
    # 创建可视化DataFrame
    viz_df = pd.DataFrame({
        'PC1': features_2d[:, 0],
        'PC2': features_2d[:, 1],
        'cluster': clustering_results['cluster'],
        'institution': feature_matrix.index
    })
    
    # 标记目标学校和相似学校
    viz_df['is_target'] = viz_df['institution'] == target_institution
    viz_df['is_similar'] = viz_df['institution'].isin(similar_schools['institution'].values)
    
    plt.figure(figsize=(15, 10))
    
    # 主聚类图
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(viz_df['PC1'], viz_df['PC2'], c=viz_df['cluster'], 
                         cmap='tab10', alpha=0.6, s=30)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('高校聚类结果 (PCA降维)')
    
    # 标记目标学校和相似学校
    target_point = viz_df[viz_df['is_target']]
    similar_points = viz_df[viz_df['is_similar']]
    
    if not target_point.empty:
        plt.scatter(target_point['PC1'], target_point['PC2'], 
                   color='red', s=200, marker='*', label='ECNU')
    
    if not similar_points.empty:
        plt.scatter(similar_points['PC1'], similar_points['PC2'], 
                   color='orange', s=100, marker='^', label='Similar to ECNU')
    
    plt.legend()
    
    # 相似度分布
    plt.subplot(2, 2, 2)
    plt.hist(similar_schools['cosine_similarity'], bins=20, alpha=0.7, color='skyblue')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('相似学校相似度分布')
    
    # 簇大小分布
    plt.subplot(2, 2, 3)
    cluster_sizes = clustering_results['cluster'].value_counts().sort_index()
    plt.bar(cluster_sizes.index, cluster_sizes.values, alpha=0.7, color='lightgreen')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Institutions')
    plt.title('各簇学校数量分布')
    
    # 相似学校的簇分布
    plt.subplot(2, 2, 4)
    similar_clusters = similar_schools['cluster'].value_counts()
    plt.pie(similar_clusters.values, labels=similar_clusters.index, autopct='%1.1f%%')
    plt.title('相似学校的簇分布')
    
    plt.tight_layout()
    plt.savefig('results/clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印PCA解释方差
    print(f"PCA解释方差比例: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")

def main():
    """主函数：执行聚类分析"""
    print("开始聚类分析...")
    
    # 1. 数据加载与预处理
    print("步骤1: 数据加载与预处理")
    df_raw = load_esi_dataframe(force_reload=False)
    df_clean = clean_and_enrich(df_raw)
    print(f"处理后的数据包含 {len(df_clean)} 条记录，{df_clean['institution'].nunique()} 所高校")
    
    # 2. 构建特征矩阵
    print("步骤2: 构建高校特征矩阵")
    feature_matrix, scaler = build_institution_features(df_clean)
    print(f"特征矩阵形状: {feature_matrix.shape}")
    
    # 3. 寻找最优聚类数量
    print("步骤3: 寻找最优聚类数量")
    optimal_k, silhouette_scores, inertias, k_range = find_optimal_clusters(feature_matrix.values)
    print(f"推荐聚类数量: {optimal_k}")
    
    # 可视化聚类评估指标
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_range, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('轮廓系数法')
    plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.plot(k_range, inertias, 'ro-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('肘部法则')
    plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # 4. 执行聚类
    print("步骤4: 执行K-means聚类")
    clustering_results, kmeans_model = perform_clustering(feature_matrix, n_clusters=optimal_k)
    print(f"聚类完成，共 {optimal_k} 个簇")
    
    # 5. 寻找与ECNU相似的学校
    print("步骤5: 寻找与华东师范大学相似的学校")
    target_institution = "EAST CHINA NORMAL UNIVERSITY"
    similar_schools = find_similar_institutions(target_institution, feature_matrix, clustering_results, n_similar=15)
    
    if similar_schools is not None:
        print(f"\n与 {target_institution} 最相似的15所学校:")
        print(similar_schools[['institution', 'cosine_similarity', 'cluster']].head(15))
        
        # 保存相似学校结果
        import os
        os.makedirs("results", exist_ok=True)
        similar_schools.to_csv("results/similar_schools_dl.csv", index=False)
        
        # 6. 分析簇特征
        print("步骤6: 分析簇特征")
        cluster_analysis, cluster_means, cluster_stats = analyze_cluster_characteristics(
            clustering_results, feature_matrix, df_clean
        )
        
        print("\n各簇统计信息:")
        print(cluster_stats)
        
        # 7. 可视化结果
        print("步骤7: 可视化聚类结果")
        visualize_clustering_results(feature_matrix, clustering_results, target_institution, similar_schools)
        
        # 8. 保存聚类结果
        clustering_results.to_csv("results/institution_clusters_dl.csv")
        print("\n聚类分析完成！")
        print(f"- 共发现 {optimal_k} 个高校群体")
        print(f"- 找到了 {len(similar_schools)} 所与ECNU相似的学校")
        print(f"- 结果已保存到 results/ 目录")
        
        # 打印相似学校的详细信息
        print(f"\n与ECNU最相似的5所学校:")
        top_similar = similar_schools.head()
        for idx, row in top_similar.iterrows():
            print(f"  {row['institution']} (相似度: {row['cosine_similarity']:.3f}, 簇: {row['cluster']})")
    
    else:
        print("未找到目标学校或聚类失败")

if __name__ == "__main__":
    main()