import numpy as np
import open3d as o3d

# 读取txt点云
points = np.loadtxt(r"E:\油桶口数据集\自制数据集\工训数据集\vpp000.txt")

# 创建点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])

# 去噪（可选）
pcd, _ = pcd.remove_statistical_outlier(
    nb_neighbors=30,
    std_ratio=2.0
)
pcd = pcd.voxel_down_sample(voxel_size=0.01)
# 估计法向量
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.05,
        max_nn=30
    )
)

# Poisson重建
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd,
    depth=9
)

# 删除低密度区域
densities = np.asarray(densities)
vertices_to_remove = densities < np.quantile(densities, 0.01)
mesh.remove_vertices_by_mask(vertices_to_remove)

# 保存网格
o3d.io.write_triangle_mesh("mesh.obj", mesh)

# 显示
o3d.visualization.draw_geometries([mesh])
o3d.visualization.draw_geometries([pcd])