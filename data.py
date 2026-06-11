import open3d as o3d
import numpy as np


def visualize_point_cloud(txt_file_path, has_color=False):
    """
    可视化txt格式的点云数据集

    参数:
        txt_file_path: txt文件路径
        has_color: 是否包含颜色信息（若为True，每行数据应为x y z r g b，r/g/b范围0-255）
    """
    try:
        # 读取txt文件，假设数据以空格/逗号分隔，无表头
        # 若有其他分隔符，可修改delimiter参数（如delimiter=','）
        data = np.loadtxt(txt_file_path, delimiter=' ')

        # 检查数据维度是否符合要求
        if has_color:
            if data.shape[1] != 6:
                raise ValueError("若has_color=True，每行数据必须包含6个值（x y z r g b）")
            # 提取坐标（前3列）和颜色（后3列，需归一化到0-1）
            points = data[:, :3]
            colors = data[:, 3:6] / 255.0  # 转换为0-1范围
        else:
            if data.shape[1] < 3:
                raise ValueError("若has_color=False，每行数据至少包含3个值（x y z）")
            points = data[:, :3]
            colors = None  # 无颜色时使用默认颜色

        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        num_points = len(pcd.points)
        base_dim = 3  # 坐标(x,y,z)固定为3维
        extra_dims = []  # 存储附加属性的维度信息

        # 检查是否有颜色（RGB）
        if pcd.has_colors():
            extra_dims.append("颜色(r,g,b) - 3维")
            base_dim += 3
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # 添加坐标系（可选，用于参考方向）
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        print("=" * 50)
        print(f"PLY点云维度信息：")
        print(f"1. 总点数：{num_points} 个")
        print(f"2. 单个点的维度组成：")
        print(f"   - 基础坐标(x,y,z) - 3维")
        if extra_dims:
            for dim in extra_dims:
                print(f"   - {dim}")
        else:
            print(f"   - 无附加属性（仅坐标）")
        print(f"3. 单个点的总维度：{base_dim} 维")
        print("=" * 50)

        # 可视化
        o3d.visualization.draw_geometries([pcd, coordinate_frame], window_name="点云可视化")

    except FileNotFoundError:
        print(f"错误：未找到文件 {txt_file_path}")
    except Exception as e:
        print(f"出错：{str(e)}")


# 示例用法
if __name__ == "__main__":
    # 替换为你的txt文件路径
    txt_path =r"E:\油桶口数据集\自制数据集\工训数据集\vpp000.txt"  # 例如："data/points.txt"

    # 若点云包含颜色信息（x y z r g b），设置has_color=True
    visualize_point_cloud(txt_path, has_color=False)