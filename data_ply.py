import sys
import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QSpinBox,
    QDoubleSpinBox,
    QMessageBox
)


class PointCloudGUI(QWidget):

    def __init__(self):
        super().__init__()

        self.pcd = None
        self.labels = None

        self.init_ui()

    def init_ui(self):

        self.setWindowTitle("点云分割系统")
        self.setGeometry(300, 200, 400, 300)

        # 文件路径
        self.path_label = QLabel("未加载文件")

        # eps参数
        self.eps_box = QDoubleSpinBox()
        self.eps_box.setValue(20)
        self.eps_box.setRange(0.1, 1000)
        self.eps_box.setSingleStep(0.1)

        # min_points参数
        self.min_points_box = QSpinBox()
        self.min_points_box.setValue(50)
        self.min_points_box.setRange(1, 10000)

        # 聚类ID
        self.cluster_box = QSpinBox()
        self.cluster_box.setValue(0)
        self.cluster_box.setRange(0, 1000)

        # 按钮
        self.load_btn = QPushButton("加载点云")
        self.segment_btn = QPushButton("开始分割")
        self.extract_btn = QPushButton("提取目标")
        self.save_btn = QPushButton("保存目标")

        # 绑定
        self.load_btn.clicked.connect(self.load_point_cloud)
        self.segment_btn.clicked.connect(self.segment_point_cloud)
        self.extract_btn.clicked.connect(self.extract_cluster)
        self.save_btn.clicked.connect(self.save_cluster)

        # 布局
        layout = QVBoxLayout()

        layout.addWidget(self.path_label)

        layout.addWidget(QLabel("eps"))
        layout.addWidget(self.eps_box)

        layout.addWidget(QLabel("min_points"))
        layout.addWidget(self.min_points_box)

        layout.addWidget(QLabel("提取聚类ID"))
        layout.addWidget(self.cluster_box)

        layout.addWidget(self.load_btn)
        layout.addWidget(self.segment_btn)
        layout.addWidget(self.extract_btn)
        layout.addWidget(self.save_btn)

        self.setLayout(layout)

    # --------------------------
    # 加载点云
    # --------------------------

    def load_point_cloud(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择点云文件",
            "",
            "Point Cloud (*.ply *.txt)"
        )

        if not file_path:
            return

        self.file_path = file_path

        ext = os.path.splitext(file_path)[1]

        try:

            if ext == ".txt":

                points = np.loadtxt(file_path)

                self.pcd = o3d.geometry.PointCloud()

                self.pcd.points = o3d.utility.Vector3dVector(
                    points[:, :3]
                )

            else:

                self.pcd = o3d.io.read_point_cloud(file_path)

            self.path_label.setText(file_path)

            QMessageBox.information(
                self,
                "成功",
                f"加载成功\n点数: {len(self.pcd.points)}"
            )

            o3d.visualization.draw_geometries(
                [self.pcd],
                window_name="原始点云"
            )

        except Exception as e:

            QMessageBox.critical(
                self,
                "错误",
                str(e)
            )

    # --------------------------
    # 点云分割
    # --------------------------

    def segment_point_cloud(self):

        if self.pcd is None:

            QMessageBox.warning(
                self,
                "警告",
                "请先加载点云"
            )

            return

        try:

            pcd = self.pcd.voxel_down_sample(
                voxel_size=0.01
            )

            eps = self.eps_box.value()

            min_points = self.min_points_box.value()

            self.labels = np.array(
                pcd.cluster_dbscan(
                    eps=eps,
                    min_points=min_points,
                    print_progress=True
                )
            )

            if self.labels.size == 0:
                raise ValueError("聚类失败")

            self.segmented_pcd = pcd

            max_label = self.labels.max()

            print(f"聚类数量: {max_label + 1}")

            # 上色
            colors = plt.get_cmap("tab20")(
                self.labels / (
                    max_label if max_label > 0 else 1
                )
            )

            colors[self.labels < 0] = 0

            pcd.colors = o3d.utility.Vector3dVector(
                colors[:, :3]
            )

            o3d.visualization.draw_geometries(
                [pcd],
                window_name="分割结果"
            )

            QMessageBox.information(
                self,
                "完成",
                f"检测到 {max_label + 1} 个聚类"
            )

        except Exception as e:

            QMessageBox.critical(
                self,
                "错误",
                str(e)
            )

    # --------------------------
    # 提取聚类
    # --------------------------

    def extract_cluster(self):

        if self.labels is None:

            QMessageBox.warning(
                self,
                "警告",
                "请先进行分割"
            )

            return

        try:

            target_label = self.cluster_box.value()

            idx = np.where(
                self.labels == target_label
            )[0]

            if len(idx) == 0:
                raise ValueError("该聚类不存在")

            self.target_pcd = self.segmented_pcd.select_by_index(idx)

            print(f"目标点数: {len(self.target_pcd.points)}")

            o3d.visualization.draw_geometries(
                [self.target_pcd],
                window_name=f"聚类 {target_label}"
            )

        except Exception as e:

            QMessageBox.critical(
                self,
                "错误",
                str(e)
            )

    # --------------------------
    # 保存点云
    # --------------------------

    def save_cluster(self):

        if not hasattr(self, "target_pcd"):

            QMessageBox.warning(
                self,
                "警告",
                "请先提取目标"
            )

            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存点云",
            "target.ply",
            "PLY Files (*.ply)"
        )

        if not save_path:
            return

        o3d.io.write_point_cloud(
            save_path,
            self.target_pcd
        )

        QMessageBox.information(
            self,
            "成功",
            "保存成功"
        )


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = PointCloudGUI()

    window.show()

    sys.exit(app.exec_())