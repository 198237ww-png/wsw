import numpy as np
import matplotlib.pyplot as plt

def compute_angles(q, p, neighbors):
    v_main = p - q
    angles = []
    for x in neighbors:
        v = x - q
        cos_theta = np.dot(v_main, v) / (np.linalg.norm(v_main) * np.linalg.norm(v))
        theta = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        angles.append(theta)
    return sorted(angles)


def plot_structure(q, p, neighbors, title):
    plt.figure()

    # 中心点和主点
    plt.scatter(*q)
    plt.scatter(*p)

    # 主方向
    v_main = p - q
    plt.arrow(q[0], q[1], v_main[0], v_main[1], length_includes_head=True, head_width=0.1)

    # 邻居
    for x in neighbors:
        plt.scatter(*x)
        plt.arrow(q[0], q[1], *(x - q), length_includes_head=True, head_width=0.08)

    angles = compute_angles(q, p, neighbors)

    plt.title(title + f"\nAngles: {[round(a,1) for a in angles]}")
    plt.axis('equal')
    plt.grid()
    plt.show()

    return angles


# =========================
# ✅ 情况1：两个相似结构（L形 + 旋转）
# =========================

# 点云1（L）
q1 = np.array([0, 0])
p1 = np.array([2, 0])
neighbors1 = [
    np.array([0, 2]),
    np.array([1, 1])
]

angles1 = plot_structure(q1, p1, neighbors1, "Structure A (L-shape)")

# 点云2（旋转后的L）
q2 = np.array([0, 0])
p2 = np.array([0, 2])
neighbors2 = [
    np.array([-2, 0]),
    np.array([-1, 1])
]

angles2 = plot_structure(q2, p2, neighbors2, "Structure B (Rotated L-shape)")


# =========================
# ❌ 情况2：不同结构（直线）
# =========================

q3 = np.array([0, 0])
p3 = np.array([2, 0])
neighbors3 = [
    np.array([3, 0]),
    np.array([4, 0])
]

angles3 = plot_structure(q3, p3, neighbors3, "Structure C (Line)")


# =========================
# 📊 对比角度分布
# =========================
plt.figure()
plt.hist(angles1, bins=10, alpha=0.5, label="L-shape A")
plt.hist(angles2, bins=10, alpha=0.5, label="L-shape B (rotated)")
plt.hist(angles3, bins=10, alpha=0.5, label="Line")
plt.legend()
plt.title("Angle Distribution Comparison")
plt.show()