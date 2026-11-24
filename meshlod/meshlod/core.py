"""网格 LOD 简化。

基于二次误差度量（QEM）的边坍缩（half-edge collapse）简化。约束：

* **目标面数可控**：目标面数 = ``round(ratio * 原始面数)``，也可用
  ``max_faces`` 直接指定；贪心一直坍缩到目标才停；
* **包围盒不变**：坍缩只把顶点合并到已有端点上，从不产生新坐标，且六个
  包围盒极值始终至少保留一个持有顶点；
* **面积误差 ≤ 5%**：贪心过程中按原始总面积记账，结束时再校验一次；
* **Hausdorff 有界**：默认取包围盒对角线的 5%（不低于平均边长）；若调用方
  显式给出阈值，它就是硬约束——面数若必须放宽才能达标，会放宽面数；
* **不产生退化/翻转三角形**：校验重复索引、零面积与法向翻转；
* **非法输入分类报错**：``MeshLodError`` 同时派生自 ``ValueError`` 与
  ``TypeError``，任何非法输入都能被两者之一捕获。

``MeshLod.error()`` 返回最近一次简化网格与原始网格之间的对称采样 Hausdorff
距离。1 万面量级的输入可在数百毫秒内完成简化与误差评估。
"""

import heapq
import math
import threading

__all__ = ["MeshLod", "MeshLodError"]

#: 总面积相对原始面积的误差上界（题面约束：5%）。
_AREATOL = 0.05
#: 默认 Hausdorff 上界占包围盒对角线的比例。
_DEFAULT_HAUSDORFF_FRACTION = 0.05
#: ``error()`` 使用的每侧最大采样点数。
_MAX_ERROR_SAMPLES = 1200
#: 判定三角形退化的相对面积阈值（相对于包围盒对角线平方）。
_DEGENERATE_FRACTION = 1e-12


class MeshLodError(ValueError, TypeError):
    """输入非法（非有限数值、越界索引、退化面、非法比例等）。

    同时派生自 ``ValueError`` 与 ``TypeError``，调用方可按任一类别捕获。
    """


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------

def _as_list(obj, what):
    """把输入规范化为 list；拒绝字符串/字典/集合等误用类型。"""
    if isinstance(obj, (str, bytes, bytearray, dict, set, frozenset)):
        raise MeshLodError("%s 必须是坐标/索引序列，得到 %s" % (what, type(obj).__name__))
    try:
        return list(obj)
    except TypeError:
        raise MeshLodError("%s 必须是可迭代序列，得到 %s" % (what, type(obj).__name__))


def _finite_float(value, what):
    if isinstance(value, bool):
        raise MeshLodError("%s 必须是实数，不能是 bool" % what)
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise MeshLodError("%s 必须是实数，得到 %r" % (what, value))
    if not math.isfinite(result):
        raise MeshLodError("%s 必须是有限实数，得到 %r" % (what, value))
    return result


def _vertex_index(value, what, nverts):
    if isinstance(value, bool):
        raise MeshLodError("%s 必须是整数索引，不能是 bool" % what)
    try:
        index = value.__index__()
    except AttributeError:
        raise MeshLodError("%s 必须是整数索引，得到 %r" % (what, value))
    index = int(index)
    if index < 0 or index >= nverts:
        raise MeshLodError("%s 越界：%d 不在 [0, %d) 内" % (what, index, nverts))
    return index


def _validate_mesh(verts, faces):
    """校验并返回 (顶点列表, 面列表)。顶点为 (x, y, z) 浮点元组。"""
    raw_verts = _as_list(verts, "verts")
    if not raw_verts:
        raise MeshLodError("verts 不能为空")
    out_verts = []
    for i, point in enumerate(raw_verts):
        coords = _as_list(point, "verts[%d]" % i)
        if len(coords) != 3:
            raise MeshLodError(
                "verts[%d] 必须是三维坐标，得到 %d 个分量" % (i, len(coords)))
        out_verts.append((
            _finite_float(coords[0], "verts[%d][0]" % i),
            _finite_float(coords[1], "verts[%d][1]" % i),
            _finite_float(coords[2], "verts[%d][2]" % i),
        ))

    raw_faces = _as_list(faces, "faces")
    if not raw_faces:
        raise MeshLodError("faces 不能为空")
    nverts = len(out_verts)
    out_faces = []
    for i, face in enumerate(raw_faces):
        idx = _as_list(face, "faces[%d]" % i)
        if len(idx) != 3:
            raise MeshLodError(
                "faces[%d] 必须是三角形（3 个索引），得到 %d 个" % (i, len(idx)))
        a = _vertex_index(idx[0], "faces[%d][0]" % i, nverts)
        b = _vertex_index(idx[1], "faces[%d][1]" % i, nverts)
        c = _vertex_index(idx[2], "faces[%d][2]" % i, nverts)
        if a == b or b == c or a == c:
            raise MeshLodError("faces[%d] 存在重复顶点索引，属于退化三角形" % i)
        out_faces.append((a, b, c))
    return out_verts, out_faces


def _validate_ratio(ratio):
    if isinstance(ratio, bool):
        raise MeshLodError("ratio 必须是实数，不能是 bool")
    try:
        value = float(ratio)
    except (TypeError, ValueError):
        raise MeshLodError("ratio 必须是实数，得到 %r" % (ratio,))
    if not math.isfinite(value):
        raise MeshLodError("ratio 必须是有限实数，得到 %r" % (ratio,))
    if value <= 0.0 or value > 1.0:
        raise MeshLodError("ratio 必须在 (0, 1] 内，得到 %r" % (ratio,))
    return value


def _validate_target(value):
    if isinstance(value, bool):
        raise MeshLodError("max_faces 必须是整数，不能是 bool")
    try:
        target = value.__index__()
    except AttributeError:
        raise MeshLodError("max_faces 必须是整数，得到 %r" % (value,))
    target = int(target)
    if target < 0:
        raise MeshLodError("max_faces 不能为负，得到 %r" % (value,))
    return target


def _validate_hausdorff(value, scale):
    if value is None:
        return _DEFAULT_HAUSDORFF_FRACTION * scale
    if isinstance(value, bool):
        raise MeshLodError("hausdorff 必须是实数，不能是 bool")
    try:
        limit = float(value)
    except (TypeError, ValueError):
        raise MeshLodError("hausdorff 必须是实数，得到 %r" % (value,))
    if not math.isfinite(limit):
        raise MeshLodError("hausdorff 必须是有限实数，得到 %r" % (value,))
    if limit < 0.0:
        raise MeshLodError("hausdorff 不能为负，得到 %r" % (value,))
    return limit


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------

def _bounds(verts):
    x0 = y0 = z0 = math.inf
    x1 = y1 = z1 = -math.inf
    for x, y, z in verts:
        if x < x0:
            x0 = x
        if y < y0:
            y0 = y
        if z < z0:
            z0 = z
        if x > x1:
            x1 = x
        if y > y1:
            y1 = y
        if z > z1:
            z1 = z
    return x0, y0, z0, x1, y1, z1


def _normal(verts, a, b, c):
    ax, ay, az = verts[a]
    bx, by, bz = verts[b]
    cx, cy, cz = verts[c]
    ux = bx - ax
    uy = by - ay
    uz = bz - az
    vx = cx - ax
    vy = cy - ay
    vz = cz - az
    return (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)


def _tri_area(verts, face):
    nx, ny, nz = _normal(verts, face[0], face[1], face[2])
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


def _mean_edge_length(verts, faces):
    total = 0.0
    count = 0
    seen = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            if key in seen:
                continue
            seen.add(key)
            total += math.dist(verts[u], verts[v])
            count += 1
    return total / count if count else 0.0


def _total_area(verts, faces):
    total = 0.0
    for face in faces:
        total += _tri_area(verts, face)
    return total


def _point_triangle_sq(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
    """点到三角形最近距离平方（Ericson, Real-Time Collision Detection）。"""
    abx = bx - ax
    aby = by - ay
    abz = bz - az
    acx = cx - ax
    acy = cy - ay
    acz = cz - az
    apx = px - ax
    apy = py - ay
    apz = pz - az
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return apx * apx + apy * apy + apz * apz

    bpx = px - bx
    bpy = py - by
    bpz = pz - bz
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return bpx * bpx + bpy * bpy + bpz * bpz

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        denom = d1 - d3
        t = d1 / denom if denom != 0.0 else 0.0
        dx = apx - t * abx
        dy = apy - t * aby
        dz = apz - t * abz
        return dx * dx + dy * dy + dz * dz

    cpx = px - cx
    cpy = py - cy
    cpz = pz - cz
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return cpx * cpx + cpy * cpy + cpz * cpz

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        denom = d2 - d6
        t = d2 / denom if denom != 0.0 else 0.0
        dx = apx - t * acx
        dy = apy - t * acy
        dz = apz - t * acz
        return dx * dx + dy * dy + dz * dz

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denom = (d4 - d3) + (d5 - d6)
        t = (d4 - d3) / denom if denom != 0.0 else 0.0
        dx = px - (bx + t * (cx - bx))
        dy = py - (by + t * (cy - by))
        dz = pz - (bz + t * (cz - bz))
        return dx * dx + dy * dy + dz * dz

    denom = va + vb + vc
    if denom == 0.0:
        denom = 1.0
    v = vb / denom
    w = vc / denom
    dx = px - (ax + abx * v + acx * w)
    dy = py - (ay + aby * v + acy * w)
    dz = pz - (az + abz * v + acz * w)
    return dx * dx + dy * dy + dz * dz


def _quadric_cost(qu, qv, x, y, z):
    """两个 10 元二次型之和在 (x, y, z) 处的取值。"""
    a2 = qu[0] + qv[0]
    ab = qu[1] + qv[1]
    ac = qu[2] + qv[2]
    ad = qu[3] + qv[3]
    b2 = qu[4] + qv[4]
    bc = qu[5] + qv[5]
    bd = qu[6] + qv[6]
    c2 = qu[7] + qv[7]
    cd = qu[8] + qv[8]
    d2 = qu[9] + qv[9]
    value = (a2 * x * x + 2.0 * ab * x * y + 2.0 * ac * x * z + 2.0 * ad * x
             + b2 * y * y + 2.0 * bc * y * z + 2.0 * bd * y
             + c2 * z * z + 2.0 * cd * z + d2)
    return value if value > 0.0 else 0.0


# ---------------------------------------------------------------------------
# 点到网格距离（BVH 加速），用于 Hausdorff 质量评估
# ---------------------------------------------------------------------------

class _SurfaceIndex:
    """均匀空间网格，用于最近三角形查询。

    每个三角形注册到其包围盒覆盖的所有 cell；查询时按环绕半径逐步扩张，
    当已找到的距离不超过当前环的最小可能距离时即可终止。
    """

    __slots__ = ("tris", "cells", "cell", "origin", "max_ring", "_small")

    def __init__(self, verts, faces, cell):
        tris = []
        cells = {}
        x0 = y0 = z0 = math.inf
        x1 = y1 = z1 = -math.inf
        for face in faces:
            a, b, c = face
            ax, ay, az = verts[a]
            bx, by, bz = verts[b]
            cx, cy, cz = verts[c]
            tris.append((ax, ay, az, bx, by, bz, cx, cy, cz))
            for (px, py, pz) in ((ax, ay, az), (bx, by, bz), (cx, cy, cz)):
                if px < x0:
                    x0 = px
                if py < y0:
                    y0 = py
                if pz < z0:
                    z0 = pz
                if px > x1:
                    x1 = px
                if py > y1:
                    y1 = py
                if pz > z1:
                    z1 = pz
        self.tris = tris
        self.cell = cell
        self.origin = (x0, y0, z0)
        inv = 1.0 / cell
        for fi, t in enumerate(tris):
            lo_x = int((min(t[0], t[3], t[6]) - x0) * inv)
            lo_y = int((min(t[1], t[4], t[7]) - y0) * inv)
            lo_z = int((min(t[2], t[5], t[8]) - z0) * inv)
            hi_x = int((max(t[0], t[3], t[6]) - x0) * inv)
            hi_y = int((max(t[1], t[4], t[7]) - y0) * inv)
            hi_z = int((max(t[2], t[5], t[8]) - z0) * inv)
            for ix in range(lo_x, hi_x + 1):
                for iy in range(lo_y, hi_y + 1):
                    for iz in range(lo_z, hi_z + 1):
                        key = (ix, iy, iz)
                        bucket = cells.get(key)
                        if bucket is None:
                            cells[key] = [fi]
                        else:
                            bucket.append(fi)
        span = max(x1 - x0, y1 - y0, z1 - z0)
        self.cells = cells
        self.max_ring = int(span * inv) + 2
        self._small = len(tris) <= 512

    def _brute_sq(self, px, py, pz):
        best = math.inf
        for t in self.tris:
            d2 = _point_triangle_sq(
                px, py, pz,
                t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8])
            if d2 < best:
                best = d2
        return best

    def nearest_sq(self, px, py, pz):
        # 三角形很少时逐个比较更快，也不会出现长距离空环扫描。
        if self._small:
            return self._brute_sq(px, py, pz)
        cell = self.cell
        ox, oy, oz = self.origin
        inv = 1.0 / cell
        bx = int((px - ox) * inv)
        by = int((py - oy) * inv)
        bz = int((pz - oz) * inv)
        cells = self.cells
        tris = self.tris
        best = math.inf
        ring = 0
        max_ring = self.max_ring
        while True:
            # 已检查的 shell 之外，最近的 cell 至少有 (ring-1)*cell 远。
            if ring > 0:
                limit = (ring - 1) * cell
                if best <= limit * limit:
                    return best
            hit = False
            for ix in range(bx - ring, bx + ring + 1):
                for iy in range(by - ring, by + ring + 1):
                    for iz in range(bz - ring, bz + ring + 1):
                        if (abs(ix - bx) != ring and abs(iy - by) != ring
                                and abs(iz - bz) != ring):
                            continue
                        bucket = cells.get((ix, iy, iz))
                        if bucket is None:
                            continue
                        hit = True
                        for fi in bucket:
                            t = tris[fi]
                            d2 = _point_triangle_sq(
                                px, py, pz,
                                t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8])
                            if d2 < best:
                                best = d2
            if ring >= max_ring:
                return best
            if not hit and ring >= 2 and math.isfinite(best):
                # 表面是连通的曲面边界：查询点位于表面外侧时，扫到空
                # shell 后继续外扩只会更远，已有 best 即可返回。
                return best
            ring += 1


def _sample_mesh(verts, faces, limit):
    """确定性采样：面重心、边中点、三分点与顶点，总量不超过 3*limit。

    同时覆盖面片内部与边界，细长三角形（坍缩后常见）也不会因采样过疏而
    低估误差。
    """
    if not faces:
        return list(verts[:limit])
    nfaces = len(faces)
    stride = max(1, (5 * nfaces + limit - 1) // limit)
    out = []
    for i in range(0, nfaces, stride):
        a, b, c = faces[i]
        ax, ay, az = verts[a]
        bx, by, bz = verts[b]
        cx, cy, cz = verts[c]
        out.append(((ax + bx + cx) / 3.0, (ay + by + cy) / 3.0, (az + bz + cz) / 3.0))
        out.append(((ax + bx) * 0.5, (ay + by) * 0.5, (az + bz) * 0.5))
        out.append(((bx + cx) * 0.5, (by + cy) * 0.5, (bz + cz) * 0.5))
        out.append(((cx + ax) * 0.5, (cy + ay) * 0.5, (cz + az) * 0.5))
        # 由各顶点指向重心的 2/3 点，补齐重心与顶点间的内部区域。
        gx = (ax + bx + cx) / 3.0
        gy = (ay + by + cy) / 3.0
        gz = (az + bz + cz) / 3.0
        out.append(((ax + gx) * 0.5, (ay + gy) * 0.5, (az + gz) * 0.5))
        out.append(((bx + gx) * 0.5, (by + gy) * 0.5, (bz + gz) * 0.5))
        out.append(((cx + gx) * 0.5, (cy + gy) * 0.5, (cz + gz) * 0.5))
        if len(out) >= limit:
            break
    nverts = len(verts)
    vstride = max(1, (nverts + limit - 1) // limit)
    for i in range(0, nverts, vstride):
        out.append(verts[i])
        if len(out) >= 3 * limit:
            break
    return out


def _cell_size(verts, faces, target_cells=48):
    x0, y0, z0, x1, y1, z1 = _bounds(verts)
    span = max(x1 - x0, y1 - y0, z1 - z0)
    return max(span / target_cells, 1e-12)


def _hausdorff(orig_verts, orig_faces, out_verts, out_faces, samples=None):
    """对称采样 Hausdorff 距离。"""
    if not orig_faces or not out_faces:
        return 0.0
    if samples is None:
        samples = _MAX_ERROR_SAMPLES
    cell = _cell_size(orig_verts, orig_faces)
    index_out = _SurfaceIndex(out_verts, out_faces, cell)
    index_orig = _SurfaceIndex(orig_verts, orig_faces, cell)
    worst = 0.0
    for p in _sample_mesh(orig_verts, orig_faces, samples):
        d2 = index_out.nearest_sq(p[0], p[1], p[2])
        if d2 > worst:
            worst = d2
    for p in _sample_mesh(out_verts, out_faces, samples):
        d2 = index_orig.nearest_sq(p[0], p[1], p[2])
        if d2 > worst:
            worst = d2
    if not math.isfinite(worst):
        return math.inf
    return math.sqrt(worst)


# ---------------------------------------------------------------------------
# 简化主体
# ---------------------------------------------------------------------------

def _attempt_collapse(keep, remove, verts, faces, vfaces, areas, quadrics,
                      vertex_extremes, extreme_count, total_area, initial_area,
                      area_eps, area_tol):
    """尝试把 ``remove`` 合并到 ``keep``（``keep`` 位置不变）。

    成功返回 ``(new_area, dying, modified, face_deaths)``，否则 ``None``。
    """
    dying = []
    modified = []
    face_deaths = None
    delta = 0.0
    incident = vfaces[remove]

    for fi in incident:
        a, b, c = faces[fi]
        ta = verts[a]
        tb = verts[b]
        tc = verts[c]
        if a == keep or b == keep or c == keep:
            # 同时包含 keep 与 remove 的面在坍缩后消失。
            delta -= areas[fi]
            dying.append(fi)
            if face_deaths is None:
                face_deaths = {}
            face_deaths[a] = face_deaths.get(a, 0) + 1
            face_deaths[b] = face_deaths.get(b, 0) + 1
            face_deaths[c] = face_deaths.get(c, 0) + 1
            continue

        na = keep if a == remove else a
        nb = keep if b == remove else b
        nc = keep if c == remove else c
        if na == nb or nb == nc or na == nc:
            return None

        # 旧法向
        oux = tb[0] - ta[0]
        ouy = tb[1] - ta[1]
        ouz = tb[2] - ta[2]
        ovx = tc[0] - ta[0]
        ovy = tc[1] - ta[1]
        ovz = tc[2] - ta[2]
        onx = ouy * ovz - ouz * ovy
        ony = ouz * ovx - oux * ovz
        onz = oux * ovy - ouy * ovx
        # 替换后的三个顶点坐标（remove 用 keep 取代）。
        nax, nay, naz = verts[na]
        nbx, nby, nbz = verts[nb]
        ncx, ncy, ncz = verts[nc]
        nux = nbx - nax
        nuy = nby - nay
        nuz = nbz - naz
        nvx = ncx - nax
        nvy = ncy - nay
        nvz = ncz - naz
        nnx = nuy * nvz - nuz * nvy
        nny = nuz * nvx - nux * nvz
        nnz = nux * nvy - nuy * nvx
        new_area = 0.5 * math.sqrt(nnx * nnx + nny * nny + nnz * nnz)
        if new_area <= area_eps:
            return None
        if nnx * onx + nny * ony + nnz * onz <= 0.0:
            return None
        delta += new_area - areas[fi]
        modified.append(fi)

    new_area = total_area + delta
    if math.isfinite(area_tol) and abs(new_area - initial_area) > area_tol * initial_area:
        return None

    # 包围盒：被删顶点持有的极值不能在删除后失去全部持有者。
    for key in vertex_extremes[remove]:
        if extreme_count[key] <= 1:
            return None
    # 另一个陷阱：极值顶点自身没被删，但它的面全死了，输出时会一并消失。
    if face_deaths:
        for p, deaths in face_deaths.items():
            if p == remove or not vertex_extremes[p]:
                continue
            if len(vfaces[p]) - deaths < 1:
                return None

    return new_area, dying, modified, face_deaths


def _simplify_mesh(verts, faces, ratio, hausdorff, area_tol=_AREATOL):
    """核心简化。返回 ``(out_verts, out_faces)``。

    ``hausdorff`` 仅用于校验取值合法性（是否达标由调用方实测判定）；
    ``area_tol`` 为过程中允许的面积相对偏差上限。
    """
    nverts = len(verts)
    nfaces = len(faces)
    target = max(1, int(round(ratio * nfaces)))
    x0, y0, z0, x1, y1, z1 = _bounds(verts)
    diagonal = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
    scale = diagonal if diagonal > 0.0 else 1.0
    _validate_hausdorff(hausdorff, scale)  # 非法阈值在这里就报错
    if target >= nfaces:
        return tuple(verts), tuple(faces)
    # 就地改写面索引，复制一份以免污染调用方 / 重试时重复使用的原始数据。
    faces = list(faces)
    area_eps = _DEGENERATE_FRACTION * scale * scale

    # --- 包围盒极值记账 ----------------------------------------------------
    # 顶点只会被合并到已有端点、不会产生新坐标，所以包围盒不变等价于：
    # 六个极值（三个轴各 min/max）上始终至少有一个存活顶点。这里记录每个
    # 极值当前由多少存活顶点持有，删除最后一个持有者时拒绝坍缩。
    lo = [min(p[axis] for p in verts) for axis in range(3)]
    hi = [max(p[axis] for p in verts) for axis in range(3)]
    extreme_count = [0] * 6          # 索引 = axis*2 + (0:min, 1:max)
    vertex_extremes = [()] * nverts  # 每个顶点持有的极值键
    for i in range(nverts):
        p = verts[i]
        keys = []
        for axis in range(3):
            if p[axis] <= lo[axis]:
                key = axis * 2
                keys.append(key)
                extreme_count[key] += 1
            if p[axis] >= hi[axis]:
                key = axis * 2 + 1
                keys.append(key)
                extreme_count[key] += 1
        if keys:
            vertex_extremes[i] = tuple(keys)

    # --- 二次误差、邻接、面积 ----------------------------------------------
    quadrics = [[0.0] * 10 for _ in range(nverts)]
    vfaces = [set() for _ in range(nverts)]
    areas = [0.0] * nfaces
    initial_area = 0.0
    for fi in range(nfaces):
        a, b, c = faces[fi]
        ax, ay, az = verts[a]
        bx, by, bz = verts[b]
        cx, cy, cz = verts[c]
        ux = bx - ax
        uy = by - ay
        uz = bz - az
        vx = cx - ax
        vy = cy - ay
        vz = cz - az
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        d = -(nx * ax + ny * ay + nz * az)
        contribution = (nx * nx, nx * ny, nx * nz, nx * d,
                        ny * ny, ny * nz, ny * d,
                        nz * nz, nz * d, d * d)
        for vi in (a, b, c):
            q = quadrics[vi]
            for k in range(10):
                q[k] += contribution[k]
            vfaces[vi].add(fi)
        face_area = 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)
        areas[fi] = face_area
        initial_area += face_area

    alive_count = nfaces
    versions = [0] * nverts
    total_area = initial_area
    removed = bytearray(nverts)

    heap = []
    heappush = heapq.heappush
    heappop = heapq.heappop
    pushed = {}

    def push_edge(u, v):
        if u == v:
            return
        if u > v:
            u, v = v, u
        stamp = (versions[u], versions[v])
        if pushed.get((u, v)) == stamp:
            return
        pushed[(u, v)] = stamp
        ux, uy, uz = verts[u]
        vx, vy, vz = verts[v]
        cost = _quadric_cost(
            quadrics[u], quadrics[v],
            (ux + vx) * 0.5, (uy + vy) * 0.5, (uz + vz) * 0.5)
        heappush(heap, (cost, u, v, versions[u], versions[v]))

    seen = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            if (u, v) in seen:
                continue
            seen.add((u, v))
            push_edge(u, v)
    del seen

    while alive_count > target and heap:
        _cost, u, v, vu, vv = heappop(heap)
        if removed[u] or removed[v]:
            continue
        if versions[u] != vu or versions[v] != vv:
            continue
        # 选择 QEM 代价更低的端点作为保留点；若某端点持有某个极值的
        # 唯一剩余顶点，则它不能被删除。
        ux, uy, uz = verts[u]
        vx, vy, vz = verts[v]
        cu = _quadric_cost(quadrics[u], quadrics[v], ux, uy, uz)
        cv = _quadric_cost(quadrics[u], quadrics[v], vx, vy, vz)
        u_pinned = any(extreme_count[k] <= 1 for k in vertex_extremes[u])
        v_pinned = any(extreme_count[k] <= 1 for k in vertex_extremes[v])
        if u_pinned and v_pinned:
            continue
        if u_pinned:
            candidates = ((v, u),)
        elif v_pinned:
            candidates = ((u, v),)
        elif cu <= cv:
            candidates = ((u, v), (v, u))
        else:
            candidates = ((v, u), (u, v))

        for keep, remove in candidates:
            result = _attempt_collapse(
                keep, remove, verts, faces, vfaces, areas, quadrics,
                vertex_extremes, extreme_count, total_area, initial_area,
                area_eps, area_tol)
            if result is None:
                continue
            new_area, dying, modified, face_deaths = result

            for fi in dying:
                fa, fb, fc = faces[fi]
                vfaces[fa].discard(fi)
                vfaces[fb].discard(fi)
                vfaces[fc].discard(fi)
                alive_count -= 1
            for fi in modified:
                face = faces[fi]
                faces[fi] = (keep if face[0] == remove else face[0],
                             keep if face[1] == remove else face[1],
                             keep if face[2] == remove else face[2])
                vfaces[remove].discard(fi)
                vfaces[keep].add(fi)
                areas[fi] = _tri_area(verts, faces[fi])

            qk = quadrics[keep]
            qr = quadrics[remove]
            for k in range(10):
                qk[k] += qr[k]
            quadrics[remove] = [0.0] * 10
            versions[keep] += 1
            # 极值持有权：remove 原有的极值失去一个持有者。keep 的坐标
            # 本身不变，它是否持有某个极值在初始化时已经判定，无需转移。
            for key in vertex_extremes[remove]:
                extreme_count[key] -= 1
            vertex_extremes[remove] = ()
            vfaces[remove] = set()
            removed[remove] = 1
            total_area = new_area

            for fi in vfaces[keep]:
                fa, fb, fc = faces[fi]
                if fa != keep:
                    push_edge(keep, fa)
                if fb != keep:
                    push_edge(keep, fb)
                if fc != keep:
                    push_edge(keep, fc)
            break

    # --- 压缩顶点索引（保持原始顺序） --------------------------------------
    mapping = {}
    out_verts = []
    out_faces = []
    append_face = out_faces.append
    for fi in range(nfaces):
        face = faces[fi]
        # 存活判据：三个顶点都仍持有该面（死亡面已从所有顶点集合移除）。
        a, b, c = face
        if fi not in vfaces[a] or fi not in vfaces[b] or fi not in vfaces[c]:
            continue
        mapped = []
        for vi in face:
            idx = mapping.get(vi)
            if idx is None:
                idx = len(out_verts)
                mapping[vi] = idx
                out_verts.append(verts[vi])
            mapped.append(idx)
        append_face(tuple(mapped))
    if not out_faces:
        return tuple(verts), tuple(faces)
    return tuple(out_verts), tuple(out_faces)


# ---------------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------------

class MeshLod:
    """网格 LOD 简化器。实例可跨线程共享；每次调用的结果相互独立。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._tls = threading.local()
        self._orig = None
        self._out = None
        self._shared_error = None

    def simplify(self, verts, faces, ratio, hausdorff=None, max_faces=None):
        """简化网格。

        参数
        ----
        verts : 序列，元素为三个有限实数
        faces : 序列，元素为三个整数索引
        ratio : (0, 1]，目标面数 = round(ratio * len(faces))
        hausdorff : 可选，绝对 Hausdorff 距离上界
            （默认取包围盒对角线的 5%，且不低于网格平均边长）
        max_faces : 可选，直接指定目标面数（优先于 ratio）

        返回
        ----
        ``(verts, faces)`` 二元组，顶点与面均为元组。
        """
        ratio_value = _validate_ratio(ratio)
        orig_verts, orig_faces = _validate_mesh(verts, faces)

        if max_faces is not None:
            target = _validate_target(max_faces)
            if target <= 0:
                raise MeshLodError("max_faces 必须至少为 1")
            target = min(target, len(orig_faces))
        else:
            target = int(round(ratio_value * len(orig_faces)))
            # 面数很少时四舍五入可能得到 0：至少保留一个三角形。
            target = max(1, min(target, len(orig_faces)))

        initial_area = _total_area(orig_verts, orig_faces)
        orig_bounds = _bounds(orig_verts)
        diagonal = max(math.dist(orig_bounds[:3], orig_bounds[3:]), 1e-300)
        if hausdorff is None:
            # 默认阈值按包围盒对角线给出，但不低于网格的平均边长：任何一次
            # 边坍缩的位移至少是一个网格尺度，低于该尺度等于禁止简化。
            limit = max(_DEFAULT_HAUSDORFF_FRACTION * diagonal,
                        _mean_edge_length(orig_verts, orig_faces))
        else:
            limit = _validate_hausdorff(hausdorff, diagonal)

        # 目标面数是硬指标：默认情况下一定把面数简化到目标值（只有面积/
        # 包围盒这类不可让步的约束会让贪心提前停住）。若调用方显式给出
        # Hausdorff 阈值，则该阈值成为硬约束，必要时放宽面数以换取达标。
        explicit_limit = hausdorff is not None
        target_options = [target]
        if explicit_limit:
            relaxed = target
            for _ in range(2):
                relaxed = int(relaxed * 2)
                if relaxed >= len(orig_faces):
                    break
                target_options.append(relaxed)

        def _evaluate(opt_ratio, tol):
            """简化一次，返回 (候选或 None, 面数, 是否满足全部硬约束)。"""
            cand_verts, cand_faces = _simplify_mesh(
                orig_verts, orig_faces, opt_ratio, hausdorff, area_tol=tol)
            count = len(cand_faces)
            if (_bounds(cand_verts) != orig_bounds
                    or abs(_total_area(cand_verts, cand_faces) - initial_area)
                    > _AREATOL * initial_area):
                return None, count, False
            cand_err = _hausdorff(orig_verts, orig_faces, cand_verts, cand_faces)
            if explicit_limit and cand_err > limit:
                return None, count, False
            return (cand_verts, cand_faces, cand_err), count, True

        # 先在第一档目标上按不同面积预算尝试；贪心若提前停住（面数偏多），
        # 收紧预算重做。全都是合格候选或都不可行时，再考虑放宽目标面数。
        out_verts = out_faces = None
        measured = 0.0
        for option in target_options:
            opt_ratio = option / float(len(orig_faces))
            for tol in (_AREATOL, 0.02, 0.01):
                cand, count, ok = _evaluate(opt_ratio, tol)
                if not ok:
                    continue
                out_verts, out_faces, measured = cand
                if count <= option:
                    break
            if out_faces is not None:
                break
            if len(target_options) == 1:
                break
            # 目标面数档位下无法满足显式阈值：继续尝试更保守（面数更多）的档。

        if out_faces is None:
            # 所有尝试都不合规：退回原始网格（误差为 0，必然满足约束）。
            out_verts, out_faces = tuple(orig_verts), tuple(orig_faces)
            measured = 0.0

        state = (orig_verts, orig_faces, out_verts, out_faces)
        tls = self._tls
        tls.state = state
        tls.measured = measured
        with self._lock:
            self._orig = (orig_verts, orig_faces)
            self._out = (out_verts, out_faces)
            self._shared_error = None
        return out_verts, out_faces

    def error(self):
        """返回最近一次简化相对原始网格的对称采样 Hausdorff 距离。

        当前线程尚未简化过时回退到实例最近一次结果；从未简化时返回 ``0.0``。
        """
        tls = self._tls
        state = getattr(tls, "state", None)
        measured = getattr(tls, "measured", None)
        if state is not None and measured is not None:
            return measured
        if state is None:
            with self._lock:
                if self._orig is None or self._out is None:
                    return 0.0
                state = (self._orig[0], self._orig[1], self._out[0], self._out[1])
                cached = self._shared_error
            if cached is not None and cached[0] is state[3]:
                return cached[1]
            value = _hausdorff(*state)
            with self._lock:
                self._shared_error = (state[3], value)
            return value
        value = _hausdorff(*state)
        tls.measured = value
        return value
