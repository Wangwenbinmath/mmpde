from fealpy.backend import backend_manager as bm
from fealpy.mesh import (TriangleMesh,
                        QuadrangleMesh, 
                        TetrahedronMesh, 
                        HexahedronMesh,
                        LagrangeTriangleMesh,
                        LagrangeQuadrangleMesh)
import matplotlib.pyplot as plt

class MeshGenerator():
    def __init__(self, nx, ny, nz = None ,box=[0,1,0,1], meshtype='tri', p = 1,delete_bound=None):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.box = box
        self.meshtype = meshtype
        self.p = p
        self.delete_bound = delete_bound
    def __call__(self, *args, **kwds):
        mesh = self.get_mesh()
        return mesh
    def thr(self,p):
        x = p[...,0]
        y=  p[...,1]
        area = self.delete_bound
        in_x = (x >= area[0]) & (x <= area[1])
        in_y = (y >= area[2]) & (y <= area[3])
        if p.shape[-1] == 3:
            z = p[...,2]
            in_z = (z >= area[4]) & (z <= area[5])
            return in_x & in_y & in_z
        return  in_x & in_y
    
    @classmethod
    def get_mesh(cls ,  nx, ny, nz = None ,box=[0,1,0,1], meshtype='tri',p = 1,delete_bound=None):
        instance = cls(nx, ny, nz, box, meshtype,p, delete_bound)
        if instance.delete_bound is  None:
            thr = None
        else:
            thr = instance.thr
        if instance.meshtype == 'tri':
            return TriangleMesh.from_box( instance.box,instance.nx, instance.ny,threshold=thr)
        elif instance.meshtype == 'quad':
            return QuadrangleMesh.from_box(instance.box, instance.nx, instance.ny, threshold=thr)
        elif instance.meshtype == 'tet':
            if instance.nz is None:
                raise ValueError("nz must be given")
            if len(instance.box) != 6:
                raise ValueError("box is not correct")
            return TetrahedronMesh.from_box(instance.box, instance.nx, instance.ny, instance.nz, threshold=thr)
        elif instance.meshtype == 'hex':
            if instance.nz is None:
                raise ValueError("nz must be given")
            if len(instance.box) != 6:
                raise ValueError("box is not correct")
            return HexahedronMesh.from_box(instance.box,instance.nx, instance.ny, instance.nz, threshold=thr)
        elif instance.meshtype == 'lagtri':
            tri = TriangleMesh.from_box(instance.box, instance.nx, instance.ny , threshold=thr)
            return LagrangeTriangleMesh.from_triangle_mesh(tri, p=instance.p)
        elif instance.meshtype == 'lagquad':
            quad = QuadrangleMesh.from_box(instance.box, instance.nx, instance.ny , threshold=thr)
            return LagrangeQuadrangleMesh.from_quadrangle_mesh(quad, p=instance.p)
        else:
            raise ValueError("meshtype must be tri, quad, tet or hex")
# 可视化
def high_order_meshploter(mesh, uh= None, model='mesh', scat_node=True , scat_index = slice(None)):
    nodes = mesh.node
    n = mesh.p
    def lagrange_interpolation(points, num_points=100, n = n):
        """
        @brief 利用拉格朗日插值构造曲线
        @param points: 插值点的列表 [(x0, y0), (x1, y1), ..., (xp, yp)]
        @param num_points: 曲线上点的数量
        @param n: 插值多项式的次数
        @return: 曲线上点的坐标数组
        """
        t = bm.linspace(0, n, num_points)

        def L(k, t):
            Lk = bm.ones_like(t)
            for i in range(n + 1):
                if i != k:
                    Lk *= (t - i) / (k - i)
            return Lk
        GD = points.shape[-1]
        curve = bm.zeros((t.shape[0], points.shape[0],GD), dtype=bm.float64)
        for k in range(n + 1):
            Lk = L(k, t)
            for i in range(GD):
                xk = points[:,k,i]
                bm.add_at(curve , (...,i) ,Lk[:,None] * xk)
        return curve
    fig = plt.figure()
    edges = mesh.edge
    if model == 'mesh':
        p = nodes[edges]
        curve = lagrange_interpolation(p,n=n)
        plt.plot(curve[...,0], curve[...,1], 'b-',linewidth=0.5)
        if scat_node:
            plt.scatter(nodes[scat_index, 0], 
                        nodes[scat_index, 1],s = 3, color='r')  # 绘制节点
        plt.gca().set_aspect('equal')
        plt.axis('off') # 关闭坐标轴
    elif model == 'surface':
        points = bm.concat([nodes, uh[...,None]], axis=-1)
        ax = fig.add_subplot(111, projection='3d')
        p = points[edges]
        curve = lagrange_interpolation(p,n=n).transpose(1,0,2)
        nan_separator = bm.full((curve.shape[0], 1, 3), bm.nan)  # 每条曲线之间插入 NaN
        concatenated = bm.concat([curve, nan_separator], axis=1)  # 插入分隔符
        curve = concatenated.reshape(-1, 3) 
        ax.plot(curve[..., 0], curve[..., 1], curve[..., 2], linewidth=1)
        if scat_node:
            ax.scatter(points[scat_index, 0], 
                       points[scat_index, 1], 
                       points[scat_index, 2],s = 2, color='r')  # 绘制节点
    plt.show()