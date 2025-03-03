from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.mesh import TriangleMesh,TetrahedronMesh,QuadrangleMesh
from fealpy.mesh import LagrangeTriangleMesh,LagrangeQuadrangleMesh
from fealpy.functionspace import LagrangeFESpace,ParametricLagrangeFESpace
from fealpy.fem import (BilinearForm 
                        ,ScalarDiffusionIntegrator
                        ,LinearForm
                        ,ScalarSourceIntegrator
                        ,ScalarConvectionIntegrator
                        ,DirichletBC)
from fealpy.solver import spsolve
from scipy.sparse.linalg import spsolve as spsolve1
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix,spdiags,block_diag,bmat
from sympy import *
from typing import Any ,Union,Optional
import pyamg

_U =  Union[TriangleMesh,
      TetrahedronMesh,
      QuadrangleMesh,
      LagrangeTriangleMesh,
      LagrangeQuadrangleMesh]

class Mesh_Data_Harmap():
    def __init__(self,mesh:_U,
                     Vertex) -> None:
        self.mesh = mesh
        self.node = mesh.entity('node')
        self.cell = mesh.entity('cell')
        self.kwargs0 = bm.context(self.node)
        self.kwargs1 = bm.context(self.cell)
        self.itype = self.cell.dtype
        self.ftype = self.node.dtype
        self.device = self.mesh.device
        self.TD = mesh.top_dimension()
        self.isinstance_mesh_type(mesh)
        
        self.Vertex = Vertex
        self.isconvex = self.is_convex()
        self.bdidx_constructor()
        self.globdata_constructor()
        
    def isinstance_mesh_type(self,mesh):
        """
        @brief 判断网格类型
        """
        if isinstance(mesh, TriangleMesh):
            self.mesh_type = "TriangleMesh"
            self.mesh_class = TriangleMesh
            self.g_type = "Simplexmesh"
            self.assambly_method = "fast"
            self.p = 1
        elif isinstance(mesh, TetrahedronMesh):
            self.mesh_type = "TetrahedronMesh"
            self.mesh_class = TetrahedronMesh
            self.g_type = "Simplexmesh"
            self.assambly_method = "fast"
            self.p = 1
        elif isinstance(mesh, LagrangeTriangleMesh):
            self.mesh_type = "LagrangeTriangleMesh"
            self.linermesh = mesh.linearmesh
            self.mesh_class = LagrangeTriangleMesh
            self.g_type = "Simplexmesh"
            self.assambly_method = "isopara"
            self.p = mesh.p
        elif isinstance(mesh, QuadrangleMesh):
            self.mesh_type = "QuadrangleMesh"
            self.mesh_class = QuadrangleMesh
            self.g_type = "Tensormesh"
            self.assambly_method = None
            self.pcell = self.cell[:,[0,3,1,2]]
            self.p = 1
        elif isinstance(mesh, LagrangeQuadrangleMesh):
            self.mesh_type = "LagrangeQuadrangleMesh"
            self.linermesh = mesh.linearmesh
            self.mesh_class = LagrangeQuadrangleMesh
            self.g_type = "Tensormesh"
            self.assambly_method = "isopara"
            self.p = mesh.p
        else:
            raise TypeError("Unsupported mesh type")
        if self.mesh_type != "QuadrangleMesh":
            self.pcell = self.cell
    
    def bdidx_constructor(self) -> TensorLike:
        """
        @brief 边界拓扑索引构造器
        """
        self.isBdNode = self.mesh.boundary_node_flag()
        self.BdNodeidx = self.mesh.boundary_node_index()
        self.BdFaceidx = self.mesh.boundary_face_index()
        self.BDNN = len(self.BdNodeidx)

        if (self.mesh_type == "LagrangeTriangleMesh" or 
            self.mesh_type == "LagrangeQuadrangleMesh"):
            p_mesh = self.linermesh
        else:
            p_mesh = self.mesh
        if self.isconvex:
            (self.Vertex_idx,
             self.Bdinnernode_idx,
             self.Arrisnode_idx) = self.get_various_bdidx(p_mesh)
        else:
            (self.Vertex_idx,
             self.Bdinnernode_idx,
             self.sort_BdNode_idx) = self.get_various_bdidx(p_mesh)
            "Align boundary points with vertices"
            if self.sort_BdNode_idx[0] != self.Vertex_idx[0]:
                K = bm.where(self.sort_BdNode_idx == self.Vertex_idx[0])[0][0]
                self.sort_BdNode_idx = bm.roll(self.sort_BdNode_idx,-K)
    
    def globdata_constructor(self) -> TensorLike:
        """
        @brief 全局信息构造器
        """
        self.NN = self.mesh.number_of_nodes()
        self.NLI = self.mesh.number_of_local_ipoints(self.p)
        self.NC = self.mesh.number_of_cells()
        if self.TD == 2:
            if (self.mesh_type == "LagrangeTriangleMesh" or 
                self.mesh_type == "LagrangeQuadrangleMesh"):
                self.node2face = self.mesh.node_to_face()
                self.space = ParametricLagrangeFESpace(self.mesh, p=self.p)
            else:
                self.node2face = self.mesh.node_to_face()
                self.space = LagrangeFESpace(self.mesh, p=self.p)
            if self.isconvex:
                self.Bi_Pnode_normal,self.b_val0 = self.get_normal_information(self.mesh)
                self.Bi_Lnode_normal = self.Bi_Pnode_normal
                self.logic_vertex = self.node[self.Vertex_idx]
            else:
                self.Bi_Pnode_normal = self.get_normal_information(self.mesh)
                self.Bi_Lnode_normal,self.b_val0,self.logic_vertex = self.get_logic_boundary()
                
        elif self.TD == 3:
            self.node2face = self.mesh.node_to_face()
            self.space = LagrangeFESpace(self.mesh, p=self.p)
            self.Bi_Lnode_normal,self.Ar_Lnode_normal,bcollection = self.get_normal_information(self.mesh)
            self.Bi_Pnode_normal = self.Bi_Lnode_normal
            self.logic_vertex = self.node[self.Vertex_idx]
            self.b_val0 = bcollection[0]
            self.b_val1 = bcollection[1]
            self.b_val2 = bcollection[2]

        self.cell2dof = self.space.cell_to_dof()
        # 目前暂时如此保存,避免重复计算
        shape = (self.NC,self.NLI,self.NLI)
        self.I = bm.broadcast_to(self.cell2dof[:, :, None], shape=shape)
        self.J = bm.broadcast_to(self.cell2dof[:, None, :], shape=shape)
        if self.g_type == "Tensormesh":
            ml = bm.multi_index_matrix(self.p,self.TD-1,dtype=self.ftype)/self.p
            self.multi_index = (ml,ml)
        else:
            self.multi_index = bm.multi_index_matrix(self.p,self.TD,dtype=self.ftype)/self.p

    def is_convex(self):
        """
        判断边界是否是凸的
        """
        from scipy.spatial import ConvexHull
        Vertex = self.Vertex
        hull = ConvexHull(Vertex)
        return len(Vertex) == len(hull.vertices)
    
    def sort_bdnode_and_bdface(self,mesh:_U) -> TensorLike:
        BdNodeidx = mesh.boundary_node_index()
        BdEdgeidx = mesh.boundary_face_index()
        node = mesh.node
        edge = mesh.edge
        
        node2edge = mesh.node_to_edge().toarray()
        bdnode2edge = node2edge[BdNodeidx][:,BdEdgeidx]
        i,j = bm.nonzero(bdnode2edge)
        bdnode2edge = j.reshape(-1,2)
        glob_bdnode2edge = bm.zeros_like(node,**self.kwargs1)
        glob_bdnode2edge = bm.set_at(glob_bdnode2edge,BdNodeidx,BdEdgeidx[bdnode2edge])
        
        sort_glob_bdedge_idx_list = []
        sort_glob_bdnode_idx_list = []

        start_bdnode_idx = BdNodeidx[0]
        sort_glob_bdnode_idx_list.append(start_bdnode_idx)
        current_node_idx = start_bdnode_idx
        
        for i in range(bdnode2edge.shape[0]):
            if edge[glob_bdnode2edge[current_node_idx,0],1] == current_node_idx:
                next_edge_idx = glob_bdnode2edge[current_node_idx,1]
            else:
                next_edge_idx = glob_bdnode2edge[current_node_idx,0]
            sort_glob_bdedge_idx_list.append(next_edge_idx)
            next_node_idx = edge[next_edge_idx,1]
            # 处理空洞区域
            if next_node_idx == start_bdnode_idx:
                if i < bdnode2edge.shape[0] - 1:
                    remian_bdnode_idx = list(set(BdNodeidx)-set(sort_glob_bdnode_idx_list))
                    start_bdnode_idx = remian_bdnode_idx[0]
                    next_node_idx = start_bdnode_idx
                else:
                # 闭环跳出循环
                    break
            sort_glob_bdnode_idx_list.append(next_node_idx)
            current_node_idx = next_node_idx
        sort_glob_bdnode_idx = bm.array(sort_glob_bdnode_idx_list,**self.kwargs1)
        sort_glob_bdedge_idx = bm.array(sort_glob_bdedge_idx_list,**self.kwargs1)
        if (self.mesh_type == "LagrangeTriangleMesh" or 
            self.mesh_type == "LagrangeQuadrangleMesh"):
            Ledge = self.mesh.edge
            sort_glob_bdedge = Ledge[sort_glob_bdedge_idx]
            sort_glob_bdnode_idx = sort_glob_bdedge[:,:-1].flatten()

        return sort_glob_bdnode_idx,sort_glob_bdedge_idx
    
    def get_node2face_norm(self,mesh:_U) -> None:
        BdNodeidx = mesh.boundary_node_index()
        BdFaceidx = mesh.boundary_face_index()
        TD = mesh.top_dimension()
        node2face = mesh.node_to_face()
        bd_node2face = node2face.toarray()[BdNodeidx][:,BdFaceidx]
        i , j = bm.nonzero(bd_node2face)
        bdfun = mesh.face_unit_normal(index=BdFaceidx[j])
        tolerance = 1e-8
        bdfun_rounded = bm.round(bdfun / tolerance) * tolerance
        normal,inverse = bm.unique(bdfun_rounded,return_inverse=True ,axis = 0)
        _,index,counts = bm.unique(i,return_index=True,return_counts=True)
        cow = bm.max(counts)
        r = bm.min(counts)

        inverse = bm.asarray(inverse,**self.kwargs1)
        node2face_normal = -bm.ones((BdNodeidx.shape[0],cow),**self.kwargs1)
        node2face_normal = bm.set_at(node2face_normal,(slice(None),slice(r)),
                                     inverse[index[:,None]+bm.arange(r ,**self.kwargs1)])
        for i in range(cow-r):
            isaimnode = counts > r+i
            node2face_normal = bm.set_at(node2face_normal,(isaimnode,r+i) ,
                                            inverse[index[isaimnode]+r+i])
        
        for i in range(node2face_normal.shape[0]):
            x = node2face_normal[i]
            unique_vals = bm.unique(x[x >= 0])
            result = -bm.ones(TD, **self.kwargs1)
            result = bm.set_at(result, slice(len(unique_vals)), unique_vals)
            node2face_normal = bm.set_at(node2face_normal, (i,slice(TD)) , result)

        return node2face_normal[:,:TD],normal
    
    def get_various_bdidx(self,mesh:_U) -> TensorLike:
        node2face_normal,normal = self.get_node2face_norm(mesh)
        BdNodeidx = mesh.boundary_node_index()
        Bdinnernode_idx = BdNodeidx[node2face_normal[:,1] < 0]
        if (self.mesh_type == "LagrangeTriangleMesh" or 
            self.mesh_type == "LagrangeQuadrangleMesh"):
            BdFaceidx = self.BdFaceidx
            LBdedge = self.mesh.edge[BdFaceidx]
            Bdinnernode_idx = bm.concatenate([Bdinnernode_idx,LBdedge[:,1:-1].flatten()])
        is_convex = self.isconvex
        Arrisnode_idx = None
        if is_convex:
            Vertex_idx = BdNodeidx[node2face_normal[:,-1] >= 0]
            if mesh.TD == 3:
                Arrisnode_idx = BdNodeidx[(node2face_normal[:,1] >= 0) & (node2face_normal[:,-1] < 0)]
            return Vertex_idx,Bdinnernode_idx,Arrisnode_idx
        else:
            if self.Vertex is None:
                raise ValueError('The boundary is not convex, you must give the Vertex')
            minus = mesh.node - self.Vertex[:,None]
            judge_vertex = bm.array(bm.sum(minus**2,axis=-1) < 1e-10,**self.kwargs1)
            K = bm.arange(mesh.number_of_nodes(),**self.kwargs1)
            Vertex_idx = bm.matmul(judge_vertex,K)
            sort_Bdnode_idx,sort_Bdface_idx = self.sort_bdnode_and_bdface(mesh)
            return Vertex_idx,Bdinnernode_idx,sort_Bdnode_idx
    
    def get_normal_information(self,mesh:_U) -> TensorLike:
        """
        @brief get_normal_information: 获取边界点法向量
        """
        Bdinnernode_idx = self.Bdinnernode_idx
        BdFaceidx = self.BdFaceidx
        node2face = self.node2face.toarray()
        if self.TD == 3:
            Arrisnode_idx = self.Arrisnode_idx
            Ar_node2face = node2face[Arrisnode_idx][:,BdFaceidx]
            i0 , j0 = bm.nonzero(Ar_node2face)
            bdfun0 = mesh.face_unit_normal(index=BdFaceidx[j0])
            normal0,inverse0 = bm.unique(bdfun0,return_inverse=True ,axis = 0)
            _,index0,counts0 = bm.unique(i0,return_index=True,return_counts=True)   
            maxcount = bm.max(counts0)
            mincount = bm.min(counts0)
            Ar_node2normal_idx = -bm.ones((len(Arrisnode_idx),maxcount),**self.kwargs1)
            Ar_node2normal_idx = bm.set_at(Ar_node2normal_idx,
                                            (slice(None),slice(mincount)),
                                            inverse0[index0[:,None]+bm.arange(mincount)])
            for i in range(maxcount-mincount):
                isaimnode = counts0 > mincount+i
                Ar_node2normal_idx = bm.set_at(Ar_node2normal_idx,(isaimnode,mincount+i) , 
                                                inverse0[index0[isaimnode]+mincount+i])
            for i in range(Ar_node2normal_idx.shape[0]):
                x = Ar_node2normal_idx[i]
                unique_vals = bm.unique(x[x >= 0])
                Ar_node2normal_idx = bm.set_at(Ar_node2normal_idx,(i,slice(len(unique_vals))),unique_vals)
            Ar_node2normal = normal0[Ar_node2normal_idx[:,:2]]  
            
        if (self.mesh_type == "LagrangeTriangleMesh" or 
            self.mesh_type == "LagrangeQuadrangleMesh"):
            LBdFace = mesh.face[BdFaceidx]
            LBd_node2face = bm.zeros((self.NN , 2),  **self.kwargs1)
            LBd_node2face = bm.set_at(LBd_node2face , (LBdFace[:,:-1],0) , BdFaceidx[:,None])
            LBd_node2face = bm.set_at(LBd_node2face , (LBdFace[:,1:],1) , BdFaceidx[:,None])
            LBdi_node2face = LBd_node2face[Bdinnernode_idx]
            linear_mesh = mesh.linearmesh
            Bi_node_normal  = linear_mesh.face_unit_normal(index=LBdi_node2face[:,0])
        else:
            Bi_node2face = node2face[Bdinnernode_idx][:,BdFaceidx]
            i1 , j1 = bm.nonzero(Bi_node2face)
            bdfun1 = mesh.face_unit_normal(index=BdFaceidx[j1])
            _,index1 = bm.unique(i1,return_index=True)
            Bi_node_normal = bdfun1[index1]
        
        if not self.isconvex:
            return Bi_node_normal
        else:
            Binode = self.node[Bdinnernode_idx]
            b_val0 = bm.sum(Binode * Bi_node_normal,axis=1)
            if self.TD == 2:
                return Bi_node_normal, b_val0
            else:
                Ar_node = self.node[Arrisnode_idx]
                b_val1 = bm.sum(Ar_node*Ar_node2normal[:,0,:],axis=1)
                b_val2 = bm.sum(Ar_node*Ar_node2normal[:,1,:],axis=1)
                return Bi_node_normal, Ar_node2normal, (b_val0, b_val1, b_val2)

    def get_logic_boundary(self,p = None) -> TensorLike:
        """
        @brief 获取逻辑区域以及法向信息
        """
        sBdNodeidx = self.sort_BdNode_idx
        Bdinnernode_idx = self.Bdinnernode_idx

        node = self.node
        Vertexidx = self.Vertex_idx
        physics_domain = node[Vertexidx]
        num_sides = physics_domain.shape[0]
        angles = bm.linspace(0,2*(1-1/(num_sides))*bm.pi,num_sides,**self.kwargs0)
        logic_domain = bm.stack([bm.cos(angles),bm.sin(angles)],axis=1)
        Lside_vector = bm.roll(logic_domain,-1,axis=0) - logic_domain

        Lside_vector_rotated = bm.stack([Lside_vector[:, 1], -Lside_vector[:, 0]], axis=1)
        Lside_length = bm.linalg.norm(Lside_vector_rotated, axis=1)
        Logic_unit_norm = Lside_vector_rotated / Lside_length[:, None]
        b_part = bm.sum(Logic_unit_norm * logic_domain,axis=1)
        
        K = bm.where(sBdNodeidx[:,None] == Vertexidx)[0]
        K = bm.concat([K,bm.array([len(sBdNodeidx)])])
        if p is None:
            Lun_repeat = bm.repeat(Logic_unit_norm,K[1:]-K[:-1],axis=0)
            bp_repeat = bm.repeat(b_part,K[1:]-K[:-1],axis=0)
            LBd_node2unorm = bm.zeros((self.NN , 2),  **self.kwargs0)
            b_val = bm.zeros(self.NN,  **self.kwargs0)
            bm.index_add(LBd_node2unorm , sBdNodeidx , Lun_repeat)
            bm.index_add(b_val , sBdNodeidx , bp_repeat)
            LBd_node2unorm = LBd_node2unorm[Bdinnernode_idx]
            b_val = b_val[Bdinnernode_idx]
            
            return LBd_node2unorm,b_val,logic_domain
        else:
            logic_bdnode = bm.zeros_like(node,**self.kwargs0)
            Pside_vector = bm.roll(physics_domain,-1,axis=0) - physics_domain
            Pside_length = bm.linalg.norm(Pside_vector,axis=1)
            rate = Lside_length / Pside_length
            theta = bm.arctan2(Lside_vector[:,1],Lside_vector[:,0]) -\
                    bm.arctan2(Pside_vector[:,1],Pside_vector[:,0])
            ctheta = bm.cos(theta)
            stheta = bm.sin(theta)
            R = bm.concat([ctheta,stheta,
                        -stheta,ctheta],axis=0).reshape(2,2,num_sides).T
            A = rate[:,None,None] * R
            A_repeat = bm.repeat(A,K[1:]-K[:-1],axis=0)
            PVertex_repeat = bm.repeat(physics_domain,K[1:]-K[:-1],axis=0)
            LVertex_repeat = bm.repeat(logic_domain,K[1:]-K[:-1],axis=0)
            Aim_vector = (A_repeat@((node[sBdNodeidx]-PVertex_repeat)[:,:,None])).reshape(-1,2)
            logic_bdnode = bm.set_at(logic_bdnode,sBdNodeidx,Aim_vector+LVertex_repeat)
            map = bm.where((node[:,None] == p).all(axis=2))[0]
            return logic_bdnode[map]
    
    def get_logic_node_init(self) -> TensorLike:
        """
        @brief 获取初始逻辑网格的节点坐标(作为初始猜测值)
        """
        bdc = self.get_logic_boundary
        p = self.p 
        space = self.space
        bform = BilinearForm(space)
        bform.add_integrator(ScalarDiffusionIntegrator(q=p+1,method=self.assambly_method))
        A = bform.assembly()
        lform = LinearForm(space)
        lform.add_integrator(ScalarSourceIntegrator(source=0,q=p+1,method=self.assambly_method))
        F = lform.assembly()
        bc0 = DirichletBC(space = space, gd = lambda p : bdc(p)[:,0])
        bc1 = DirichletBC(space = space, gd = lambda p : bdc(p)[:,1])
        uh0 = space.function()
        uh1 = space.function()
        A1, F1 = bc0.apply(A, F, uh0)
        A2, F2 = bc1.apply(A, F, uh1)
        uh0 = bm.set_at(uh0 , slice(None), spsolve(A1, F1 , solver="scipy"))
        uh1 = bm.set_at(uh1 , slice(None), spsolve(A2, F2 , solver="scipy"))
        logic_node = bm.concat([uh0[:,None],uh1[:,None]],axis=1)
        return logic_node


class Harmap_MMPDE(Mesh_Data_Harmap):
    def __init__(self, 
                 mesh:_U, 
                 uh:TensorLike,
                 beta :float ,
                 Vertex:TensorLike,
                 alpha = 0.5, 
                 mol_times = 3 ) -> None:
        """
        @param mesh: 初始物理网格
        @param uh: 物理网格上的解
        @param pde: 微分方程基本信息
        @param beta: 控制函数的参数
        @param alpha: 移动步长控制参数
        @param mol_times: 磨光次数
        @param redistribute: 是否预处理边界节点
        """
        super().__init__(mesh = mesh,
                         Vertex = Vertex)
        self.uh = uh
        self.beta = beta
        self.alpha = alpha
        self.mol_times = mol_times

        self.cm = mesh.entity_measure('cell')
        self.star_measure = self.get_star_measure()
        self.G,self.M = self.get_control_function()
        self.A , self.b = self.get_linear_constraint()
        self.get_logic_mesh()
        self.tol = self.caculate_tol()
        self.clear_unused_attributes()
        
    def get_logic_mesh(self):
        """
        @brief 获取逻辑网格
        """
        if not self.is_convex():
            if self.TD == 3:
                raise ValueError('Non-convex polyhedra cannot construct a logical mesh')
            self.logic_node,_ = self.solve_move_LogicNode(bm.ones(self.NC,**self.kwargs0)[:,None],isinit=True)
            if self.mesh_type == "LagrangeTriangleMesh":
                self.logic_mesh = self.mesh_class.from_triangle_mesh(self.linermesh, self.p)
                self.logic_mesh.node = self.logic_node
            elif self.mesh_type == "LagrangeQuadrangleMesh":
                self.logic_mesh = self.mesh_class.from_quadrangle_mesh(self.linermesh, self.p)
                self.logic_mesh.node = self.logic_node
            else:
                logic_cell = self.cell
                self.logic_mesh = self.mesh_class(self.logic_node,logic_cell)  
        else:
            if self.mesh_type == "LagrangeTriangleMesh":
                self.logic_mesh = self.mesh_class.from_triangle_mesh(self.linermesh, self.p)
            elif self.mesh_type == "LagrangeQuadrangleMesh":
                self.logic_mesh = self.mesh_class.from_quadrangle_mesh(self.linermesh, self.p)
            else:
                self.logic_mesh = self.mesh_class(bm.copy(self.node),self.cell)
            self.logic_node = self.logic_mesh.node

    def clear_unused_attributes(self):
        """
        @brief 清理不再需要的属性以释放内存
        """
        attributes_to_clear = [
            'node2face',
            'sort_BdNode_idx',
            'logic_mesh',
            'Vertex_idx',
            'logic_vertex',
            'b_val0','b_val1','b_val2',  
        ]
        for attr in attributes_to_clear:
            if hasattr(self, attr):
                delattr(self, attr)

    def get_star_measure(self)->TensorLike:
        """
        @brief 计算每个节点的星的测度
        """
        NN = self.NN
        star_measure = bm.zeros(NN,**self.kwargs0)
        bm.index_add(star_measure , self.cell , self.cm[:,None])
        return star_measure
    
    def get_control_function(self)->TensorLike:
        """
        @brief 计算控制函数
        """
        cell = self.cell
        cm = self.cm
        multi_index = self.multi_index
        space = self.space
        gphi = space.grad_basis(multi_index)
        pcell = self.pcell
        guh_incell = bm.einsum('cqid , ci -> cqd ',gphi,self.uh[pcell]) # (NC,ldof,GD)
        max_norm_guh = bm.max(bm.linalg.norm(guh_incell,axis=-1))
        M = bm.sqrt(1 + self.beta * bm.sum(guh_incell**2,axis=-1)/max_norm_guh**2) # (NC,ldof)
        M_incell = bm.mean(M,axis=-1)
        for k in range(self.mol_times):
            M = bm.zeros(self.NN,**self.kwargs0)
            bm.index_add(M , cell, (cm *M_incell)[: , None])
            M /= self.star_measure
            M_incell = bm.mean(M[cell],axis=-1)
        qf = self.mesh.quadrature_formula(self.p+2, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        M_incell = space.value(M,bcs)
        return 1/M_incell,M
    
    def get_control_function_new(self)->TensorLike:
        """
        @brief 计算控制函数
        """
        cell = self.cell
        cm = self.cm
        multi_index = self.multi_index
        space = self.space
        gphi = space.grad_basis(multi_index)
        pcell = self.pcell
        guh_incell = bm.einsum('cqid , ci -> cqd ',gphi,self.uh[pcell]) # (NC,ldof,GD)
        max_norm_guh = bm.max(bm.linalg.norm(guh_incell,axis=-1))
        M = bm.sqrt(1 + self.beta * bm.sum(guh_incell**2,axis=-1)/max_norm_guh**2) # (NC,ldof)
        
        p = 1
        cell = self.cell
        if self.mesh_type == "LagrangeTriangleMesh" :
            cell = bm.concat([cell[:,0:3],cell[:,[1,3,4]],
                                  cell[:,[1,4,2]],cell[:,[2,4,5]]],axis = 0)
            M = bm.concat([M[:,0:3],M[:,[1,3,4]],
                                  M[:,[1,4,2]],M[:,[2,4,5]]],axis = 0)
            linear_split_mesh = TriangleMesh(self.node,cell)
            multi_index = bm.multi_index_matrix(p,2,dtype=self.ftype)/p
        elif self.mesh_type == "LagrangeQuadrangleMesh":
            cell = bm.concat([cell[:,[0,3,4,1]],cell[:,[1,4,5,2]],
                                    cell[:,[3,6,7,4]],cell[:,[4,7,8,5]]],axis = 0)
            M = bm.concat([M[:,[0,3,4,1]],M[:,[1,4,5,2]],
                                    M[:,[3,6,7,4]],M[:,[4,7,8,5]]],axis = 0)
            linear_split_mesh = QuadrangleMesh(self.node,cell)
            ml = bm.multi_index_matrix(p,self.TD-1,dtype=self.ftype)/p
            multi_index = (ml,ml)
        cm = linear_split_mesh.entity_measure('cell')
        
        M_incell = bm.mean(M,axis=-1)
        NN = self.NN
        star_measure = bm.zeros(NN,**self.kwargs0)
        bm.index_add(star_measure , cell , cm[:,None])

        for k in range(self.mol_times):
            M = bm.zeros(self.NN,**self.kwargs0)
            bm.index_add(M , cell, (cm *M_incell)[: , None])
            M /= star_measure
            M_incell = bm.mean(M[cell],axis=-1)

        qf = self.mesh.quadrature_formula(self.p+1)
        bcs = qf.get_quadrature_points_and_weights()[0]
        phi = self.space.basis(bcs)
        M_incell = bm.einsum('cqi , ci  -> cq ', phi , M[self.cell])
        return 1/M_incell,M

    def get_stiff_matrix(self,
                         mesh:_U,
                         G:TensorLike):
        """
        @brief 组装刚度矩阵
        @param mesh: 物理网格
        @param G: 控制函数
        """
        cm = mesh.entity_measure('cell')
        qf = mesh.quadrature_formula(self.p+2)
        bcs, ws = qf.get_quadrature_points_and_weights()
        space = self.space
        gphi = space.grad_basis(bcs)
        GDOF = space.number_of_global_dofs()
        if (self.mesh_type == "LagrangeTriangleMesh" or 
            self.mesh_type == "LagrangeQuadrangleMesh"):
            rm = mesh.reference_cell_measure()
            J = mesh.jacobi_matrix(bcs)
            cm = rm * bm.linalg.det(J)
        H = bm.einsum('q , cqid , cq ,cqjd, c... -> cij ',ws, gphi ,G , gphi,cm)
        I,J = self.I,self.J
        H = csr_matrix((H.flatten(), (I.flatten(), J.flatten())), shape=(GDOF, GDOF))
        return H
    
    def get_linear_constraint(self):
        """
        @brief 组装线性约束
        """
        logic_vertex = self.logic_vertex
        BdNodeidx = self.BdNodeidx
        Vertex_idx = self.Vertex_idx
        Bdinnernode_idx = self.Bdinnernode_idx
        Binnorm = self.Bi_Lnode_normal

        NN = self.NN
        BDNN = self.BDNN
        VNN = len(Vertex_idx)

        b = bm.zeros(NN, **self.kwargs0)
        b_val0 = self.b_val0
        b = bm.set_at(b , Bdinnernode_idx , b_val0)
        A_diag = bm.zeros((self.TD , NN)  , **self.kwargs0)
        A_diag = bm.set_at(A_diag , (...,Bdinnernode_idx) , Binnorm.T)
        A_diag = bm.set_at(A_diag , (0,Vertex_idx) , 1)
        if self.TD == 2:
            b = bm.set_at(b , Vertex_idx , logic_vertex[:,0])
            b = bm.concatenate([b[BdNodeidx],logic_vertex[:,1]])
            A_diag = A_diag[:,BdNodeidx]

            A = bmat([[spdiags(A_diag[0], 0, BDNN, BDNN, format='csr'),
                       spdiags(A_diag[1], 0, BDNN, BDNN, format='csr')],
                      [csr_matrix((VNN, BDNN)),
                       csr_matrix((bm.ones(VNN),
                                  (bm.arange(VNN), Vertex_idx)), 
                        shape=(VNN, NN))[:, BdNodeidx]]], format='csr')

        elif self.TD == 3:
            Arrisnode_idx = self.Arrisnode_idx
            Arnnorm = self.Ar_Lnode_normal
            ArNN = len(Arrisnode_idx)
            b_val1 = self.b_val1
            b_val2 = self.b_val2
            b = bm.set_at(b , Arrisnode_idx , b_val1)
            b = bm.set_at(b , Vertex_idx , logic_vertex[:,0])[BdNodeidx]
            b = bm.concatenate([b,b_val2,logic_vertex[:,1],logic_vertex[:,2]])       

            A_diag = bm.set_at(A_diag , (...,Arrisnode_idx) , Arnnorm[:,0,:].T)
            A_diag = A_diag[:,BdNodeidx]
            
            index1 = NN * bm.arange(self.TD) + Arrisnode_idx[:,None]
            index2 = NN * bm.arange(self.TD) + BdNodeidx[:,None]
            rol_Ar = bm.repeat(bm.arange(ArNN)[None,:],3,axis=0).flatten()
            cow_Ar = index1.T.flatten()
            data_Ar = Arnnorm[:,1,:].T.flatten()
            Ar_constraint = csr_matrix((data_Ar,(rol_Ar, cow_Ar)),shape=(ArNN,3*NN))
            Vertex_constraint1 = csr_matrix((bm.ones(VNN,dtype=bm.float64),
                                (bm.arange(VNN),Vertex_idx + NN)),shape=(VNN,3*NN))
            Vertex_constraint2 = csr_matrix((bm.ones(VNN,dtype=bm.float64),
                                (bm.arange(VNN),Vertex_idx + 2 * NN)),shape=(VNN,3*NN))

            A_part = bmat([[spdiags(A_diag[0], 0, BDNN, BDNN, format='csr'),
                            spdiags(A_diag[1], 0, BDNN, BDNN, format='csr'),
                            spdiags(A_diag[2], 0, BDNN, BDNN, format='csr')]], format='csr')
            A = bmat([[A_part],
                      [Ar_constraint[:, index2.T.flatten()]],
                      [Vertex_constraint1[:, index2.T.flatten()]],
                      [Vertex_constraint2[:, index2.T.flatten()]]], format='csr')
        return A,b

    def solve_move_LogicNode(self,G, isinit = False):
        """
        @brief 交替求解逻辑网格点
        @param process_logic_node: 新逻辑网格点
        @param move_vector_field: 逻辑网格点移动向量场
        """
        isBdNode = self.isBdNode
        TD = self.TD
        BDNN = self.BDNN
        INN = self.NN - BDNN
        H = self.get_stiff_matrix(self.mesh,G)
        H11 = H[~isBdNode][:, ~isBdNode]
        H12 = H[~isBdNode][:, isBdNode]
        H21 = H[isBdNode][:, ~isBdNode]
        H22 = H[isBdNode][:, isBdNode]
        A,b= self.A,self.b
        if isinit:
            logic_node = self.get_logic_node_init()
            process_logic_node = bm.copy(logic_node)
        else:
            logic_node = self.logic_node
            process_logic_node = bm.copy(self.logic_node)
        # 移动逻辑网格点
        F = (-H12 @ process_logic_node[isBdNode, :]).T.flatten()
        H0 = block_diag([H11]*TD,format='csr')
        ml = pyamg.ruge_stuben_solver(H0)
        move_innerlogic_node = bm.asarray(ml.solve(F, tol=1e-7),**self.kwargs0)
        move_innerlogic_node = move_innerlogic_node.reshape((TD, INN)).T
        process_logic_node = bm.set_at(process_logic_node, ~isBdNode, move_innerlogic_node)

        F = (-H21 @ move_innerlogic_node).T.flatten()
        F = bm.asarray(F, **self.kwargs0)
        b0 = bm.concatenate((F,b),axis=0)

        A1 = block_diag([H22]*TD,format='csr')
        A0 = bmat([[A1,A.T],[A,None]],format='csr')

        move_bdlogic_node = bm.asarray(spsolve1(A0,b0)[:TD*BDNN],**self.kwargs0)
        move_bdlogic_node = move_bdlogic_node.reshape((TD, BDNN)).T
        process_logic_node = bm.set_at(process_logic_node , isBdNode, move_bdlogic_node)
        move_vector_field = logic_node - process_logic_node

        return process_logic_node,move_vector_field

    def get_physical_node(self,move_vertor_field,logic_node_move):
        """
        @brief 计算物理网格点
        @param move_vertor_field: 逻辑网格点移动向量场
        @param logic_node_move: 移动后的逻辑网格点
        """
        node = self.node
        cell = self.cell
        cm = self.cm
        TD = self.TD
        p = self.p
        space = self.space
        multi_index = self.multi_index 
        gphi = space.grad_basis(multi_index)
        grad_x = bm.zeros((self.NN,TD,TD),**self.kwargs0)
        pcell = self.pcell
        grad_X_incell = bm.einsum('cin, cqim -> cqnm',logic_node_move[pcell], gphi)
        grad_x_incell = bm.linalg.inv(grad_X_incell)*cm[:,None,None,None]
        bm.index_add(grad_x , pcell , grad_x_incell)
        grad_x /= self.star_measure[:,None,None]
        delta_x = (grad_x @ move_vertor_field[:,:,None]).reshape(-1,TD)

        if TD == 3:
            self.Bi_Pnode_normal = self.Bi_Lnode_normal
            Ar_Pnode_normal = self.Ar_Lnode_normal
            Arrisnode_idx = self.Arrisnode_idx
            dot1 = bm.sum(Ar_Pnode_normal * delta_x[Arrisnode_idx,None],axis=-1)
            doap = dot1[:,0,None] * Ar_Pnode_normal[:,0,:] + dot1[:,1,None] * Ar_Pnode_normal[:,1,:]
            delta_x = bm.set_at(delta_x , Arrisnode_idx , delta_x[Arrisnode_idx] - doap)
        
        Bdinnernode_idx = self.Bdinnernode_idx
        dot = bm.sum(self.Bi_Pnode_normal * delta_x[Bdinnernode_idx],axis=1)
        delta_x = bm.set_at(delta_x , Bdinnernode_idx ,
                            delta_x[Bdinnernode_idx] - dot[:,None] * self.Bi_Pnode_normal)

        if self.mesh_type == "TriangleMesh" or self.mesh_type == "TetrahedronMesh":
            A = bm.swapaxes((node[cell[:,1:]] - node[cell[:,0,None]]),axis1 = 1,axis2 = 2)
            C = bm.swapaxes((delta_x[cell[:,1:]] - delta_x[cell[:,0,None]]),axis1 = 1,axis2 = 2)
        elif self.mesh_type == "QuadrangleMesh":
            smatrix = bm.array([[0,1,3],[1,2,0],[2,3,1],[3,0,2]],**self.kwargs1)
            scell = cell[:,smatrix].reshape(-1,3)
            A = bm.swapaxes((node[scell[:,1:]] - node[scell[:,0,None]]),axis1 = 1,axis2 = 2)
            C = bm.swapaxes((delta_x[scell[:,1:]] - delta_x[scell[:,0,None]]),axis1 = 1,axis2 = 2)

        # 物理网格点移动距离
        if TD == 2:
            if (self.mesh_type == "LagrangeTriangleMesh" or 
                self.mesh_type == "LagrangeQuadrangleMesh"):
                qf = self.mesh.quadrature_formula(p+2)
                bc,ws = qf.get_quadrature_points_and_weights()
                J = self.mesh.jacobi_matrix(bc=bc)
                gphi = self.mesh.grad_shape_function(bc,p = p, variables='u')
                dJ = bm.einsum('...in, ...qim -> ...qnm',delta_x[pcell],gphi)
                a = bm.linalg.det(dJ)@ws
                c = bm.linalg.det(J)@ws
                b = (J[...,0,0]*dJ[...,1,1] + J[...,1,1]*dJ[...,0,0] \
                - J[...,0,1]*dJ[...,1,0] - J[...,1,0]*dJ[...,0,1])@ws             
            else:
                a = bm.linalg.det(C)
                c = bm.linalg.det(A)
                b = A[:,0,0]*C[:,1,1] - A[:,0,1]*C[:,1,0] + C[:,0,0]*A[:,1,1] - C[:,0,1]*A[:,1,0]
            discriminant = b**2 - 4*a*c
            right_idx = bm.where(discriminant >= 0)[0]
            x = bm.concatenate([(-b[right_idx] + bm.sqrt(discriminant[right_idx]))/(2*a[right_idx]),
                                (-b[right_idx] - bm.sqrt(discriminant[right_idx]))/(2*a[right_idx])])
        else:
            # 三维情况，求解三次方程
            a0,a1,a2 = C[:,1,1]*C[:,2,2] - C[:,1,2]*C[:,2,1],\
                    C[:,1,2]*C[:,2,0] - C[:,1,0]*C[:,2,2],\
                    C[:,1,0]*C[:,2,1] - C[:,1,1]*C[:,2,0]
            b0,b1,b2 = A[:,1,1]*C[:,2,2] - A[:,1,2]*C[:,2,1] + C[:,1,1]*A[:,2,2] - C[:,1,2]*A[:,2,1],\
                    A[:,1,0]*C[:,2,2] - A[:,1,2]*C[:,2,0] + C[:,1,0]*A[:,2,2] - C[:,1,2]*A[:,2,0],\
                    A[:,1,0]*C[:,2,1] - A[:,1,1]*C[:,2,0] + C[:,1,0]*A[:,2,1] - C[:,1,1]*A[:,2,0]
            c0,c1,c2 = A[:,1,1]*A[:,2,2] - A[:,1,2]*A[:,2,1],\
                    A[:,1,0]*A[:,2,2] - A[:,1,2]*A[:,2,0],\
                    A[:,1,0]*A[:,2,1] - A[:,1,1]*A[:,2,0]
            a = C[:,0,0]*a0 - C[:,0,1]*a1 + C[:,0,2]*a2
            ridx = bm.where(a> 1e-14)[0]
            b = A[:,0,0]*a0 - A[:,0,1]*a1 + A[:,0,2]*a2 + C[:,0,0]*b0 - C[:,0,1]*b1 + C[:,0,2]*b2
            c = A[:,0,0]*b0 - A[:,0,1]*b1 + A[:,0,2]*b2 + C[:,0,0]*c0 - C[:,0,1]*c1 + C[:,0,2]*c2
            d = A[:,0,0]*c0 - A[:,0,1]*c1 + A[:,0,2]*c2
            a,b,c,d = a[ridx],b[ridx],c[ridx],d[ridx]
            # 使用卡尔达诺公式求解三次方程的实根
            p = (3 * a * c - b**2) / (3 * a**2)
            q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)
            discriminant = (q / 2)**2 + (p / 3)**3
            ridx = bm.where(discriminant >= 0)[0]
            nidx = bm.where(discriminant < 0)[0]
            sqdr = bm.sqrt(discriminant[ridx])
            sqdn = bm.sqrt(-discriminant[nidx])
            u0 = (-q[ridx] / 2 + sqdr)**(1 / 3)
            u1 = (-q[nidx] / 2 + sqdn * 1j)**(1 / 3)
            v0 = (-q[ridx] / 2 - sqdr)**(1 / 3)
            v1 = (-q[nidx] / 2 - sqdn * 1j)**(1 / 3)

            upv = bm.zeros_like(a , **self.kwargs0)
            umv = bm.zeros_like(a , **self.kwargs0)
            upv = bm.set_at(upv , ridx , (u0 + v0).real)
            upv = bm.set_at(upv , nidx , (u1 + v1).real)
            umv = bm.set_at(umv , ridx , (u0 - v0).real)
            umv = bm.set_at(umv , nidx , (u1 - v1).real)

            move_dis = b / (3 * a)
            x1 = upv - move_dis
            sq3 = bm.sqrt(bm.array([3],**self.kwargs0))
            x2 = -upv / 2 + umv * sq3 * 1j / 2 - move_dis
            x3 = -upv / 2 - umv * sq3 * 1j / 2 - move_dis
            x = bm.concatenate([x1, x2, x3])
        positive_x = bm.where(x.real>0, x.real, 1)
        eta = bm.min(positive_x)
        node = node +  self.alpha*eta* delta_x
        return node

    def interpolate(self,move_node):
        """
        @brief 将解插值到新网格上
        @param move_node: 移动后的物理节点
        """
        delta_x = self.node - move_node
        cell = self.cell
        mesh = self.mesh
        space = self.space
        qf = mesh.quadrature_formula(self.p+2,'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        if self.mesh_type == "LagrangeTriangleMesh":
            # cell2dof = bm.concat([cell[:,0:3],cell[:,[1,3,4]],
            #                       cell[:,[1,4,2]],cell[:,[2,4,5]]],axis = 0)
            # # cell2dof = cell
            # GDOF = self.NN
            # mesh0 = TriangleMesh(self.node,cell2dof)
            # space1 = LagrangeFESpace(mesh0, p=1)
            # phi = space1.basis(bc = bcs)
            # gphi = space1.grad_basis(bc = bcs)
            # cm = mesh0.entity_measure('cell')
            # M = bm.einsum('q , cqi ,cqj, c -> cij ',ws, phi ,phi ,cm)  
            # P = bm.einsum('q , cqid , cid ,cqj ,c -> cij' , ws , gphi, delta_x[cell2dof], phi, cm)
            cell2dof = self.cell2dof
            GDOF = space.number_of_global_dofs()
            phi = space.basis(bcs)
            gphi = space.grad_basis(bcs)
            rm = mesh.reference_cell_measure()
            J = mesh.jacobi_matrix(bcs)
            cm = rm * bm.linalg.det(J)
            M = bm.einsum('q , cqi ,cqj, cq -> cij ',ws, phi ,phi ,cm)
            P = bm.einsum('q , cqid , cid ,cqj ,cq -> cij' , ws , gphi, delta_x[cell2dof], phi, cm)
        elif self.mesh_type == "LagrangeQuadrangleMesh":
            cell2dof = self.cell2dof
            GDOF = space.number_of_global_dofs()
            phi = space.basis(bcs)
            gphi = space.grad_basis(bcs)
            rm = mesh.reference_cell_measure()
            J = mesh.jacobi_matrix(bcs)
            cm = rm * bm.linalg.det(J)
            M = bm.einsum('q , cqi ,cqj, cq -> cij ',ws, phi ,phi ,cm)
            P = bm.einsum('q , cqid , cid ,cqj ,cq -> cij' , ws , gphi, delta_x[cell2dof], phi, cm)
        else:
            cell2dof = self.cell2dof
            GDOF = space.number_of_global_dofs()
            phi = space.basis(bcs)
            gphi = space.grad_basis(bcs)
            cm = self.cm
            M = bm.einsum('q , cqi ,cqj, c -> cij ',ws, phi ,phi ,cm)  
            P = bm.einsum('q , cqid , cid ,cqj ,c -> cij' , ws , gphi, delta_x[cell2dof], phi, cm)

        I,J = self.I,self.J
        M = csr_matrix((M.flatten(), (I.flatten(), J.flatten())), shape=(GDOF, GDOF))
        P = csr_matrix((P.flatten(), (I.flatten(), J.flatten())), shape=(GDOF, GDOF))
        # ml = pyamg.ruge_stuben_solver(M)
        def ODEs(t,y):
            f = spsolve1(M, P @ y)
            return f
        # 初值条件  
        uh0 = self.uh
        # 范围
        tau_span = [0,1]
        # 求解
        sol = solve_ivp(ODEs,tau_span,uh0,method='RK23').y[:,-1]
        sol = bm.asarray(sol,**self.kwargs0)
        return sol
    
    def interpolate_1(self,move_node):
        """
        @brief 一个用于测试插值效能的函数和interpolate函数类似
        """
        delta_x = self.node - move_node
        cell = self.cell
        mesh = self.mesh
        space = self.space
        qf = mesh.quadrature_formula(self.p+1,'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        cell2dof = self.cell2dof
        GDOF = space.number_of_global_dofs()
        if self.mesh_type == "LagrangeTriangleMesh":
            cell2dof = bm.concat([cell[:,0:3],cell[:,[1,3,4]],
                                  cell[:,[1,4,2]],cell[:,[2,4,5]]],axis = 0)
            # cell2dof = cell
            GDOF = self.NN
            mesh = TriangleMesh(self.node,cell2dof)
            space = LagrangeFESpace(mesh, p=1)
        I,J = self.I,self.J
        def ODEs(t,y):
            new_node = self.node - t * delta_x
            mesh.node = new_node
            space.mesh = mesh
            phi = space.basis(bcs)
            gphi = space.grad_basis(bcs)
            cm = mesh.entity_measure('cell')
            M = bm.einsum('q , cqi ,cqj, c -> cij ',ws, phi ,phi ,cm)  
            P = bm.einsum('q , cqid , cid ,cqj ,c -> cij' , ws , gphi, delta_x[cell2dof], phi, cm)
            M = csr_matrix((M.flatten(), (I.flatten(), J.flatten())), shape=(GDOF, GDOF))
            P = csr_matrix((P.flatten(), (I.flatten(), J.flatten())), shape=(GDOF, GDOF))
            f = spsolve1(M, P @ y)
            return f
        # 初值条件
        uh0 = self.uh
        # 范围
        tau_span = [0,1]
        # 求解
        sol = solve_ivp(ODEs,tau_span,uh0,method='RK23').y[:,-1]
        return sol
    
    def handle_method(self, new_node, pde, method):
        """
        @brief handle_method: 处理各种插值方式
        """
        if method == 'interpolate':
            return self.interpolate(new_node)
        elif method == 'interpolate_1':
            return self.interpolate_1(new_node)
        elif method == 'solution':
            if pde is None:
                raise ValueError("pde must be provided for method 'solution'")
            return pde.solution(new_node)
        elif method == 'poisson':
            self.mesh.node = new_node
            self.space.mesh = self.mesh
            am = self.assambly_method
            if pde is None:
                raise ValueError("pde must be provided for method 'poisson'")
            bform = BilinearForm(self.space)
            lform = LinearForm(self.space)
            SDI = ScalarDiffusionIntegrator(q=self.p+2, method=am)
            SSI = ScalarSourceIntegrator(source=pde.source, method=am)
            bform.add_integrator(SDI)
            lform.add_integrator(SSI)
            A = bform.assembly()
            b = lform.assembly()
            bc = DirichletBC(self.space, gd=pde.dirichlet)
            A, b = bc.apply(A, b)
            return spsolve(A, b, solver='scipy')
        elif method == 'convect_diff':
            if pde is None:
                raise ValueError("pde must be provided for conservat_diff'")
            mesh = self.mesh
            mesh.node = new_node
            am = self.assambly_method
            self.space.mesh = mesh
            source = pde.source(mesh.node)
            qf = mesh.quadrature_formula(self.p+2, 'cell')
            bcs, ws = qf.get_quadrature_points_and_weights()
            source = self.space.value(source, bcs)
            q = self.p+2
            a = pde.a[None,None,:]
            b = pde.b
            SDI = ScalarDiffusionIntegrator(coef= b, q = q, method=am)
            SCI = ScalarConvectionIntegrator(coef = a, q = q, method=am)
            SSI = ScalarSourceIntegrator(source=source, q = q, method=am)
            bform = BilinearForm(self.space)
            lform = LinearForm(self.space)
            bform.add_integrator(SDI,SCI)
            lform.add_integrator(SSI)
            SSI.clear()
            A = bform.assembly()
            b = lform.assembly()
            bc = DirichletBC(self.space, pde.dirichlet)
            A,b = bc.apply(A,b)
            return spsolve(A, b,  solver='scipy')
        else:
            raise ValueError(f"Unknown method: {method}")
        
    def construct(self,new_node,pde = None,method = 'interpolate'):
        """
        @brief construct: 重构信息
        @param new_mesh:新的节点
        @param pde: pde 数据
        @param method: 插值方法
        """
        # node 更新之前完成插值
        self.uh = self.handle_method(new_node,pde,method)
        if method != 'poisson':
            self.mesh.node = new_node
            self.space.mesh = self.mesh
        self.node = new_node
        self.cm = self.mesh.entity_measure('cell')
        self.star_measure = self.get_star_measure()
        self.G,self.M = self.get_control_function()

    def preprocessor(self,uh0,pde = None, steps = 10):
        """
        @brief preprocessor: 预处理器
        @param steps: 伪时间步数
        """
        self.uh = 1/steps * uh0
        self.G,self.M = self.get_control_function()
        for i in range(steps):
            t = (i+1)/steps
            self.uh = t * uh0
            self.mesh , self.uh = self.mesh_redistribution(self.uh,pde = pde)
            # self.mesh , self.uh = self.mesh_redistribution(self.uh)
            uh0 = self.uh
        return self.mesh , self.uh
            

    def caculate_tol(self):
        """
        @brief caculate_tol: 计算容许误差
        """
        logic_mesh = self.logic_mesh
        logic_cm = logic_mesh.entity_measure('cell')
        logic_em = logic_mesh.entity_measure('edge')
        cell2edge = logic_mesh.cell_to_edge()
        em_cell = logic_em[cell2edge]
        p = self.p
        if self.TD == 3:
            mul = em_cell[:,:3]*bm.flip(em_cell[:, 3:],axis=1)
            p = 0.5*bm.sum(mul,axis=1)
            d = bm.min(bm.sqrt(p*(p-mul[:,0])*(p-mul[:,1])*(p-mul[:,2]))/(3*logic_cm))
        else:
            if self.g_type == "Simplexmesh" :
                d = bm.min(bm.prod(em_cell,axis=1)/(2*logic_cm)).item()
            else:
                logic_node = logic_mesh.node
                logic_cell = logic_mesh.cell
                k = bm.arange((p+1)**2 , **self.kwargs1)
                k = k.reshape(p+1,p+1)
                con0 = logic_node[logic_cell[:,k[0,0]]]
                con1 = logic_node[logic_cell[:,k[-1,-1]]]
                con2 = logic_node[logic_cell[:,k[0,-1]]]
                con3 = logic_node[logic_cell[:,k[-1,0]]]
                e0 = bm.linalg.norm(con0 - con1,axis=1)
                e1 = bm.linalg.norm(con2 - con3,axis=1)
                d = bm.min(bm.array([e0,e1])).item()
        return d*0.1/p
    
    def mesh_redistribution(self ,uh, tol = None , pde = None ,method = 'interpolate',maxit = 1000):
        """
        @brief mesh_redistribution: 网格重构算法
        @param tol: 容许误差
        @param maxit 最大迭代次数
        @param pde: pde 数据(可选)
        @param method: 插值方法(默认为 interpolate)
        """
        import matplotlib.pyplot as plt
        self.uh = uh
        if tol is None:
            tol = self.tol
            print(f'容许误差为{tol}')

        for i in range(maxit):
            logic_node,vector_field = self.solve_move_LogicNode(G = self.G)
            
            L_infty_error = bm.max(bm.linalg.norm(self.logic_node - logic_node,axis=1))
            print(f'第{i+1}次迭代的差值为{L_infty_error}')
            if L_infty_error < tol:
                print(f'迭代总次数:{i+1}次')
                return self.mesh , self.uh
            elif i == maxit - 1:
                print('超出最大迭代次数')
                break
            node = self.get_physical_node(vector_field,logic_node)
            self.construct(node ,pde = pde , method = method)
