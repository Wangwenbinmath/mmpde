from fealpy.backend import backend_manager as bm
import matplotlib.pyplot as plt
from harmap_mmpde import *
from mesh_generator import MeshGenerator,high_order_meshploter
from pde_data_2d import Poissondata2d
from harmap_mmpde_data import *
from sympy import *
from fealpy.functionspace import LagrangeFESpace,ParametricLagrangeFESpace
from fealpy.fem import (BilinearForm 
                                     ,ScalarDiffusionIntegrator
                                     ,LinearForm
                                     ,ScalarSourceIntegrator
                                     ,DirichletBC)
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix


class PDEData():
    def __init__(self , u :str ,x : str ,y : str, D = [0,1,0,1]):
        u = sympify(u)
        self.u = lambdify([x,y], u ,'numpy')
        f_str = -diff(u,x,2) - diff(u,y,2)
        self.f = lambdify([x,y], f_str)
        self.grad_ux = lambdify([x,y], diff(u,x,1))
        self.grad_uy = lambdify([x,y], diff(u,y,1))
        self.domain = D

    def domain(self):
        return self.domain
    
    def solution(self, p):
        x = p[...,0]
        y = p[...,1]

        return self.u(x,y)
    
    def source(self,p):
        x = p[...,0]
        y = p[...,1]
        return self.f(x,y)

    def gradient(self,p):
        x = p[...,0]
        y = p[...,1]
        val = bm.zeros_like(p)
        val[...,0] = self.grad_ux(x,y)
        val[...,1] = self.grad_uy(x,y)
        return val
    
    def dirichlet(self,p):
        return self.solution(p)
    

def test_harmap_mmpde(beta , mol_times , redistribute):
    mesh = MeshGenerator.get_mesh(nx = 30 , ny = 30 , meshtype='tri')
    pde = PDEData(function_data['u2'] , x='x',y='y' , D = [0,1,0,1])
    print('Number of points:', mesh.number_of_nodes())
    print('Number of cells:', mesh.number_of_cells())

    p = 1
    init_mesh = TriangleMesh(mesh.node,mesh.cell)
    space = LagrangeFESpace(init_mesh, p=p)
    uh0 = space.interpolate(pde.solution)
    Vertex = bm.array([[0,0],[1,0],[1,1],[0,1]],dtype= bm.float64)
    MDH = Mesh_Data_Harmap(init_mesh, Vertex)
    Vertex_idx,Bdinnernode_idx,Arrisnode_idx = MDH.get_basic_infom()
    HMP = Harmap_MMPDE(init_mesh,uh0,beta = beta 
                       ,Vertex_idx=Vertex_idx 
                       ,Bdinnernode_idx=Bdinnernode_idx
                       ,Arrisnode_idx=Arrisnode_idx
                       ,mol_times= mol_times , redistribute=redistribute)
    
    mesh_moved,uh_moved = HMP.mesh_redistribution(uh0,maxit=2000)

    node= mesh_moved.node
    cell = mesh_moved.cell
    space = LagrangeFESpace(mesh_moved, p=p)
    uh = space.interpolate(pde.solution)
    error0 = mesh.error(space.function(array = uh0) ,pde.solution)
    error1 = mesh_moved.error(space.function(array = uh) ,pde.solution)
    
    print('旧网格插值误差:',error0)
    print('新网格插值误差:',error1)
    
    error0_color = mesh.error(space.function(array = uh0) ,pde.solution,celltype=True)
    error1_color = mesh_moved.error(space.function(array = uh) ,pde.solution,celltype=True)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_trisurf(node[:, 0], node[:, 1], pde.solution(node),
                     triangles = cell, cmap='viridis', 
                     edgecolor='blue',linewidth=0.2)
    
    fig , axes0 = plt.subplots(1,2)
    mesh.add_plot(axes0[0])
    mesh.add_plot(axes0[1] ,cellcolor=error0_color)
    fig = plt.figure()
    axes1 = fig.gca()
    mesh_moved.add_plot(axes1)
    # fig , axes1 = plt.subplots(1,2)
    # mesh_moved.add_plot(axes1[0])
    # mesh_moved.add_plot(axes1[1] ,cellcolor=error1_color)
    plt.show()

def test_quadmesh_harmap(beta , moltimes):
    # delete_bound = bm.array([0.51,1,0,0.49])
    delete_bound = None
    mesh = MeshGenerator.get_mesh(nx = 30, ny = 30 , meshtype='quad' , delete_bound=delete_bound)
    pde = Poissondata2d(function_data['u4'] , x='x',y='y' , D = [0,1,0,1])
    print('Number of points:', mesh.number_of_nodes())
    print('Number of cells:', mesh.number_of_cells())
    
    p = 1
    init_mesh = QuadrangleMesh(mesh.node,mesh.cell)
    
    space = LagrangeFESpace(init_mesh, p=p)
    uh0 = space.interpolate(pde.solution)
    Vertex = bm.array([[0,0],[1,0],[1,1],[0,1]],dtype= bm.float64)
    # Vertex = bm.array([[0,0],[0.5,0],[0.5,0.5],[1,0.5],[1,1],[0,1]],dtype= bm.float64)
    MDH = Mesh_Data_Harmap(init_mesh, Vertex)
    Vertex_idx,Bdinnernode_idx,sort_bdnode_idx = MDH.get_basic_infom()
    HMP = Harmap_MMPDE(init_mesh,uh0,beta = beta
                       ,Vertex_idx=Vertex_idx
                       ,Bdinnernode_idx=Bdinnernode_idx
                       ,sort_BdNode_idx=sort_bdnode_idx
                       ,mol_times= moltimes)
    M = HMP.M
    # pln , mvf = HMP.solve_move_LogicNode()
    # new_node = HMP.get_physical_node(mvf , pln)
    # mesh_moved = QuadrangleMesh(new_node,init_mesh.cell)
    mesh_moved,uh_moved = HMP.mesh_redistribution(uh0,pde = pde,method='interpolate')
    logic_mesh = HMP.logic_mesh
    fig = plt.figure()
    ax1 = fig.gca()
    # ax2 = fig.add_subplot(111, projection='3d')
    mesh_moved.add_plot(ax1)
    plt.show()

def test_lagtrimesh_harmap(beta , moltimes):
    delete_bound = None
    p = 2
    mesh = MeshGenerator.get_mesh(nx = 30, ny = 30 , meshtype='lagtri' ,p = p, delete_bound=delete_bound)
    pde = Poissondata2d(function_data['u3'] , x='x',y='y' , D = [0,1,0,1])
    print('Number of points:', mesh.number_of_nodes())
    print('Number of cells:', mesh.number_of_cells())
    
    space = ParametricLagrangeFESpace(mesh, p=p)
    uh0 = space.interpolate(pde.solution)
    Vertex = bm.array([[0,0],[1,0],[1,1],[0,1]],dtype= bm.float64)
    # Vertex = bm.array([[0,0],[0.5,0],[0.5,0.5],[1,0.5],[1,1],[0,1]],dtype= bm.float64)
    MDH = Mesh_Data_Harmap(mesh, Vertex)
    Vertex_idx,Bdinnernode_idx,sort_bdnode_idx = MDH.get_basic_infom()
    HMP = Harmap_MMPDE(mesh,uh0,beta = beta
                       ,Vertex_idx=Vertex_idx
                       ,Bdinnernode_idx=Bdinnernode_idx
                       ,sort_BdNode_idx=sort_bdnode_idx
                       ,mol_times= moltimes)
    M = HMP.M
    # pln , mvf = HMP.solve_move_LogicNode()
    # new_node = HMP.get_physical_node(mvf , pln)
    # mesh_moved = QuadrangleMesh(new_node,init_mesh.cell)
    mesh_moved,uh_moved = HMP.mesh_redistribution(uh0,pde = pde,method='interpolate')
    high_order_meshploter(mesh_moved,uh_moved,model='mesh',scat_node=False)

def test_lagquadmesh_harmap(beta , moltimes):
    # delete_bound = bm.array([0.51,1,0,0.49])
    delete_bound = None
    p = 2
    mesh = MeshGenerator.get_mesh(nx = 20, ny = 20 , meshtype='lagquad' ,p = p, delete_bound=delete_bound)
    mesh0 = MeshGenerator.get_mesh(nx = 20, ny = 20 , meshtype='lagquad' ,p = p, delete_bound=delete_bound)
    pde = Poissondata2d(function_data['u1'] , x='x',y='y' , D = [0,1,0,1])
    print('Number of points:', mesh.number_of_nodes())
    print('Number of cells:', mesh.number_of_cells())

    space = ParametricLagrangeFESpace(mesh, p=p)
    uh0 = space.interpolate(pde.solution)
    Vertex = bm.array([[0,0],[1,0],[1,1],[0,1]],dtype= bm.float64)
    # Vertex = bm.array([[0,0],[0.5,0],[0.5,0.5],[1,0.5],[1,1],[0,1]],dtype= bm.float64)
    HMP = Harmap_MMPDE(mesh,uh0,beta = beta
                       ,Vertex=Vertex
                       ,alpha= 0.25
                       ,mol_times= moltimes)
    new_mesh , new_uh = HMP.mesh_redistribution(uh0,pde = pde,method='solution')
    error0 = mesh0.error(space.function(array = uh0) ,pde.solution)
    error1 = new_mesh.error(space.function(array = new_uh) ,pde.solution)

    print('旧网格插值误差:',error0)
    print('新网格插值误差:',error1)
    print('误差比:',error1/error0)

    high_order_meshploter(new_mesh,model='mesh',scat_node=True)

def test_new_harmap(beta , moltimes):
    # delete_bound = bm.array([0.51,1,0,0.49])
    delete_bound = None
    p = 2
    mesh = MeshGenerator.get_mesh(nx = 20, ny = 20 , meshtype='lagtri' ,p = p, delete_bound=delete_bound)
    pde = Poissondata2d(function_data['u4'] , x='x',y='y' , D = [0,1,0,1])
    print('Number of points:', mesh.number_of_nodes())
    print('Number of cells:', mesh.number_of_cells())
    print(mesh.node[mesh.cell[0]])
    space = ParametricLagrangeFESpace(mesh, p=p)
    uh0 = space.interpolate(pde.solution)
    Vertex = bm.array([[0,0],[1,0],[1,1],[0,1]],dtype= bm.float64)
    # Vertex = bm.array([[0,0],[0.5,0],[0.5,0.5],[1,0.5],[1,1],[0,1]],dtype= bm.float64)
    HMP = Harmap_MMPDE(mesh,uh0,beta = beta
                       ,Vertex=Vertex
                       ,mol_times= moltimes)

    # logic_mesh = HMP.logic_mesh
    new_mesh , new_uh = HMP.mesh_redistribution(uh0,pde = pde,method='interpolate')
    high_order_meshploter(new_mesh,model='mesh',scat_node=True)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    M = HMP.M
    # _,M = HMP.get_control_function()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    node = new_mesh.node
    cell = new_mesh.cell
    # cell = bm.concat([cell[:,[0,3,4,1]],cell[:,[1,4,5,2]],
    #                                 cell[:,[3,6,7,4]],cell[:,[4,7,8,5]]],axis = 0)
    cell = bm.concat([cell[:,0:3],cell[:,[1,3,4]],
                                  cell[:,[1,4,2]],cell[:,[2,4,5]]],axis = 0)
    verts = node[cell]
    z = M[cell]
    verts_3d = bm.concat([verts,z[...,None]],axis = -1)
    poly = Poly3DCollection(verts_3d, edgecolors='k', linewidths=0.2, alpha=0.5)
    ax.add_collection3d(poly)
    z_max = bm.max(M)
    z_min = bm.min(M)
    ax.set_zlim(z_min, z_max)
    ax.set_title('Surface plot of M')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('M')
    plt.show()
    # node = mesh.node
    # cell = mesh.cell
    # cell = bm.concat([cell[:,0:3],cell[:,[1,3,4]],
    #                               cell[:,[1,4,2]],cell[:,[2,4,5]]],axis = 0)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111, projection='3d')
    # ax1.plot_trisurf(node[:, 0], node[:, 1], M,
    #                  triangles = cell, cmap='viridis', 
    #                  edgecolor='blue',linewidth=0.2)
    # plt.show()

if __name__ == '__main__':
    beta_ = 400
    mol_times = 28
    redistribute = False
    # test_harmap_mmpde(beta_ , mol_times , redistribute)
    # test_quadmesh_harmap(beta_ , mol_times)
    # test_lagtrimesh_harmap(beta_ , mol_times)
    test_lagquadmesh_harmap(beta_ , mol_times)
    # test_new_harmap(beta_ , mol_times)
    

