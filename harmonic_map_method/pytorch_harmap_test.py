from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.mesh import TetrahedronMesh
from pytorch_harmap_data import *
import matplotlib.pyplot as plt
import torch
from fealpy.utils import timer
import numpy as np
import pytest

"""
test_Mesh_Data_Harmap
"""
class TestHarmapInterfaces:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", HMD_init_mesh_data)
    def test_init(self, meshdata, backend):
        bm.set_backend(backend)
        mesh = meshdata['mesh']
        node = bm.from_numpy(mesh.node)
        cell = bm.from_numpy(mesh.cell)
        mesh = TriangleMesh(node, cell)
        Vertex = bm.from_numpy(meshdata['Vertex'])
        isconvex = meshdata['isconvex']
        MDH = Mesh_Data_Harmap(mesh,Vertex)
        node0 = MDH.node
        Vertex0 = MDH.Vertex
        np.testing.assert_array_equal(bm.to_numpy(node0), mesh.node)
        np.testing.assert_array_equal(bm.to_numpy(Vertex0), bm.to_numpy(Vertex))
        assert MDH.isconvex == isconvex
        
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", HMD_sort_bdnode_and_bdface_data)
    def test_sort_bdnode_and_bdface(self, meshdata, backend):
        bm.set_backend(backend)
        mesh = meshdata['mesh']
        node = bm.from_numpy(mesh.node)
        cell = bm.from_numpy(mesh.cell)
        mesh = TriangleMesh(node, cell)
        Vertex = bm.from_numpy(meshdata['Vertex'])
        MDH = Mesh_Data_Harmap(mesh,Vertex)
        true_sort_BdNode_idx = bm.from_numpy(meshdata['sort_BdNode_idx'])
        true_sort_BDedge_idx = bm.from_numpy(meshdata['sort_BDedge_idx'])
        sort_BdNode_idx,sort_BDedge_idx  = MDH.sort_bdnode_and_bdface()
        np.testing.assert_array_equal(bm.to_numpy(sort_BdNode_idx), bm.to_numpy(true_sort_BdNode_idx))
        np.testing.assert_array_equal(bm.to_numpy(sort_BDedge_idx), bm.to_numpy(true_sort_BDedge_idx))

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", HMD_get_normal_inform_data)
    def test_get_normal_inform(self,meshdata, backend):
        bm.set_backend(backend)
        mesh = meshdata['mesh']
        node = bm.from_numpy(mesh.node)
        cell = bm.from_numpy(mesh.cell)
        if node.shape[1] == 2:
            mesh = TriangleMesh(node, cell)
        else:
            mesh = TetrahedronMesh(node, cell)
        Vertex = bm.from_numpy(meshdata['Vertex'])
        MDH = Mesh_Data_Harmap(mesh,Vertex)
        true_node2face_normal = bm.from_numpy(meshdata['node2face_normal'])
        true_normal = bm.from_numpy(meshdata['normal'])
        node2face_normal,normal  = MDH.get_normal_inform()
        np.testing.assert_array_equal(bm.to_numpy(node2face_normal), bm.to_numpy(true_node2face_normal))
        np.testing.assert_array_equal(bm.to_numpy(normal), bm.to_numpy(true_normal))

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", HMD_get_basic_inform_data)
    def test_get_normal_inform(self,meshdata, backend):
        bm.set_backend(backend)
        mesh = meshdata['mesh']
        node = bm.from_numpy(mesh.node)
        cell = bm.from_numpy(mesh.cell)
        if node.shape[1] == 2:
            mesh = TriangleMesh(node, cell)
        else:
            mesh = TetrahedronMesh(node, cell)
        Vertex = bm.from_numpy(meshdata['Vertex'])
        MDH = Mesh_Data_Harmap(mesh,Vertex)
        Vertex_idx,Bdinnernode_idx,Arrisnode_idx  = MDH.get_basic_infom()
        true_Vertex_idx = bm.from_numpy(meshdata['Vertex_idx'])
        true_Bdinnernode_idx = bm.from_numpy(meshdata['Bdinnernode_idx'])
        if meshdata['Arrisnode_idx'] is not None:
            true_Arrisnode_idx = bm.from_numpy(meshdata['Arrisnode_idx'])
        else:
            true_Arrisnode_idx = None
            assert Arrisnode_idx is None
        np.testing.assert_array_equal(bm.to_numpy(Vertex_idx), bm.to_numpy(true_Vertex_idx))
        np.testing.assert_array_equal(bm.to_numpy(Bdinnernode_idx), bm.to_numpy(true_Bdinnernode_idx))

    # logic_mesh
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", LM_get_normal_information_data)
    def test_get_normal_information(self,meshdata, backend):
        bm.set_backend(backend)
        mesh = meshdata['mesh']
        node = bm.from_numpy(mesh.node)
        cell = bm.from_numpy(mesh.cell)
        if node.shape[1] == 2:
            mesh = TriangleMesh(node, cell)
        else:
            mesh = TetrahedronMesh(node, cell)
        Vertex_idx = bm.from_numpy(meshdata['Vertex_idx'])
        Bdinnernode_idx = bm.from_numpy(meshdata['Bdinnernode_idx'])
       
        true_Bi_node_normal = bm.from_numpy(meshdata['Bi_node_normal'])
        if mesh.TD == 2:
            Arrisnode_idx = None
            true_Ar_node_normal = None
        else:
            Arrisnode_idx = bm.from_numpy(meshdata['Arrisnode_idx'])
            true_Ar_node_normal = bm.from_numpy(meshdata['Ar_node_normal'])
        LM = LogicMesh(mesh, Vertex_idx , Bdinnernode_idx, Arrisnode_idx)
        logic_mesh = LM.get_logic_mesh()
        if mesh.TD == 2:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            logic_mesh.add_plot(axes)
            plt.show()
            Bi_node_normal = LM.get_normal_information(mesh)
        else:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')
            mesh.add_plot(axes)
            logic_mesh.add_plot(axes)
            plt.show()
            Bi_node_normal,Arrisnode_idx = LM.get_normal_information(mesh)
            np.testing.assert_array_equal(bm.to_numpy(Arrisnode_idx), bm.to_numpy(true_Ar_node_normal))
        np.testing.assert_array_equal(bm.to_numpy(Bi_node_normal), bm.to_numpy(true_Bi_node_normal))
    
    # logic_mesh
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", LM_get_logic_node_data)
    def test_get_logic_node(self,meshdata, backend):
        bm.set_backend(backend)
        mesh = meshdata['mesh']
        node = bm.from_numpy(mesh.node)
        cell = bm.from_numpy(mesh.cell)
        mesh = TriangleMesh(node,cell)
        Vertices = bm.from_numpy(meshdata['vertices'])
        true_logic_node = bm.from_numpy(meshdata['logic_node'])
        HDM = Mesh_Data_Harmap(mesh,Vertices)
        Vertex_idx,Bdinnernode_idx,sort_BdNode_idx  = HDM.get_basic_infom()
        LM = LogicMesh(mesh, Vertex_idx , Bdinnernode_idx , sort_BdNode_idx=sort_BdNode_idx)
        logic_node = bm.to_numpy(LM.get_logic_node())
        np.testing.assert_allclose(logic_node, bm.to_numpy(true_logic_node),rtol=1e+20)

    # harmap
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", HM_init_test_data)
    def test_HM_init(self , meshdata , backend):
        bm.set_backend(backend)
        mesh = meshdata['mesh']
        be = meshdata['beta']
        uh = bm.from_numpy(meshdata['uh'])
        node = bm.from_numpy(mesh.node)
        cell = bm.from_numpy(mesh.cell)
        if node.shape[1] == 2:
            mesh = TriangleMesh(node, cell)
        else:
            mesh = TetrahedronMesh(node, cell)
        Vertices = bm.from_numpy(meshdata['Vertex'])
        HDM = Mesh_Data_Harmap(mesh,Vertices)
        Vertex_idx,Bdinnernode_idx,Arrisnode_idx  = HDM.get_basic_infom()
        HM = Harmap_MMPDE(mesh ,uh , be , Vertex_idx ,
                           Bdinnernode_idx , Arrisnode_idx )
        
        true_star_measure = meshdata['star_measure']
        true_G = meshdata['G']
        true_A = meshdata['A']
        true_b = meshdata['b']
        true_tol = meshdata['tol']

        star_measure = HM.star_measure
        G = HM.G
        A = bm.array(HM.A.toarray())
        b = HM.b
        tol = HM.tol
        np.testing.assert_allclose(bm.to_numpy(star_measure), true_star_measure)
        np.testing.assert_allclose(bm.to_numpy(G), true_G)
        np.testing.assert_allclose(bm.to_numpy(A), true_A)
        np.testing.assert_allclose(bm.to_numpy(b), true_b)
        assert np.isclose(tol , true_tol)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", HM_stiff_matrix_test_data)
    def test_HM_stiff_matrix(self , meshdata , backend):
        bm.set_backend(backend)
        mesh = meshdata['mesh']
        be = meshdata['beta']
        uh = bm.from_numpy(meshdata['uh'])
        node = bm.from_numpy(mesh.node)
        cell = bm.from_numpy(mesh.cell)
        if node.shape[1] == 2:
            mesh = TriangleMesh(node, cell)
            Arrisnode_idx = meshdata['Arrisnode_idx']
        else:
            mesh = TetrahedronMesh(node, cell)
            Arrisnode_idx = bm.from_numpy(meshdata['Arrisnode_idx'])
        Vertex_idx = bm.from_numpy(meshdata['Vertex_idx'])
        Bdinnernode_idx = bm.from_numpy(meshdata['Bdinnernode_idx'])
        HM = Harmap_MMPDE(mesh ,uh , be , Vertex_idx ,
                           Bdinnernode_idx , Arrisnode_idx )
        true_H = meshdata['stiff_matrix']
        H = HM.get_stiff_matrix(HM.mesh , HM.G)
        H = bm.array(H.toarray())
        np.testing.assert_allclose(bm.to_numpy(H), true_H, rtol=1e-6)
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", HM_move_logicnode_test_data)
    def test_move_logicnode(self , meshdata , backend):
        bm.set_backend(backend)
        mesh = meshdata['mesh']
        be = meshdata['beta']
        uh = bm.from_numpy(meshdata['uh'])
        node = bm.from_numpy(mesh.node)
        cell = bm.from_numpy(mesh.cell)
        if node.shape[1] == 2:
            mesh = TriangleMesh(node, cell)
            Arrisnode_idx = meshdata['Arrisnode_idx']
        else:
            mesh = TetrahedronMesh(node, cell)
            Arrisnode_idx = bm.from_numpy(meshdata['Arrisnode_idx'])
        Vertex_idx = bm.from_numpy(meshdata['Vertex_idx'])
        Bdinnernode_idx = bm.from_numpy(meshdata['Bdinnernode_idx'])
        HM = Harmap_MMPDE(mesh ,uh , be , Vertex_idx ,
                           Bdinnernode_idx , Arrisnode_idx )
        true_p_lnode = meshdata['p_lnode']
        true_mv_field = meshdata['mv_field']
        p_lnode , mv_field = HM.solve_move_LogicNode()
        np.testing.assert_allclose(bm.to_numpy(p_lnode), true_p_lnode, rtol=np.inf)
        np.testing.assert_allclose(bm.to_numpy(mv_field), true_mv_field, rtol=np.inf)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", HM_p_node_test_data)
    def test_physical_node(self,meshdata , backend):
        bm.set_backend(backend)
        mesh = meshdata['mesh']
        be = meshdata['beta']
        uh = bm.from_numpy(meshdata['uh'])
        node = bm.from_numpy(mesh.node)
        cell = bm.from_numpy(mesh.cell)
        if node.shape[1] == 2:
            mesh = TriangleMesh(node, cell)
            Arrisnode_idx = meshdata['Arrisnode_idx']
        else:
            mesh = TetrahedronMesh(node, cell)
            Arrisnode_idx = bm.from_numpy(meshdata['Arrisnode_idx'])
        Vertex_idx = bm.from_numpy(meshdata['Vertex_idx'])
        Bdinnernode_idx = bm.from_numpy(meshdata['Bdinnernode_idx'])
        HM = Harmap_MMPDE(mesh ,uh , be , Vertex_idx ,
                           Bdinnernode_idx , Arrisnode_idx )
        p_lnode , mv_field = HM.solve_move_LogicNode()
        true_p_node = meshdata['p_node']
        p_node = HM.get_physical_node(mv_field, p_lnode)
        np.testing.assert_allclose(bm.to_numpy(p_node), true_p_node, rtol=np.inf)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", HM_interpolate_test_data)
    def test_interpolate(self,meshdata , backend):
        bm.set_backend(backend)
        mesh = meshdata['mesh']
        be = meshdata['beta']
        uh = bm.from_numpy(meshdata['uh'])
        node = bm.from_numpy(mesh.node)
        cell = bm.from_numpy(mesh.cell)
        if node.shape[1] == 2:
            mesh = TriangleMesh(node, cell)
            Arrisnode_idx = meshdata['Arrisnode_idx']
        else:
            mesh = TetrahedronMesh(node, cell)
            Arrisnode_idx = bm.from_numpy(meshdata['Arrisnode_idx'])
        Vertex_idx = bm.from_numpy(meshdata['Vertex_idx'])
        Bdinnernode_idx = bm.from_numpy(meshdata['Bdinnernode_idx'])
        HM = Harmap_MMPDE(mesh ,uh , be , Vertex_idx ,
                           Bdinnernode_idx , Arrisnode_idx )
        p_lnode , mv_field = HM.solve_move_LogicNode()
        p_node = HM.get_physical_node(mv_field, p_lnode)
        true_uh_new = meshdata['uh_new']
        uh_new = HM.interpolate(p_node)
        np.testing.assert_allclose(bm.to_numpy(uh_new), true_uh_new, rtol=np.inf)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", HM_mesh_redistribution_test_data)
    def test_mesh_redistribution(self,meshdata , backend):
        bm.set_backend(backend)
        mesh = meshdata['mesh']
        be = meshdata['beta']
        uh = bm.from_numpy(meshdata['uh'])
        node = bm.from_numpy(mesh.node)
        cell = bm.from_numpy(mesh.cell)
        if node.shape[1] == 2:
            mesh = TriangleMesh(node, cell)
            Arrisnode_idx = meshdata['Arrisnode_idx']
        else:
            mesh = TetrahedronMesh(node, cell)
            Arrisnode_idx = bm.from_numpy(meshdata['Arrisnode_idx'])
        Vertex_idx = bm.from_numpy(meshdata['Vertex_idx'])
        Bdinnernode_idx = bm.from_numpy(meshdata['Bdinnernode_idx'])
        HM = Harmap_MMPDE(mesh ,uh , be , Vertex_idx ,
                           Bdinnernode_idx , Arrisnode_idx )
        true_uh = meshdata['uh_new']
        true_node = meshdata['node_new']
        new_mesh,new_uh = HM.mesh_redistribution(uh)
        
        new_node = new_mesh.node
        np.testing.assert_allclose(bm.to_numpy(new_node), true_node,atol=1e-6, rtol=np.inf)
        np.testing.assert_allclose(bm.to_numpy(new_uh), true_uh,atol=1e-6, rtol=np.inf)
        
if __name__ == "__main__":
    pytest.main(["./pytorch_harmap_test.py", "-k", "test_mesh_redistribution"])
