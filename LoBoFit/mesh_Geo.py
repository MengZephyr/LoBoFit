from mesh_utils import *
from data_IO import int_numpy_to_tensor
import torch
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

EPS = 1e-8


class LaplacianOperator(object):
    def __init__(self, base_verts: torch.Tensor, faces: torch.Tensor):
        self.faces = faces
        self.edge_ij, self.w_ij = self.edge_weights_cot(base_verts)
        self.num_verts = base_verts.shape[0]

    def edge_weights_cot(self, verts, eps=EPS):
        faces = self.faces
        N = verts.shape[0]
        dtype = verts.dtype

        """
        w_ij = 0.5*(cot α + cot β) ）。
        return: edges: [E,2], w: [E]
        """
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]

        # edge_to_angle cot
        def cot(a, b):
            # cot(∠(a,b)) = <a,b> / ||a×b||
            c = (a * b).sum(-1)
            s = torch.cross(a, b, dim=-1).norm(dim=-1).clamp_min(eps)
            return c / s

        cot0 = cot(v1 - v2, v0 - v2)  # at v2 -> edge (v0,v1)
        cot1 = cot(v2 - v0, v1 - v0)  # at v0 -> edge (v1,v2)
        cot2 = cot(v0 - v1, v2 - v1)  # at v1 -> edge (v2,v0)

        # (v0,v1) -> cot2；(v1,v2) -> cot0；(v2,v0) -> cot1
        I = torch.stack([
            torch.stack([faces[:, 0], faces[:, 1]], dim=1),
            torch.stack([faces[:, 1], faces[:, 2]], dim=1),
            torch.stack([faces[:, 2], faces[:, 0]], dim=1),
        ], dim=0).reshape(-1, 2)  # [3F, 2]

        Wval = torch.cat([cot2, cot0, cot1], dim=0) * 0.5  # [3F]

        # (i<j)
        I_sorted, _ = torch.sort(I, dim=1)
        mask = I_sorted[:, 0] != I_sorted[:, 1]  # mask out i==j
        Iu = I_sorted[mask]  # [E,2]
        Wu = Wval[mask]  # [E]

        idx = Iu.t().contiguous()  # [2,E]
        A = torch.sparse_coo_tensor(idx, Wu.to(dtype), size=(N, N)).coalesce()
        ij = A.indices().t().to(torch.long)  # [Euniq, 2]
        w = A.values().to(dtype)  # [Euniq]

        keep = ij[:, 0] < ij[:, 1]
        edges = ij[keep]
        w = w[keep]

        # uniq_edges, inv = torch.unique(Iu, dim=0, return_inverse=True)  # [Euniq,2], [E]
        # w = torch.zeros(uniq_edges.size(0), device=verts.device, dtype=dtype)
        # w.scatter_add_(0, inv, Wu.to(dtype))  #


        # I = torch.stack([
        #     faces[:, 0], faces[:, 1],
        #     faces[:, 1], faces[:, 2],
        #     faces[:, 2], faces[:, 0]
        # ], dim=1).reshape(-1, 2)  # [3F*2, 2]
        # Wval = torch.cat([cot2, cot2, cot0, cot0, cot1, cot1], dim=0) * 0.5  # [3F*2]
        # a = torch.min(I[:, 0], I[:, 1])
        # b = torch.max(I[:, 0], I[:, 1])
        # key = a * verts.shape[0] + b
        #
        # uniq_key, inv = torch.unique(key, return_inverse=True)
        # w_sum = torch.zeros_like(uniq_key, dtype=verts.dtype)
        # w_sum = w_sum.scatter_add(0, inv, Wval)
        #
        # edges = torch.stack([uniq_key // verts.shape[0], uniq_key % verts.shape[0]], dim=1)

        # print(edges.min(), edges.max())
        # print(self.faces.min(), self.faces.max())
        # print(edges.shape, w.shape)
        # exit(1)
        return edges.to(torch.long), w

    def laplacian_matvec(self, v:torch.Tensor, if_norm=True, eps=EPS):
        """
        (L v)_i = sum_{j∈N(i)} w_ij (v_i - v_j)
        v: [N,3] or [N, d]
        edges: [E,2] no-direction
        w_ij: [E]
        """
        edges_ij = self.edge_ij
        w_ij = self.w_ij
        i, j = edges_ij[:, 0], edges_ij[:, 1]
        out = torch.zeros_like(v)
        diff = v[i] - v[j]  # [E, *]
        contrib = w_ij[:, None] * diff  # [E, *]
        out.index_add_(0, i, contrib)
        out.index_add_(0, j, -contrib)

        if if_norm:
            s = out.norm().clamp_min(eps).detach()
            out = out/s

        return out



class MeshTopology(object):
    def __init__(self, num_verts, geo_faces):
        self.num_verts = num_verts
        self.num_faces = geo_faces.shape[0]
        self.faces = geo_faces

        # create edge topology
        self.edges = None
        self.face_adjacency = None
        self.face_adjacency_edges = None
        self.curve_edges = None
        self.create_edge_info()
        self.curve_cycle_links = None

    def create_edge_info(self):
        self.edges, \
            self.face_adjacency, self.face_adjacency_edges, self.curve_edges = faces_to_edges_and_adjacency(self.faces)

    def to_tensor(self, device):
        self.faces = int_numpy_to_tensor(self.faces, device)
        self.face_adjacency = int_numpy_to_tensor(self.face_adjacency, device)
        self.face_adjacency_edges = int_numpy_to_tensor(self.face_adjacency_edges, device)
        self.edges = int_numpy_to_tensor(self.edges, device)

    def create_curve_links(self, check_degree2=True):
        self.curve_cycle_links = extract_cycle_edge_chains(self.curve_edges, check_degree2)

    def compute_curve_adj_cos(self, verts, if_need_wei: bool, eps: float = EPS):
        if self.curve_cycle_links is None:
            self.create_curve_links()
        num_chains, c_chains = self.curve_cycle_links

        device = verts.device
        chain_adj_cos = []
        chain_adj_wei = []
        for i in range(num_chains):
            ch = torch.as_tensor(c_chains[i], device=device, dtype=torch.long)

            e = verts[ch[:, 0]] - verts[ch[:, 1]]  # [E,3]
            e_len = e.norm(dim=-1)  # [E]
            e_unit = e / (e_len.unsqueeze(-1) + eps)  # [E,3]

            e_unit_next = torch.roll(e_unit, shifts=-1, dims=0)  # [E,3]
            e_adj_cos = (e_unit * e_unit_next).sum(dim=-1)  # [E]
            chain_adj_cos.append(e_adj_cos)

            if if_need_wei:
                e_len_next = torch.roll(e_len, shifts=-1, dims=0)  # [E]
                e_w = e_len + e_len_next  # [E]
                sum_w = e_w.sum().clamp_min(eps)
                e_w = e_w / sum_w
                chain_adj_wei.append(e_w)

        return chain_adj_cos, chain_adj_wei, num_chains

    def compute_dihedral_angle(self, verts, if_need_weight:bool):
        face_n = compute_face_normals(verts, self.faces)
        adj_angles = dihedral_angle_adjacent_faces(normals=face_n, adjacency=self.face_adjacency)

        w = None
        if if_need_weight:
            e = verts[self.face_adjacency_edges[:, 1], :] - verts[self.face_adjacency_edges[:, 0], :]
            w = e.norm(dim=-1)
            w = w / w.sum().clamp_min(EPS)

        return adj_angles, w




class StaticKDTree(object):
    def __init__(self, verts:torch.Tensor,
                       faces:torch.Tensor, device,
                       algorithm: str = "kd_tree", # 'kd_tree' | 'ball_tree' | 'auto'
                       leaf_size: int = 40,
                       metric: str = "euclidean",
                       n_jobs: int = -1,
                 ):
        self.points = compute_face_centers(verts, faces)
        self.normals = compute_face_normals(verts, faces)
        self.area = compute_mesh_areas(verts, faces)
        self.device = device

        pnts = self.points.clone()
        pnts = pnts.detach().cpu().numpy()
        #self.KD_tree = neighbors.KDTree(pnts)
        self.nn = NearestNeighbors(
            algorithm=algorithm, leaf_size=leaf_size, metric=metric, n_jobs=n_jobs
        )
        self.nn.fit(pnts)

    @torch.no_grad()
    def kneighbours(self, query: torch.Tensor, k: int = 1):
        q_np = query.detach().cpu().numpy()
        #_, vInd = self.KD_tree.query(q_np, k=k)
        #inds = [i[0] for i in vInd]
        _, inds = self.nn.kneighbors(q_np, n_neighbors=k, return_distance=True)
        inds = torch.from_numpy(np.asarray(inds)).to(query.device, dtype=torch.long).squeeze(-1)

        sel_point = self.points[inds, :]
        sel_norms = self.normals[inds, :]
        sel_area = self.area[inds]
        return inds, sel_point, sel_norms, sel_area


def creat_o3d_mesh(vertarray, facearray, ifnormals=True):
    objmesh = o3d.geometry.TriangleMesh()
    objmesh.vertices = o3d.utility.Vector3dVector(vertarray)
    objmesh.triangles = o3d.utility.Vector3iVector(facearray)
    if ifnormals:
        objmesh.compute_vertex_normals(normalized=True)

    return objmesh


class MeshHandling(object):
    def __init__(self, vertarray, facearray):
        self.objmesh = creat_o3d_mesh(vertarray, facearray, ifnormals=True)

        self.trimesh = o3d.t.geometry.TriangleMesh.from_legacy(self.objmesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        _ = self.scene.add_triangles(self.trimesh)

        self.bb_min = self.objmesh.get_min_bound()
        self.bb_max = self.objmesh.get_max_bound()
        self.half_bbLen = 0.5 * np.linalg.norm(self.bb_max-self.bb_min)

    def random_sample_points(self, num):
        sample_pc = self.objmesh.sample_points_poisson_disk(num)
        sample_pnts = np.asarray(sample_pc.points)
        sample_normals = np.asarray(sample_pc.normals)
        return sample_pnts, sample_normals

    def query_sdf(self, points):
        query_point = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
        sdf = self.scene.compute_signed_distance(query_point)
        return sdf.numpy()

    def get_vert_normals(self):
        return np.asarray(self.objmesh.vertex_normals)

    def querry_nearest_points(self, qverts):
        query_point = o3d.core.Tensor(qverts, dtype=o3d.core.Dtype.Float32)
        ans = self.scene.compute_closest_points(query_point)
        return ans['points'].numpy(), ans['primitive_ids'].numpy(), ans['primitive_normals'].numpy()

    def ray_intersection_list(self, ray_orig, ray_dir):
        #rays = np.concatenate([orig, self.half_bbLen*ray_dir], axis=-1, dtype=np.float32)
        rays = np.concatenate([ray_orig, self.half_bbLen * ray_dir], axis=-1).astype(np.float32)
        rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        ans = self.scene.list_intersections(rays)

        ray_split = ans['ray_splits'].numpy()
        hit_pos, hit_norm = self.hits_to_points_from_bary(ans)
        hit_primIDs = ans['primitive_ids'].numpy().ravel()
        hit_primUVs = ans['primitive_uvs'].numpy()
        return ray_split, hit_pos, hit_norm, hit_primIDs, hit_primUVs

    def hits_to_points_from_bary(self, ans):
        v = self.trimesh.vertex['positions'].numpy()  # [V,3]
        tri = self.trimesh.triangle['indices'].numpy()  # [F,3]
        tidx = ans['primitive_ids'].numpy().ravel()  # [K]
        uv = ans['primitive_uvs'].numpy()  # [K,2]
        u, v_uv = uv[:, 0], uv[:, 1]
        w = 1 - u - v_uv
        p0 = v[tri[tidx, 0], :]
        p1 = v[tri[tidx, 1], :]
        p2 = v[tri[tidx, 2], :]
        pts = p1 * u[:, None] + p2 * v_uv[:, None] + p0 * w[:, None]
        e1 = p1 - p0
        e2 = p2 - p0
        nrm = np.cross(e1, e2)

        n_len = np.linalg.norm(nrm, axis=1, keepdims=True)
        n_len = np.clip(n_len, 1e-12, None)
        nrm = nrm / n_len

        return pts, nrm

    def query_hits(self, ray_orig, ray_dir):
        rays = np.concatenate([ray_orig, self.half_bbLen * ray_dir], axis=-1).astype(np.float32)
        rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        ans = self.scene.cast_rays(rays)
        t_hit = ans['t_hit'].numpy() #(N, )
        prim_ids =ans['primitive_ids'].numpy().ravel() #(N, )

        hit_mask = (t_hit < self.half_bbLen * 0.5) & (prim_ids != -1)
        valid_idx = np.nonzero(hit_mask)[0]
        hit_face_ids = prim_ids[valid_idx]
        return np.asarray(valid_idx, dtype=np.int64), np.asarray(hit_face_ids, dtype=np.int64)





