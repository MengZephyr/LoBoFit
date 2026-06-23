import numpy as np
import torch

from bone_projection import create_body_bone_weights, BoneStructure, \
    Relative_Vector_Projection, Inverse_Relative_Projection, \
    select_most_related_bones_by_abkw, bone_related_cast_ray

from data_IO import readObj_vert_feats, readObj_faces, \
    float_numpy_to_tensor, int_numpy_to_tensor, float_tensor_to_numpy
from mesh_utils import kdTree_nearest_neighbor, np_feats_interpolation, compute_mesh_areas
from mesh_Geo import LaplacianOperator, MeshTopology, MeshHandling, StaticKDTree

def filter_ray_hits_withweight(ray_split, hit_pos, hit_norms, hit_primIDs, hit_primUVs, hit_weights,
                               ray_ori, ray_dir, ray_weights, *,
                               opposite_thresh=0.5, wei_align_thresh=0.3):
    R = len(ray_split) - 1
    sel_idx = np.full(R, -1, dtype=np.int64)
    sel_pos = np.full((R, 3), np.nan, dtype=np.float32)
    sel_nrm = np.full((R, 3), np.nan, dtype=np.float32)
    sel_fID = np.full(R, -1, dtype=np.int64)
    sel_fUV = np.full((R, 2), np.nan, dtype=np.float32)
    sel_t = np.full(R, np.nan, dtype=np.float32)

    d_len = np.linalg.norm(ray_dir, axis=1, keepdims=True)
    d_len = np.clip(d_len, 1e-12, None)
    d_unit = ray_dir / d_len

    for i in range(R):
        s, e = int(ray_split[i]), int(ray_split[i + 1])
        if s >= e:
            continue  # no ray hitting

        p = hit_pos[s:e]  # [M,3]
        n = hit_norms[s:e]  # [M,3]
        pw = hit_weights[s:e]  # [M,K]
        o = ray_ori[i]  # [3]
        d = d_unit[i]  # [3]
        rw = ray_weights[i]  # [k]

        # cos between ray wei and hit wei
        weicos = np.einsum('ik,k->i', pw, rw)
        # distance from the hit to the orig.
        t = np.einsum('ij,j->i', p - o[None, :], d)  # [M]
        base_mask = (t >= 0.0) & (weicos >= wei_align_thresh)
        if not np.any(base_mask):
            continue

        # cos between ray dir and the hit normal
        cos = np.einsum('ij,j->i', n, d)  # [M]

        order = np.argsort(t)
        p, n, t, cos = p[order], n[order], t[order], cos[order]
        global_ids = (np.arange(s, e))[order]

        # get the first hit at the normal opposite surface
        opp_mask = base_mask & (cos <= float(opposite_thresh))
        if np.any(opp_mask):
            first_opp_idx = np.argmax(opp_mask)  # get the first opposite normal hit id
            t_barrier = t[first_opp_idx]
            cand_mask = base_mask & (cos > float(opposite_thresh)) & (t < t_barrier)
        else:
            cand_mask = base_mask & (cos > float(opposite_thresh))

        if not np.any(cand_mask):
            continue
        # pick the farthest
        local_idx = np.argmax(t[cand_mask])
        cand_ids = np.where(cand_mask)[0]
        sel_local = cand_ids[local_idx]
        idx = int(global_ids[sel_local])

        sel_idx[i] = idx
        sel_pos[i] = hit_pos[idx]
        sel_nrm[i] = hit_norms[idx]
        sel_fID[i] = hit_primIDs[idx]
        sel_fUV[i] = hit_primUVs[idx]
        sel_t[i] = t[sel_local]

    return sel_idx, sel_pos, sel_nrm, sel_fID, sel_fUV, sel_t

class BodyMesh(object):
    def __init__(self, *, obj_name=None, lbs_name=None, bones_name=None):
        '''
        :param obj_name: source cannonical loading, target frame loading
        :param lbs_name: source & target cannonical loading
        :param bones_name: source & target frame loading
        '''

        self.LBS_weight = None
        if lbs_name is not None:
            self.LBS_weight = create_body_bone_weights(lbs_name)

        self.Bones = None
        self.Bones_Jp = None
        self.Bones_lx = None
        self.Bones_ly = None
        self.Bones_lz = None
        self.Bones_ls = None
        if bones_name is not None:
            self.Bones = BoneStructure()
            self.Bones.load_bone_from_fname(bones_name, if_correct_tail=True)
            self.Bones.create_local_coordArray()

        self.vertices = None
        self.faces = None
        if obj_name is not None:
            self.vertices = readObj_vert_feats(obj_name, flag='v')
            self.faces = readObj_faces(obj_name)

        self.mesh_h = None
        self.KDTree = None

    def lbs_to_tensor(self, device):
        self.LBS_weight = float_numpy_to_tensor(self.LBS_weight, device)

    def bones_to_tensor(self, device):
        self.Bones_Jp, self.Bones_lx, self.Bones_ly, self.Bones_lz, self.Bones_ls \
            = self.Bones.get_all_bone_info_tensor_device(device)

    def get_bones_information(self, device):
        if self.Bones_Jp is None:
            self.bones_to_tensor(device)
        return self.Bones_Jp, self.Bones_lx, self.Bones_ly, self.Bones_lz, self.Bones_ls

    def verts_project_to_abk(self, verts: torch.Tensor):
        if self.Bones_Jp is None:
            self.bones_to_tensor(verts.device)
        rel_vec, rel_len, proj_abk = Relative_Vector_Projection(pnts=verts, joints=self.Bones_Jp,
                                                                bones_s=self.Bones_ls, bones_x=self.Bones_lx,
                                                                bones_y=self.Bones_ly, bones_z=self.Bones_lz)
        return (rel_vec, rel_len), proj_abk

    def abk_project_to_verts(self, abk: torch.Tensor, weight: torch.Tensor):
        if self.Bones_Jp is None:
            self.bones_to_tensor(abk.device)

        inv_pos = Inverse_Relative_Projection(proj_abk=abk, joints=self.Bones_Jp, bones_s=self.Bones_ls,
                                              bones_x=self.Bones_lx, bones_y=self.Bones_ly, bones_z=self.Bones_lz,
                                              bones_w=weight)
        return inv_pos

    def compute_verts_sign_distance(self, query_verts: np.ndarray):
        if self.mesh_h is None:
            self.mesh_h = MeshHandling(self.vertices, self.faces)

        sdf = self.mesh_h.query_sdf(query_verts)
        return sdf

    def compute_body_hits_casting_ray_from_bones(self, query_abk: torch.Tensor, query_lbs_w: torch.Tensor,
                                                 body_lbs_w: np.ndarray, bone_id: torch.Tensor,
                                                 opposite_thresh=0.1, wei_align_thresh=0.2):
        inter_ray_ori, inter_ray_dir = bone_related_cast_ray(abk=query_abk,  # [N, J, 3]
                                                             pk_boneID=bone_id,  # [N, 1]
                                                             joints=self.Bones_Jp,  # [J, 3]
                                                             bx_coor=self.Bones_lx,
                                                             by_coor=self.Bones_ly,
                                                             bz_coor=self.Bones_lz,  # [J, 3]
                                                             blen=self.Bones_ls)
        inter_ray_ori = float_tensor_to_numpy(inter_ray_ori)
        inter_ray_dir = float_tensor_to_numpy(inter_ray_dir)

        if self.mesh_h is None:
            self.mesh_h = MeshHandling(self.vertices, self.faces)

        cc_ray_split, cc_hit_pos, cc_hit_norm, cc_hit_primIDs, cc_hit_primUVs = \
            self.mesh_h.ray_intersection_list(ray_orig=inter_ray_ori, ray_dir=inter_ray_dir)


        #np_lbs_w = float_tensor_to_numpy(body_lbs_w)

        cc_hit_weights = np_feats_interpolation(cc_hit_primIDs, cc_hit_primUVs,
                                                self.faces, body_lbs_w)

        sel_idx, sel_pos, sel_nrm, sel_fID, sel_fUV, sel_t = \
            filter_ray_hits_withweight(cc_ray_split, cc_hit_pos, cc_hit_norm,
                                       cc_hit_primIDs, cc_hit_primUVs, cc_hit_weights,
                                       inter_ray_ori, inter_ray_dir, float_tensor_to_numpy(query_lbs_w),
                                       opposite_thresh=opposite_thresh, wei_align_thresh=wei_align_thresh)

        valid_ray_ids = np.where(sel_idx != -1)[0]
        hit_verts = sel_pos[valid_ray_ids, :]
        hit_norms = sel_nrm[valid_ray_ids, :]
        hit_faces = self.faces[sel_fID[valid_ray_ids], :]

        return valid_ray_ids, hit_verts, hit_norms, hit_faces

    def query_nn(self, query_verts:torch.Tensor):
        if self.mesh_h is None:
            self.mesh_h = MeshHandling(self.vertices, self.faces)
        nn_pos, _, nn_norms =self.mesh_h.querry_nearest_points(float_tensor_to_numpy(query_verts.detach()))
        nn_pos = float_numpy_to_tensor(nn_pos, query_verts.device)
        nn_norms = float_numpy_to_tensor(nn_norms, query_verts.device)
        return nn_pos, nn_norms

    def query_pontential_collision(self, query_verts: torch.Tensor):
        if self.KDTree is None:
            self.KDTree = StaticKDTree(float_numpy_to_tensor(self.vertices, query_verts.device),
                                       int_numpy_to_tensor(self.faces, query_verts.device), query_verts.device)
        hits_faces, hits_pos, hits_norms, hits_area = self.KDTree.kneighbours(query_verts)
        return hits_faces, hits_pos, hits_norms, hits_area

    def bone_related_search_nearest_body_verts(self, query_verts: torch.Tensor, query_lbs_weights: torch.Tensor,
                                               body_lbs_weight: np.ndarray,
                                               norm_opposit_thr=0.1, lbs_wei_align_thr=0.1, if_area_weight=True):
        ''' Step_1: create intersection rays '''
        _, abk = self.verts_project_to_abk(query_verts) # project to bones
        bone_nn_id = select_most_related_bones_by_abkw(abk=abk, w=query_lbs_weights) # get the nearest bone

        valid_abk_ids, hit_body_verts, hit_body_norms, hit_body_faces = \
            self.compute_body_hits_casting_ray_from_bones(query_abk=abk, query_lbs_w=query_lbs_weights,
                                                          body_lbs_w = body_lbs_weight,
                                                          bone_id=bone_nn_id,
                                                          opposite_thresh=norm_opposit_thr,
                                                          wei_align_thresh=lbs_wei_align_thr)

        # debug
        # from data_IO import save_segment_obj
        # save_segment_obj('./test/tar_bone.obj', self.body_info.Bones.headArray, self.body_info.Bones.tailArray)
        # save_segment_obj('./test/b_cc.obj', query_verts[valid_abk_ids, :].cpu().numpy(), hit_body_verts)
        # exit(1)

        hit_body_verts = float_numpy_to_tensor(hit_body_verts, query_verts.device)
        hit_body_norms = float_numpy_to_tensor(hit_body_norms, query_verts.device)

        hit_face_wei = None
        if if_area_weight:
            body_verts = float_numpy_to_tensor(self.vertices, query_verts.device)
            area = compute_mesh_areas(verts=body_verts, faces=hit_body_faces)
            hit_face_wei = area / area.sum().clamp_min(1e-12)

        return valid_abk_ids, hit_body_verts, hit_body_norms, hit_body_faces, hit_face_wei



def catch_potential_self_intersection(verts, faces, fcenters, fnormals, eps=1e-4):
    mhandeling = MeshHandling(vertarray=verts, facearray=faces)
    # from data_IO import save_segment_obj
    # save_segment_obj('./test/normals.obj', seg_head=verts, seg_tail=verts + 0.1 * vert_normals)
    # exit(1)
    # +dir
    pf0, pf1 = mhandeling.query_hits(fcenters + eps * fnormals, fnormals)
    #print(pv_id.shape, pf_id.shape)
    # -dir
    nf0, nf1 = mhandeling.query_hits(fcenters - eps * fnormals, -fnormals)
    f0 = np.concatenate((pf0, nf0), axis=0)
    f1 = np.concatenate((pf1, nf1), axis=0)

    # print(vid.shape, fid.shape)
    # exit(1)
    return np.ascontiguousarray(f0), np.ascontiguousarray(f1)



class GarmentMesh(object):
    def __init__(self, body_info, *, obj_name=None, if_norm=False):
        self.vertices = None
        self.faces = None
        self.normals = None
        if obj_name is not None:
            self.vertices = readObj_vert_feats(obj_name, flag='v')
            self.faces = readObj_faces(obj_name)
            if if_norm:
                self.normals = readObj_vert_feats(obj_name, flag='vn')

        self.body_info = body_info
        self.LBS_weight = None

        self.graph_topo = None
        self.Laplacian_Opt = None

    def update_mesh_geo_info(self, verts: np.ndarray, faces: np.ndarray):
        self.vertices = verts
        self.faces = faces

    def create_garment_bone_weights(self, device=None):
        if self.body_info is None:
            print("class GarmentMesh >> create_garment_bone_weights >> Error: no body information!!")
            exit(1)
        if self.body_info.LBS_weight is None:
            print("class GarmentMesh >> create_garment_bone_weights >>Error: no body LBS!!")
            exit(1)

        garm_verts = self.vertices
        body_verts = self.body_info.vertices
        body_weights = self.body_info.LBS_weight

        g_bneigbors = kdTree_nearest_neighbor(garm_verts, body_verts)
        self.LBS_weight = body_weights[g_bneigbors, :]
        if device is not None:
            self.LBS_weight = float_numpy_to_tensor(self.LBS_weight, device)

    def update_LBS_weight(self, weights):
        assert weights.shape[0] == self.vertices.shape[0]
        self.LBS_weight = weights

    def update_graph_topology(self, device=None):
        if self.graph_topo is not None:
            del self.graph_topo
        self.graph_topo = MeshTopology(self.vertices.shape[0], self.faces)
        if device is not None:
            self.graph_topo.to_tensor(device)

    def get_faces(self):
        return self.graph_topo.faces

    def compute_lbs_feats_blending(self, feats: torch.Tensor):
        assert self.LBS_weight is not None and feats.dim() == 2
        fb = torch.einsum('nj, jd->nd', self.LBS_weight, feats)
        return fb

    def update_laplacian_operator(self, device):
        if self.Laplacian_Opt is not None:
            del self.Laplacian_Opt

        self.Laplacian_Opt = LaplacianOperator(float_numpy_to_tensor(self.vertices, device),
                                               int_numpy_to_tensor(self.faces, device))

    def vertices_to_tensor(self, device):
        self.vertices = float_numpy_to_tensor(self.vertices, device)

    def compute_vertices_projection_to_bones(self, verts):
        _, verts_proj_abk = self.body_info.verts_project_to_abk(verts=verts)
        return verts_proj_abk

    def compute_abk_backprojection_to_verts(self, abk, weight):
        inv_verts = self.body_info.abk_project_to_verts(abk=abk, weight=weight)
        return inv_verts

    def compute_norm_laplacian_delta(self, verts, lap_opt: LaplacianOperator, if_norm:bool=True):
        detla = lap_opt.laplacian_matvec(verts, if_norm=if_norm)
        return detla

    def compute_curve_adj_cos(self, verts, graph_top: MeshTopology, if_need_wei: bool):
        c_adj_cos, c_adj_wei, num_c = graph_top.compute_curve_adj_cos(verts=verts, if_need_wei=if_need_wei)
        return c_adj_cos, c_adj_wei, num_c

    def compute_dihedral_angle(self, verts, graph_top:MeshTopology, if_need_wei: bool):
        angle, weight = graph_top.compute_dihedral_angle(verts=verts, if_need_weight=if_need_wei)
        return angle, weight
































