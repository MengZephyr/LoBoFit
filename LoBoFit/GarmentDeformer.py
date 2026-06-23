import torch
import torch.nn as nn
from mesh_Geo import LaplacianOperator, MeshTopology
from Loss_def import compute_feats_loss, compute_chain_feats_loss, compute_reg_feats
from bone_projection import Inverse_Relative_Projection, Relative_Vector_Projection
from mesh_utils import compute_face_normals, triangle_centroids, compute_face_centers, compute_mesh_areas
from bone_projection import BonesID_dict

def filter_hits_by_normal_lbsweights(normals, base_normals, norm_thred=0.2, *,
                                     lbsweights=None, base_lbsweights=None, lbswei_thred=0.1):
    t = torch.ones(normals.shape[0]).to(normals.device)
    cond = (t>0)

    n_cos = (normals*base_normals).sum(dim=-1)
    cond = cond & (n_cos > norm_thred)

    if lbsweights is not None and base_lbsweights is not None:
        w_cos = (lbsweights*base_lbsweights).sum(dim=-1)
        cond = cond & (w_cos > lbswei_thred)

    valid_idx = torch.where(cond)[0]

    return valid_idx

def abk_upbody_segmentation(proj_abk):
    hips_abk = proj_abk[:, BonesID_dict["Hips"], :]  #[N, 3]
    hips_up = (hips_abk[:, -1] > 0)   # [N]

    neck_abk = proj_abk[:, BonesID_dict["Head"], :] #[N, 3]
    head_down = (neck_abk[:, -1] < 0)

    leftarm_abk = proj_abk[:, BonesID_dict["LeftArm"], :]
    leftarm_right = (leftarm_abk[:, -1] < 0)

    rightarm_abk = proj_abk[:, BonesID_dict["RightArm"], :]
    rightarm_left = (rightarm_abk[:, -1] < 0)

    cond = (hips_up & head_down & leftarm_right & rightarm_left)
    valid = torch.where(cond)[0]
    return cond, valid

def abk_waist_segmentation(proj_abk):
    hips_abk = proj_abk[:, BonesID_dict["Hips"], :]  # [N, 3]
    hips_up = (hips_abk[:, -1] > 0)  # [N]

    spine_abk = proj_abk[:, BonesID_dict["Spine"], :]
    spine_done = (spine_abk[:, -1] < 0)
    cond = (hips_up & spine_done)
    valid = torch.where(cond)[0]
    return cond, valid


def abk_fit_focus(proj_abk, fit_mode):
    '''
    :param proj_abk: torch.Tensor
    :param fit_mode: # [Options]: 'waist', 'torso', 'torso_waist', 'all'
    :return: focus_verts_ids
    '''
    N = proj_abk.shape[0]
    if fit_mode == 'waist':
        body_mask, garm_fit_part = abk_waist_segmentation(proj_abk)  # shared across frames
        # save_obj('./test/w_seg.obj', vertices=src_cano_garment.vertices[int_tensor_to_numpy(garm_fit_part), :])
        # exit(1)
    elif fit_mode == 'torso':
        body_mask, garm_fit_part = abk_upbody_segmentation(proj_abk)
        # save_obj('./test/u_seg.obj', vertices=src_cano_garment.vertices[int_tensor_to_numpy(garm_fit_part), :])
        # exit(1)
    elif fit_mode == 'torso_waist':
        u_mask, _ = abk_upbody_segmentation(proj_abk)
        w_mask, _ = abk_waist_segmentation(proj_abk)
        mask = u_mask | w_mask
        garm_fit_part = torch.where(mask)[0]
        # save_obj('./test/uw_seg.obj', vertices=src_cano_garment.vertices[int_tensor_to_numpy(garm_fit_part), :])
        # exit(1)
    else:
        garm_fit_part = \
            torch.tensor([i for i in range(N)], dtype=torch.int32).to(proj_abk.device)

    return garm_fit_part


class GarmentDeformer(nn.Module):
    def __init__(self, src_abk, ini_weight, joints, joints_lx, joints_ly, joints_lz, joints_ls, w_scale=0.1):
        super().__init__()
        # trainable parameters
        self.delta_abk = nn.Parameter(torch.zeros_like(src_abk))
        self.delta_weight = nn.Parameter(torch.zeros_like(ini_weight))
        self.w_scale = w_scale

        # persistent coefficients
        self.register_buffer("src_abk", src_abk)
        self.register_buffer("ini_weight", ini_weight)
        self.register_buffer("joints", joints)
        self.register_buffer("joints_lx", joints_lx)
        self.register_buffer("joints_ly", joints_ly)
        self.register_buffer("joints_lz", joints_lz)
        self.register_buffer("joints_ls", joints_ls)

    def bend_metric(self, verts, mesh_topo:MeshTopology, gt_angles, weight=None):
        # face_n = compute_face_normals(verts, faces)
        # adj_angles = dihedral_angle_adjacent_faces(normals=face_n, adjacency=adj_faces)
        adj_angles, _ = mesh_topo.compute_dihedral_angle(verts, if_need_weight=False)
        loss = compute_feats_loss(adj_angles, gt_angles, weight)
        return loss

    def laplacian_metric(self, verts, lap_opt:LaplacianOperator, gt_delta, weight=None):
        delta = lap_opt.laplacian_matvec(verts, if_norm=True)
        if weight is not None:
            sw = weight.sum().clamp_min(1e-8)
            w = weight/sw
            loss = compute_feats_loss(delta, gt_delta, w)
        else:
            loss = compute_feats_loss(delta, gt_delta)
        return loss

    def tangent_point_energy(self, verts, mesh_topo:MeshTopology, p=4, t_thickness=1e-3, eps=1e-6, tau=1e-6, weight=1.0):
        C, N, A = triangle_centroids(verts, mesh_topo.faces) # (M,3), (M,3), (M,)
        #A_tot = A.sum().clamp_min(1e-8)
        w = A
        #M = C.shape[0]
        adj_faces = mesh_topo.face_adjacency

        Ci = C[:, None, :]  # (M,1,3)
        Cj = C[None, :, :]  # (1,M,3)
        r = Cj - Ci  # (M,M,3)
        r2 = (r * r).sum(-1) + eps ** 2  # (M,M)

        dot_i = (r * N[:, None, :]).sum(-1)  # (M,M)
        dot_j = (r * N[None, :, :]).sum(-1)  # (M,M)
        sabs_i = torch.sqrt(dot_i * dot_i + (tau + t_thickness) ** 2)
        sabs_j = torch.sqrt(dot_j * dot_j + (tau + t_thickness) ** 2)

        term_i = (2.0 * sabs_i / r2).pow(p / 2)
        term_j = (2.0 * sabs_j / r2).pow(p / 2)
        phi = 0.5 * (term_i + term_j)  # (M,M)

        phi[adj_faces[:, 0], adj_faces[:, 1]] = 0.
        phi[adj_faces[:, 1], adj_faces[:, 0]] = 0.

        W = w[:, None] * w[None, :]  # (M,M)
        E = (phi * W).sum() * weight
        return E

    def curve_metric(self, verts, mesh_topo: MeshTopology, gt_adj_cos, weight=None):
        adj_cos, _, num_c = mesh_topo.compute_curve_adj_cos(verts, if_need_wei=False)
        loss = compute_chain_feats_loss(c_feats=adj_cos, gt_c_feats=gt_adj_cos, num_c=num_c, c_weights=weight)
        return loss

    def reg_w(self):
        #loss_k = compute_reg_feats(self.delta_abk[:, :, -1])
        loss_w = compute_reg_feats(self.w_scale * torch.tanh(self.delta_weight), method='sum')
        return loss_w

    def reg_k(self):
        loss_k = compute_reg_feats(self.delta_abk[:, :, -1], method='sum')
        return loss_k

    def strict_reg_k(self, verts):
        _v, _l, proj_abk = Relative_Vector_Projection(pnts=verts, joints=self.joints, bones_s=self.joints_ls,
                                                      bones_x=self.joints_lx,
                                                      bones_y=self.joints_ly, bones_z=self.joints_lz)
        delta_k = proj_abk - self.src_abk
        loss_k = compute_reg_feats(delta_k[:, :, -1], method='mean')
        return loss_k

    def normal_metric(self, verts, mesh_topo: MeshTopology, gt_normals, wei=None):
        face_n = compute_face_normals(verts, mesh_topo.faces)
        loss = compute_feats_loss(face_n, gt_normals, wei)
        return loss

    def selfnn_metric(self, verts, mesh_topo: MeshTopology, f0, f1, thred=0.003):
        f0_vid = mesh_topo.faces[f0, :]
        f1_vid = mesh_topo.faces[f1, :]
        c0 = compute_face_centers(verts, f0_vid)
        c1 = compute_face_centers(verts, f1_vid)
        w = compute_mesh_areas(verts, f0_vid).detach()
        # c0 = face_c[f0, :]
        # c1 = face_c[f1, :]
        # w = face_area[f0, :]
        diff = c0 - c1
        dist2 = (diff * diff).sum(dim=-1)
        dist = torch.sqrt(dist2)
        ll = torch.relu(thred-dist)
        loss = w * ll * ll
        return loss.sum()


    def normal_flip_metric(self, verts, mesh_topo: MeshTopology, gt_normals, wei=None):
        face_n = compute_face_normals(verts, mesh_topo.faces)
        a_sign = (face_n * gt_normals).sum(dim=-1)
        a_sign = torch.relu(-a_sign)
        if wei is not None:
            loss = (wei * a_sign * a_sign).sum()
        else:
            loss = (a_sign * a_sign).sum()
        return loss


    def hips_laplacian_difference(self, verts, hipID, lap_opt:LaplacianOperator):
        _v, _l, proj_abk = Relative_Vector_Projection(pnts=verts, joints=self.joints, bones_s=self.joints_ls,
                                                      bones_x=self.joints_lx,
                                                      bones_y=self.joints_ly, bones_z=self.joints_lz)
        hips_abk = proj_abk[:, hipID, :]
        gt_hips_abk = self.src_abk[:, hipID, :]
        h_delta = lap_opt.laplacian_matvec(hips_abk, if_norm=True)
        gt_h_delta = lap_opt.laplacian_matvec(gt_hips_abk, if_norm=True)
        loss = compute_feats_loss(h_delta, gt_h_delta)
        return loss


    def forward(self, if_opt_w:bool = True):
        new_abk = self.src_abk + self.delta_abk
        if if_opt_w:
            new_w = self.ini_weight + self.w_scale * torch.tanh(self.delta_weight) #[N, J]
            #new_w = torch.relu(new_w)
            #new_w = torch.softmax(new_w, dim=-1)
            sw = new_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            new_w = new_w/sw
        else:
            new_w = self.ini_weight
        new_inv = Inverse_Relative_Projection(proj_abk=new_abk, joints=self.joints,
                                              bones_s=self.joints_ls,
                                              bones_x=self.joints_lx, bones_y=self.joints_ly,
                                              bones_z=self.joints_lz, bones_w=new_w)
        return new_inv

    def get_new_abk(self):
        return self.src_abk + self.delta_abk

    def get_new_w(self, if_opt_w:bool = True):
        if if_opt_w:
            new_w = self.ini_weight + self.w_scale * torch.tanh(self.delta_weight)  # [N, J]
            #new_w = torch.relu(new_w)
            sw = new_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            new_w = new_w / sw
        else:
            new_w = self.ini_weight

        return new_w


class SemanticWGarmentDeformer(GarmentDeformer):
    def __init__(self, src_abk, ini_weight, joints, joints_lx, joints_ly, joints_lz, joints_ls,
                 train_semantic_list, w_scale=0.1):
        super().__init__(src_abk, ini_weight, joints, joints_lx, joints_ly, joints_lz, joints_ls, w_scale)
        self.w_train_id_list = None
        if train_semantic_list is not None:
            self.set_semantic_keep(train_semantic_list=train_semantic_list)
            N,_ = ini_weight.shape
            tJ = len(train_semantic_list)
            self.delta_weight = nn.Parameter(torch.zeros(N, tJ, device=ini_weight.device))

    def set_semantic_keep(self, train_semantic_list):
        self.w_train_id_list = []
        for txt in train_semantic_list:
            self.w_train_id_list.append(BonesID_dict[txt])
        self.w_train_id_list = torch.tensor(self.w_train_id_list).type(torch.int).to(self.ini_weight.device)

    def forward(self, if_opt_w:bool = True):
        if self.w_train_id_list is None or if_opt_w == False:
            super().forward(if_opt_w)
        else:
            new_abk = self.src_abk + self.delta_abk
            base_w = self.ini_weight  # [N, J]
            new_w = base_w.clone()
            delta = self.w_scale * torch.tanh(self.delta_weight)  # in [-0.1, 0.1]
            train_w = base_w[:, self.w_train_id_list] + delta  # [N, tJ]
            #train_w = train_w.clamp_min(1e-8)
            new_w[:, self.w_train_id_list] = train_w
            sum_w = new_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            new_w = new_w / sum_w

            # new_w = torch.zeros_like(self.ini_weight) + self.ini_weight  # [N, J]
            # train_w = self.ini_weight[:, self.w_train_id_list] + 0.1 * torch.tanh(self.delta_weight)
            # train_sw = train_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            # train_w = train_w / train_sw
            # new_w[:, self.w_train_id_list] = train_w

            new_inv = Inverse_Relative_Projection(proj_abk=new_abk, joints=self.joints,
                                                  bones_s=self.joints_ls,
                                                  bones_x=self.joints_lx, bones_y=self.joints_ly,
                                                  bones_z=self.joints_lz, bones_w=new_w)
            return new_inv


    def get_new_w(self, if_opt_w:bool = True):
        if self.w_train_id_list is None or if_opt_w == False:
            super().get_new_w(if_opt_w)
        else:
            base_w = self.ini_weight.detach()  # [N, J]
            new_w = base_w.clone()
            delta = self.w_scale * torch.tanh(self.delta_weight)  # in [-0.1, 0.1]
            train_w = base_w[:, self.w_train_id_list] + delta  # [N, tJ]
            train_w = train_w.clamp_min(1e-8)
            new_w[:, self.w_train_id_list] = train_w
            sum_w = new_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            new_w = new_w / sum_w
            return new_w





