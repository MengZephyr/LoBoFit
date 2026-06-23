from GarmentDeformer import *
from object_Handle import BodyMesh, GarmentMesh
from data_IO import load_Geo_Different_Resolution_Sampling, readObj_faces, readObj_vert_feats, save_obj, save_segment_obj, \
    float_numpy_to_tensor, int_numpy_to_tensor, float_tensor_to_numpy, savePly
from mesh_utils import sample_from_faces, compute_mesh_areas
from Loss_def import compute_loss_collision
import numpy as np
from bone_projection import KeyBonesNames
from Color_Visualize import Bone_Colors


def get_color_list():
    color_list = []
    for bnames in KeyBonesNames:
        color_list.append(Bone_Colors[bnames])
    color_list = np.array(color_list)
    return color_list


BoneColorList = get_color_list()
W_SCALE = 0.1
#W_SCALE = 0.3  # to enlarge optimal resolution field for refitting TShirt on mouse
IF_KEEP_K = False


def optimize_garment_deformer(GDeformer: GarmentDeformer, DeformerOpt, ini_lbs,
                              new_inv: torch.Tensor, src_verts: torch.Tensor,
                              mesh_topo: MeshTopology, lap_opt: LaplacianOperator, fit_parts: torch.Tensor,
                              src_body: BodyMesh, tar_body: BodyMesh,
                              iterations: int, iter_update_hits: int, update_factor: int=3, if_opt_w:bool=True,
                              *, png_name='lr_'):

    if iter_update_hits > iterations or iter_update_hits < 0:
        iter_update_hits = 0

    valid_abk_ids, hit_body_verts, hit_body_norms, hit_body_faces, hit_face_wei = None, None, None, None, None
    if iter_update_hits > 0:
        valid_abk_ids, hit_body_verts, hit_body_norms, hit_body_faces, hit_face_wei = \
            tar_body.bone_related_search_nearest_body_verts(query_verts=new_inv,
                                                            query_lbs_weights=ini_lbs,
                                                            body_lbs_weight=tar_body.LBS_weight,
                                                            norm_opposit_thr=0.1, lbs_wei_align_thr=0.1,
                                                            if_area_weight=True)
        save_segment_obj('../rst/ini_deform_map.obj', seg_head=hit_body_verts.detach().cpu().numpy(),
                         seg_tail=new_inv[valid_abk_ids, :].detach().cpu().numpy())


        valid_abk_ids = int_numpy_to_tensor(valid_abk_ids, device)

    tar_frame_garments = GarmentMesh(body_info=tar_body)

    gt_adj_cos, adj_wei, num_c = tar_frame_garments.compute_curve_adj_cos(verts=src_verts,
                                                                          graph_top=mesh_topo,
                                                                          if_need_wei=True)
    gt_bend_angle, bend_wei = tar_frame_garments.compute_dihedral_angle(verts=src_verts,
                                                                        graph_top=mesh_topo,
                                                                        if_need_wei=True)

    gt_delta = tar_frame_garments.compute_norm_laplacian_delta(verts=src_verts,
                                                               lap_opt=lap_opt, if_norm=True)

    gt_normals = compute_face_normals(src_verts, mesh_topo.faces)
    gt_area = compute_mesh_areas(src_verts, mesh_topo.faces)
    gt_area_w = gt_area / gt_area.sum().clamp_min(1e-8)

    gt_body_fit_sdf = \
        src_body.compute_verts_sign_distance(
            float_tensor_to_numpy(src_verts[fit_parts, :]))
    gt_body_fit_sdf = float_numpy_to_tensor(gt_body_fit_sdf, device)
    sel = torch.where(gt_body_fit_sdf < 0.02)[0]
    gt_sdf_focus = gt_body_fit_sdf[sel]
    fit_focus = fit_parts[sel]

    if iter_update_hits > 0:
        bnn_m = torch.isin(valid_abk_ids, fit_focus)
        bnn_fit_focus = torch.nonzero(bnn_m, as_tuple=False).squeeze(1).to(torch.int32)
        bnn_rm = torch.isin(fit_focus, valid_abk_ids)
        gt_fit_id = torch.nonzero(bnn_rm, as_tuple=False).squeeze(1).to(torch.int32)
        gt_bnn_sdf_focus = gt_sdf_focus[gt_fit_id]
    else:
        bnn_fit_focus = None
        gt_bnn_sdf_focus = None

    dhits_fidx, dhits_pnts, dhits_norms, dhits_weights = None, None, None, None
    dhits_update = False
    garment_faces = mesh_topo.faces

    num_cc = (iterations - iter_update_hits) // update_factor
    for itt in range(iterations):
        if itt >= iter_update_hits:
            if (itt - iter_update_hits) % num_cc == 0:
                dhits_update = False

        DeformerOpt.zero_grad()
        new_inv = GDeformer(if_opt_w=if_opt_w)

        # similarity constraints
        loss_bend = GDeformer.bend_metric(new_inv, mesh_topo, gt_bend_angle, bend_wei)
        loss_laplacian = GDeformer.laplacian_metric(new_inv, lap_opt, gt_delta)
        loss_curve = GDeformer.curve_metric(new_inv, mesh_topo, gt_adj_cos, adj_wei)
        loss_normal = GDeformer.normal_metric(new_inv, mesh_topo, gt_normals, gt_area_w)
        #loss_selfnn = GDeformer.selfnn_metric(new_inv, mesh_topo, selfnn_f0, selfnn_f1, thred=0.001)
        #loss_tpe = GDeformer.tangent_point_energy(new_inv, mesh_topo)

        # regularization
        loss_k = GDeformer.reg_k()
        loss_w = GDeformer.reg_w()

        # collision penalty
        if itt < iter_update_hits:
            loss_collision, sdf = compute_loss_collision(base_verts=hit_body_verts, base_norms=hit_body_norms,
                                                         verts=new_inv[valid_abk_ids, :], d_thre=0.008, wei=hit_face_wei)
            fit_sdf = sdf[bnn_fit_focus]
            loss_fitness = compute_feats_loss(fit_sdf, gt_bnn_sdf_focus, hit_face_wei[bnn_fit_focus])
        else:
            if dhits_update is not True:
                dhits_fidx, dhits_pnts, dhits_norms, dhits_weights = tar_body.query_pontential_collision(new_inv)
                dhits_update = True
            loss_collision, sdf = compute_loss_collision(base_verts=dhits_pnts,
                                                         base_norms=dhits_norms,
                                                         verts=new_inv,
                                                         d_thre=0.008, wei=dhits_weights)
            fit_sdf = sdf[fit_focus]
            loss_fitness = compute_feats_loss(fit_sdf, gt_sdf_focus, dhits_weights[fit_focus])

        loss = 10 * loss_collision + 1e2 * loss_fitness + \
               1. * loss_bend + 1 * loss_curve + 0.5 * loss_laplacian + \
               1 * loss_k + 1e-2 * loss_w

        if itt % 500 == 0 or itt == iterations - 1:
            print(itt, loss.item(), loss_bend.item(), loss_curve.item(), loss_laplacian.item(),
                  loss_collision.item(), loss_k.item(), loss_normal.item(), loss_w.item(),
                  loss_fitness.item())

        # if itt % 10 == 0:
        #     print(itt, loss.item(), loss_bend.item(), loss_curve.item(), loss_laplacian.item(),
        #           loss_collision.item(), loss_k.item(), loss_normal.item(), loss_w.item(),
        #           loss_fitness.item())
        #     new_w = GDeformer.get_new_w(if_opt_w=if_opt_w)
        #     new_wei_color = color_vertex_by_bone_weights(new_w.detach().cpu().numpy(), BoneColorList)
        #     savePly('../test_iter/'+ tar_avatar + '/' + str(frameID-1) + '_10/' + png_name + str(itt) + '.ply', new_inv.detach().cpu().numpy(),
        #             colors=new_wei_color * 255,
        #             faces=garment_faces)
        #     _, temp_abk = tar_body.verts_project_to_abk(new_inv)
        #     color_projections(abk=temp_abk.detach().cpu().numpy(),
        #                       weights=new_w.detach().cpu().numpy(), bcolors=BoneColorList,
        #                       faces=garment_faces,
        #                       saveprefix='../test_iter/'+tar_avatar + '/' + str(frameID-1) + '_10/proj_tar/' + png_name + str(itt) + '_',
        #                       bone_id_list=[4, 8, 17])

        loss.backward()
        DeformerOpt.step()

    new_inv = GDeformer(if_opt_w=if_opt_w).detach()
    new_abk = GDeformer.get_new_abk().detach()
    new_w = GDeformer.get_new_w(if_opt_w=if_opt_w).detach()

    '''resolve intersection with displacement'''
    dhits_pnts, dhits_norms = tar_body.query_nn(new_inv)
    diff = new_inv - dhits_pnts
    dist = (diff * diff).sum(dim=-1)
    dist = torch.sqrt(dist)
    dist = torch.relu(0.008 - dist)
    sel_id = torch.where(dist>0)[0]
    proj = diff * dhits_norms
    sdf = proj.sum(dim=-1)  # [N]
    cc = torch.relu(0.003 - sdf)
    displace = dhits_norms * cc.unsqueeze(dim=-1)
    rcc_inv = new_inv.detach().clone()
    rcc_inv[sel_id, :] = rcc_inv[sel_id, :] + displace[sel_id, :]

    return rcc_inv, new_inv, new_abk, new_w


def color_projections(abk, weights, bcolors, faces, saveprefix='./test/pb_', bone_id_list=None, min_wei=0.1):
    num_N, num_B, _ = abk.shape
    if bone_id_list is None:
        bone_id_list = [i for i in range(num_B)]
    for i in bone_id_list:
        pb_verts = abk[:, i, :]
        bc = bcolors[i, :]
        pb_colors = np.ones((num_N, 3))
        pb_colors[:, 0] = bc[0]*255
        pb_colors[:, 1] = bc[1]*255
        pb_colors[:, 2] = bc[2]*255

        pb_weight = weights[:, i]  # [N,]
        #min_wei = np.min(pb_weight)
        max_wei = np.max(pb_weight)
        pnt_wei = (pb_weight - min_wei) / (max_wei - min_wei)
        wei_color = np.ones((num_N, 1))
        # wei_color[:, 0] = wei_color[:, 0] * pnt_wei * 255
        # wei_color[:, 1] = wei_color[:, 1] * pnt_wei * 255
        # wei_color[:, 2] = wei_color[:, 2] * pnt_wei * 255
        # savePly(saveprefix + str(i) + '_' + KeyBonesNames[i] + '_wei.ply',
        #         verts=pb_verts, colors=wei_color, faces=faces)
        wei_color[:, 0] = wei_color[:, 0] * pnt_wei * 255
        savePly(saveprefix + str(i) + '_' + KeyBonesNames[i] + '_wei.ply',
                verts=pb_verts, colors=pb_colors, alpha=wei_color, faces=faces)


def color_vertex_by_bone_weights(bone_weights, bone_colors):
    v_color = np.einsum('nk, kd->nd', bone_weights, bone_colors)
    return v_color


if __name__ == '__main__':
    import os
    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)
    device = torch.device("cuda" if USE_CUDA else "cpu")
    #device = torch.device("cpu")

    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    prefix = '../data/'
    pose_name = 'WHip_Hop'
    dress_name = 'TShirt'
    pd = 10
    pd_name = '/PD' + str(pd) + '_G/'
    if_opt_w = True

    fit_mode = 'torso'  # Options: 'waist', 'torso', 'torso_waist', 'all'

    src_avatar = 'Manneq'
    src_prefix = prefix + src_avatar + '/' + pose_name + '/'

    tar_avatar = 'mouse'
    tar_prefix = prefix + 'target_Avatar/' + tar_avatar + '/' + pose_name + '/'

    # set up canonical information for src body, src garment
    src_cano_body = BodyMesh(obj_name=src_prefix + 'body/bA.obj',
                             bones_name=src_prefix + 'Bones/A.txt',
                             lbs_name=src_prefix + 'body_weight.npz')

    hr_verts = readObj_vert_feats(fname=src_prefix + '/' + dress_name + pd_name + '/A_pose/A_pose.obj', flag='v')
    hr_faces = readObj_faces(fname=src_prefix + '/' + dress_name + pd_name + '/A_pose/A_pose.obj', tagID=0)


    # fine to coarse downsampling
    lr_pd = 30
    lr_pd_name = '/PD' + str(lr_pd) + '_G/'
    lr_faces = readObj_faces(fname=src_prefix + '/' + dress_name + lr_pd_name + '/A_pose/A_pose.obj')

    f2c_name = src_prefix + '/' + dress_name + '/' + str(lr_pd) + '_from_' + str(pd) + '_Sampling.txt'
    c_g2u_name = src_prefix + '/' + dress_name + lr_pd_name + '/A_pose/A_Pose_g_to_u.txt'
    f2c_fSampleFaceID, f2c_fSampleABC = \
        load_Geo_Different_Resolution_Sampling(f2c_name, c_g2u_name)
    f2c_fSampleInFVIDs = int_numpy_to_tensor(hr_faces[f2c_fSampleFaceID, :], device)
    f2c_fSampleABC = float_numpy_to_tensor(f2c_fSampleABC, device)

    # coarse to fine upsampling
    c2f_name = src_prefix + '/' + dress_name + '/' + str(pd) + '_from_' + str(lr_pd) + '_Sampling.txt'
    f_g2u_name = src_prefix + '/' + dress_name + pd_name + '/A_pose/A_Pose_g_to_u.txt'
    c2f_fSampleFaceID, c2f_fSampleABC = \
        load_Geo_Different_Resolution_Sampling(c2f_name, f_g2u_name)
    c2f_fSampleInFVIDs = int_numpy_to_tensor(lr_faces[c2f_fSampleFaceID, :], device)
    c2f_fSampleABC = float_numpy_to_tensor(c2f_fSampleABC, device)

    lr_verts = sample_from_faces(float_numpy_to_tensor(hr_verts, device), f2c_fSampleInFVIDs, f2c_fSampleABC)
    # save_obj('./test/lr_a.obj', vertices=lr_verts.detach().cpu().numpy(), faces=lr_faces)
    # exit(1)

    hr_mesh_topology = MeshTopology(hr_verts.shape[0], hr_faces)
    lr_mesh_topology = MeshTopology(lr_verts.shape[0], lr_faces)

    src_cano_garment = GarmentMesh(body_info=src_cano_body)
    src_cano_garment.update_mesh_geo_info(float_tensor_to_numpy(lr_verts), lr_faces)
    src_cano_garment.create_garment_bone_weights(device=device)

    ini_garment_weight = src_cano_garment.LBS_weight  # shared across frames

    tar_cano_body = BodyMesh(lbs_name=tar_prefix + 'body_weight.npz')
    tar_body_lbs_w = tar_cano_body.LBS_weight

    # get lr and hr fit part
    _lrc, lr_abk = src_cano_body.verts_project_to_abk(lr_verts)
    lr_fit_parts = abk_fit_focus(lr_abk, fit_mode=fit_mode)
    _hrc, hr_abk = src_cano_body.verts_project_to_abk(float_numpy_to_tensor(hr_verts, device))
    hr_fit_parts = abk_fit_focus(hr_abk, fit_mode=fit_mode)
    # from data_IO import save_index
    # save_index('./test/no-fit.txt', hr_fit_parts)
    # exit(1)

    frame_list = [51]

    for frameID in frame_list:
        # note: garment ID = body ID - 1
        src_frame_body = BodyMesh(obj_name=src_prefix + 'body/b' + str(frameID).zfill(4) + '.obj',
                                  bones_name=src_prefix + '/Bones/' + str(frameID) + '.txt')

        frame_hr_verts = readObj_vert_feats(fname=src_prefix + '/' + dress_name + pd_name + str(frameID - 1) + '.obj')

        frame_hr_verts = float_numpy_to_tensor(frame_hr_verts, device)
        frame_lr_verts = sample_from_faces(frame_hr_verts, f2c_fSampleInFVIDs, f2c_fSampleABC)

        # hr_face_centers = compute_face_centers(frame_hr_verts, hr_faces)
        # hr_face_normals = compute_face_normals(frame_hr_verts, hr_faces)
        # hr_selfnn_f0, hr_selfnn_f1 = \
        #     catch_potential_self_intersection(float_tensor_to_numpy(frame_hr_verts), hr_faces,
        #                                       fcenters=float_tensor_to_numpy(hr_face_centers),
        #                                       fnormals=float_tensor_to_numpy(hr_face_normals))
        #
        # lr_face_centers = compute_face_centers(frame_lr_verts, lr_faces)
        # lr_face_normals = compute_face_normals(frame_lr_verts, lr_faces)
        # lr_selfnn_f0, lr_selfnn_f1 = \
        #     catch_potential_self_intersection(float_tensor_to_numpy(frame_lr_verts), lr_faces,
        #                                       fcenters=float_tensor_to_numpy(lr_face_centers),
        #                                       fnormals=float_tensor_to_numpy(lr_face_normals))
        # print(lr_selfnn_f0.shape, lr_selfnn_f1.shape)
        #save_obj('../test_LoBoFit/src_coarse.obj', vertices=frame_lr_verts.cpu().numpy(), faces=lr_faces)
        # save_segment_obj('./test/self_hrnn.obj', seg_head=hr_face_centers[hr_selfnn_f0, :].cpu().numpy(),
        #                  seg_tail=hr_face_centers[hr_selfnn_f1, :].cpu().numpy())



        #save_obj('../test_c/src_lr.obj', frame_lr_verts.cpu().numpy(), faces=lr_faces)

        lr_laplacian = LaplacianOperator(frame_lr_verts, int_numpy_to_tensor(lr_faces, device))
        hr_laplacian = LaplacianOperator(frame_hr_verts, int_numpy_to_tensor(hr_faces, device))

        src_frame_garment = GarmentMesh(src_frame_body)
        src_joints, src_lx, src_ly, src_lz, src_ls = src_frame_body.get_bones_information(device=device)
        save_segment_obj('../rst/src_bone.obj', seg_head=src_joints.cpu().numpy(),
                         seg_tail=(src_joints+src_lz*src_ls).cpu().numpy())
        #new_wei_color = color_vertex_by_bone_weights(ini_garment_weight.detach().cpu().numpy(), BoneColorList)

        # tar
        tar_frame_body = BodyMesh(obj_name=tar_prefix + 'body/b' + str(frameID).zfill(4) + '.obj',
                                  bones_name=tar_prefix + '/Bones/' + str(frameID) + '.txt')
        tar_frame_body.LBS_weight = tar_body_lbs_w.copy()

        '''
        low resolution level
        '''
        _, lr_src_abk = src_frame_body.verts_project_to_abk(verts=frame_lr_verts)
        lr_new_inv = tar_frame_body.abk_project_to_verts(abk=lr_src_abk, weight=ini_garment_weight)
        # debug
        save_obj('../rst/ini_g.obj', lr_new_inv.cpu().numpy(), faces=lr_faces)

        tar_joints, tar_lx, tar_ly, tar_lz, tar_ls = tar_frame_body.get_bones_information(device=device)
        save_segment_obj('../rst/' + tar_avatar + '_bone.obj', seg_head=tar_joints.cpu().numpy(),
                         seg_tail=(tar_joints+tar_lz*tar_ls).cpu().numpy())

        lr_garment_deformer = \
            GarmentDeformer(lr_src_abk, ini_garment_weight, tar_joints, tar_lx, tar_ly, tar_lz, tar_ls,
                            w_scale=W_SCALE).to(device)
        lr_garment_opt = \
            torch.optim.AdamW(params=lr_garment_deformer.parameters(), lr=0.005, betas=(0.9, 0.999),
                              amsgrad=True, fused=True)

        print(frameID, ' -> low resolution deformation starts....')
        lr_rcc_inv, lr_new_inv, lr_new_abk, lr_new_w = \
            optimize_garment_deformer(GDeformer=lr_garment_deformer, DeformerOpt=lr_garment_opt,
                                      ini_lbs=ini_garment_weight,
                                      new_inv=lr_new_inv, src_verts=frame_lr_verts,
                                      mesh_topo=lr_mesh_topology, lap_opt=lr_laplacian,
                                      fit_parts=lr_fit_parts,
                                      src_body=src_frame_body, tar_body=tar_frame_body,
                                      iterations=8000, iter_update_hits=4000, update_factor=2, if_opt_w=if_opt_w,
                                      png_name='lr_')

        save_obj('../rst/new_c.obj', lr_rcc_inv.detach().cpu().numpy(), faces=lr_faces)
        print(frameID, ' -> low resolution deformation done...')

        '''
        high reolution level
        '''
        hr_new_inv = sample_from_faces(lr_rcc_inv, c2f_fSampleInFVIDs, c2f_fSampleABC)
        save_obj('../rst/ini_f.obj', hr_new_inv.detach().cpu().numpy(), faces=hr_faces)

        hr_ini_weights = sample_from_faces(lr_new_w, c2f_fSampleInFVIDs, c2f_fSampleABC)

        _, hr_ini_abk = tar_frame_body.verts_project_to_abk(hr_new_inv)
        if IF_KEEP_K:
            _, hr_frame_abk = src_frame_body.verts_project_to_abk(frame_hr_verts)
            hr_ini_abk[:, :, -1] = hr_frame_abk[:, :, -1]

        hr_garment_deformer = \
            GarmentDeformer(hr_ini_abk, hr_ini_weights, tar_joints, tar_lx, tar_ly, tar_lz, tar_ls,
                            w_scale=W_SCALE).to(device)
        hr_garment_opt = torch.optim.AdamW(params=hr_garment_deformer.parameters(), lr=0.001, betas=(0.9, 0.999),
                                           amsgrad=True, fused=True)

        print(frameID, ' -> hight resolution deformation starts....')
        hr_rcc_inv, hr_new_inv, hr_new_abk, hr_new_w = \
            optimize_garment_deformer(GDeformer=hr_garment_deformer, DeformerOpt=hr_garment_opt,
                                      ini_lbs=hr_ini_weights,
                                      new_inv=hr_new_inv, src_verts=frame_hr_verts,
                                      mesh_topo=hr_mesh_topology, lap_opt=hr_laplacian,
                                      fit_parts=hr_fit_parts,
                                      src_body=src_frame_body, tar_body=tar_frame_body,
                                      iterations=8000, iter_update_hits=-1, update_factor=2, if_opt_w=if_opt_w,
                                      png_name='hr_')
        save_obj('../rst/' + tar_avatar + '/' + dress_name + '/' + pose_name + '/ours/' + str(frameID-1) + '.obj',
                 hr_new_inv.detach().cpu().numpy(), faces=hr_faces)
        save_obj('../rst/' + tar_avatar + '/' + dress_name + '/' + pose_name + '/ours/rcc/' + str(frameID-1) + '.obj',
                 hr_rcc_inv.detach().cpu().numpy(), faces=hr_faces)

        # _, hr_new_abk = tar_frame_body.verts_project_to_abk(hr_new_inv)
        #
        # color_projections(abk=hr_new_abk.detach().cpu().numpy(),
        #                   weights=hr_new_w.detach().cpu().numpy(), bcolors=BoneColorList, faces=hr_faces,
        #                   saveprefix='../test_iter/'+tar_avatar + '/' + str(frameID-1) + '/proj_tar/rst_',
        #                   bone_id_list=[4, 8, 17])
        # new_wei_color = color_vertex_by_bone_weights(hr_new_w.detach().cpu().numpy(), BoneColorList)
        # savePly('../test_iter/'+tar_avatar + '/' + str(frameID-1) + '/' + 'tar_garm_rst.ply', hr_rcc_inv.detach().cpu().numpy(), colors=new_wei_color * 255,
        #         faces=hr_faces)

        print(frameID, ' -> high resolution deformation done...')



