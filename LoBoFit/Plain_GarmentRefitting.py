from object_Handle import BodyMesh, GarmentMesh
from Loss_def import compute_loss_collision
from data_IO import save_obj, float_numpy_to_tensor, int_numpy_to_tensor, save_segment_obj, float_tensor_to_numpy, int_tensor_to_numpy

from mesh_utils import compute_mesh_areas
from GarmentDeformer import *


if __name__ == '__main__':
    import os
    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)
    device = torch.device("cuda" if USE_CUDA else "cpu")
    #device = torch.device("cpu")

    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    prefix = '../data/'
    pose_name = 'WHip_Hop'
    dress_name = 'ShirringDress'
    pd = 10
    pd_name = '/PD' + str(pd) + '_G/'

    fit_mode = 'torso_waist'  # Options: 'waist', 'torso', 'torso_waist', 'free'

    src_avatar = 'Manneq'
    src_prefix = prefix + src_avatar + '/' + pose_name + '/'

    tar_avatar = 'ortiz'
    tar_prefix = prefix + 'target_Avatar/' + tar_avatar + '/' + pose_name + '/'

    # set up canonical information for src body, src garment and tar body
    src_cano_body = BodyMesh(obj_name=src_prefix + 'body/bA.obj',
                             bones_name=src_prefix+'Bones/A.txt',
                             lbs_name=src_prefix + 'body_weight.npz')
    src_cano_garment = GarmentMesh(body_info=src_cano_body,
                                   obj_name=src_prefix + '/' + dress_name + pd_name + '/A_pose/A_pose.obj')
    # c_j, c_lx, c_ly, c_lz, c_ls = src_cano_body.get_bones_information(device)
    # save_segment_obj('../test/src_bone.obj', seg_head=c_j.cpu().numpy(),
    #                  seg_tail=(c_j+c_lz*c_ls).cpu().numpy())
    # exit(1)

    # segment body center part
    _c, src_cano_abk = src_cano_body.verts_project_to_abk(float_numpy_to_tensor(src_cano_garment.vertices, device))
    if fit_mode == 'waist':
        body_mask,garm_fit_part = abk_waist_segmentation(src_cano_abk) # shared across frames
        # save_obj('./test/w_seg.obj', vertices=src_cano_garment.vertices[int_tensor_to_numpy(garm_fit_part), :])
        # exit(1)
    elif fit_mode == 'torso':
        body_mask, garm_fit_part = abk_upbody_segmentation(src_cano_abk)
        # save_obj('./test/u_seg.obj', vertices=src_cano_garment.vertices[int_tensor_to_numpy(garm_fit_part), :])
        # exit(1)
    elif fit_mode == 'torso_waist':
        u_mask, _ = abk_upbody_segmentation(src_cano_abk)
        w_mask, _ = abk_waist_segmentation(src_cano_abk)
        mask = u_mask | w_mask
        garm_fit_part = torch.where(mask)[0]
        # save_obj('./test/uw_seg.obj', vertices=src_cano_garment.vertices[int_tensor_to_numpy(garm_fit_part), :])
        # exit(1)
    else:
        garm_fit_part = \
            torch.tensor([i for i in range(src_cano_garment.vertices.shape[0])], dtype=torch.int32).to(device)

    #
    #

    src_cano_garment.create_garment_bone_weights(device=device) # propagate LBS weight from body

    src_cano_garment.update_graph_topology(device=device) # shared across frames
    src_cano_body.lbs_to_tensor(device=device)
    ini_garment_weight = src_cano_garment.LBS_weight # shared across frames
    garment_faces = src_cano_garment.faces # shared across frames

    tar_cano_body = BodyMesh(lbs_name=tar_prefix + 'body_weight.npz')
    tar_body_lbs_w = tar_cano_body.LBS_weight
    tar_body_lbs_w_tensor = float_numpy_to_tensor(tar_body_lbs_w, device)


    def coll_alpha_step(ini_alpha, itt, start_step=0, step_size=1000, gamma=0.5, mini_thre=0.3):
        ini_alpha = min(ini_alpha, 1.)
        if itt < start_step:
            alpha = 1.
        else:
            alpha = ini_alpha * gamma ** ((itt - start_step) // step_size)
            if alpha < mini_thre:
                alpha = 0.
        return alpha

    #frame_list = [i*10+1 for i in range(21)]
    frame_list = [151]

    for frameID in frame_list:
        # set up posed information for src body, src garment and tar body
        # src
        src_frame_body = BodyMesh(obj_name=src_prefix + 'body/b' + str(frameID).zfill(4) + '.obj',
                                  bones_name=src_prefix + '/Bones/' + str(frameID) + '.txt')

        src_frame_garment = GarmentMesh(obj_name=src_prefix + '/' + dress_name + pd_name + str(frameID - 1) + '.obj',
                                        if_norm=True,
                                        body_info=src_frame_body)
        src_frame_garment.update_laplacian_operator(device=device)
        src_frame_garment.vertices_to_tensor(device=device)

        src_joints, src_lx, src_ly, src_lz, src_ls = src_frame_body.get_bones_information(device=device)

        src_abk = src_frame_garment.compute_vertices_projection_to_bones(verts=src_frame_garment.vertices)

        # tar
        tar_frame_body = BodyMesh(obj_name=tar_prefix + 'body/b' + str(frameID).zfill(4) + '.obj',
                                  bones_name=tar_prefix + '/Bones/' + str(frameID) + '.txt')

        tar_frame_garments = GarmentMesh(body_info=tar_frame_body)

        # compute intersection mapping, key to drive the garment mesh deformation
        new_inv = tar_frame_garments.compute_abk_backprojection_to_verts(abk=src_abk, weight=ini_garment_weight)

        # debug
        # save_obj('./test/ini_g.obj', new_inv.cpu().numpy(), faces=garment_faces)
        # exit(1)

        valid_abk_ids, hit_body_verts, hit_body_norms, hit_body_faces, hit_face_wei = \
            tar_frame_body.bone_related_search_nearest_body_verts(query_verts=new_inv,
                                                                  query_lbs_weights=ini_garment_weight,
                                                                  body_lbs_weight=tar_body_lbs_w,
                                                                  norm_opposit_thr=0.1, lbs_wei_align_thr=0.1,
                                                                  if_area_weight=True)
        valid_abk_ids = int_numpy_to_tensor(valid_abk_ids, device)

        # to optimize \Delta abkw
        # N, J, _ = src_abk.shape
        # delta_abkw = torch.zeros(N, J, 4).type(torch.float32).to(device)  # [N, K, 4]
        # delta_abkw.requires_grad = True
        tar_joints, tar_lx, tar_ly, tar_lz, tar_ls = tar_frame_body.get_bones_information(device=device)


        tar_garment_deformer = \
            GarmentDeformer(src_abk, ini_garment_weight, tar_joints, tar_lx, tar_ly, tar_lz, tar_ls, w_scale=0.1).to(device)

        opt = torch.optim.AdamW(params=tar_garment_deformer.parameters(), lr=0.005, betas=(0.9, 0.999), amsgrad=True)


        # get gt metric for optimization supervision
        lap_opt = src_frame_garment.Laplacian_Opt
        mesh_topo = src_cano_garment.graph_topo
        src_fg_normals = float_numpy_to_tensor(src_frame_garment.normals, device)
        src_fg_normals = torch.nn.functional.normalize(src_fg_normals, eps=1e-12, dim=1)
        tar_body_faces_tensor = int_numpy_to_tensor(tar_frame_body.faces, device)
        #tar_scale = src_cano_garment.compute_lbs_feats_blending(tar_ls)
        # src_g_faces = src_frame_garment.get_faces()
        # print(src_g_faces.shape)

        gt_adj_cos, adj_wei, num_c = tar_frame_garments.compute_curve_adj_cos(verts=src_frame_garment.vertices,
                                                                              graph_top=mesh_topo,
                                                                              if_need_wei=True)
        gt_bend_angle, bend_wei = tar_frame_garments.compute_dihedral_angle(verts=src_frame_garment.vertices,
                                                                            graph_top=mesh_topo,
                                                                            if_need_wei=True)

        #src_scale = src_cano_garment.compute_lbs_feats_blending(src_ls)
        gt_delta = src_frame_garment.compute_norm_laplacian_delta(verts=src_frame_garment.vertices,
                                                                  lap_opt=lap_opt, if_norm=True)

        gt_normals = compute_face_normals(src_frame_garment.vertices, mesh_topo.faces)
        gt_area = compute_mesh_areas(src_frame_garment.vertices, mesh_topo.faces)
        gt_area_w = gt_area / gt_area.sum().clamp_min(1e-8)

        gt_body_fit_sdf = \
            src_frame_body.compute_verts_sign_distance(
                float_tensor_to_numpy(src_frame_garment.vertices[garm_fit_part, :]))
        gt_body_fit_sdf = float_numpy_to_tensor(gt_body_fit_sdf, device)
        sel = torch.where(gt_body_fit_sdf < 0.02)[0]
        gt_sdf_focus = gt_body_fit_sdf[sel]
        fit_focus = garm_fit_part[sel]
        #save_obj('./test/fit.obj', vertices=src_frame_garment.vertices[fit_focus].detach().cpu().numpy())

        bnn_m = torch.isin(valid_abk_ids, fit_focus)
        bnn_fit_focus = torch.nonzero(bnn_m, as_tuple=False).squeeze(1).to(torch.int32)
        # print(valid_abk_ids.shape)
        # print(bnn_fit_focus)
        # save_obj('./test/fit_abk.obj',
        #          vertices=src_frame_garment.vertices[valid_abk_ids[bnn_fit_focus]].detach().cpu().numpy())


        bnn_rm = torch.isin(fit_focus, valid_abk_ids)
        gt_fit_id = torch.nonzero(bnn_rm, as_tuple=False).squeeze(1).to(torch.int32)
        gt_bnn_sdf_focus = gt_sdf_focus[gt_fit_id]
        # print(fit_focus.shape)
        # print(gt_fit_id)
        # save_obj('./test/fit_re.obj',
        #          vertices=src_frame_garment.vertices[fit_focus[gt_fit_id]].detach().cpu().numpy())
        # exit(1)


        dhits_fidx, dhits_pnts, dhits_norms, dhits_weights = None, None, None, None
        dhits_valid_id = None
        dhits_update = False

        num_iter = 12000
        ini_coll_alpha = 1.
        for itt in range(num_iter + 1):
            coll_alpha = coll_alpha_step(ini_coll_alpha, itt, start_step=5000, step_size=2000)
            #coll_alpha = 1.
            if itt >= 5000:
                coll_alpha = 0.
                if (itt-5000) % 2000 == 0:
                    dhits_update = False
            opt.zero_grad()

            new_inv = tar_garment_deformer(if_opt_w=True)

            # similarity constraints
            loss_bend = tar_garment_deformer.bend_metric(new_inv, mesh_topo, gt_bend_angle, bend_wei)
            loss_laplacian = tar_garment_deformer.laplacian_metric(new_inv, lap_opt, gt_delta)
            loss_curve = tar_garment_deformer.curve_metric(new_inv, mesh_topo, gt_adj_cos, adj_wei)
            loss_normal = tar_garment_deformer.normal_metric(new_inv, mesh_topo, gt_normals, gt_area_w)

            #loss_hips_laplacian = tar_garment_deformer.hips_laplacian_difference(new_inv, BonesID_dict["Hips"], lap_opt)

            # regularization
            loss_k = tar_garment_deformer.reg_k()
            loss_w = tar_garment_deformer.reg_w()
            #loss_k = tar_garment_deformer.strict_reg_k(new_inv)


            # collision penalty
            loss_collision_s, sdf = compute_loss_collision(base_verts=hit_body_verts, base_norms=hit_body_norms,
                                                           verts=new_inv[valid_abk_ids, :], d_thre=0.008, wei=hit_face_wei)

            if coll_alpha < 1.:
                # save_obj('./test/inv_g_' + str(itt) + '.obj', new_inv.detach().cpu().numpy(), faces=garment_faces)
                # s_h = new_inv[valid_abk_ids, :]
                # s_t = hit_body_verts
                # save_segment_obj('./test/bnn.obj', seg_head=s_h.detach().cpu().numpy(),
                #                  seg_tail=s_t.detach().cpu().numpy())
                if dhits_update is not True:
                    dhits_fidx, dhits_pnts, dhits_norms, dhits_weights = tar_frame_body.query_pontential_collision(new_inv)
                    #dhits_valid_id = filter_hits_by_normal_lbsweights(src_fg_normals, dhits_norms, norm_thred=-0.2)
                    dhits_update = True
                # compute hits lbs weights
                #dhits_faces = tar_body_faces_tensor[dhits_fidx, :]
                #dhits_LBS = compute_feats_face_centers(tar_body_lbs_w_tensor, dhits_faces)
                # print(dhits_LBS.shape, ini_garment_weight.shape)
                # exit(1)
                # valid_id = filter_hits_by_normal_lbsweights(num_v=ini_garment_weight.shape[0],
                #                                             device=ini_garment_weight.device,
                #                                             lbsweights=ini_garment_weight,
                #                                             base_lbsweights=dhits_LBS, lbswei_thred=0.1)

                # valid_id = filter_hits_by_normal_lbsweights(src_fg_normals, dhits_norms, norm_thred = 0.1,
                #                                             lbsweights=ini_garment_weight,
                #                                             base_lbsweights=dhits_LBS, lbswei_thred=0.1)

                # s_h = new_inv[dhits_valid_id, :]
                # s_t = dhits_pnts[dhits_valid_id, :]
                # save_segment_obj('./test/nn.obj', seg_head=s_h.detach().cpu().numpy(),
                #                  seg_tail=s_t.detach().cpu().numpy())
                # exit(1)
                # loss_collision_d, sdf = compute_loss_collision(base_verts=dhits_pnts[dhits_valid_id, :],
                #                                                base_norms=dhits_norms[dhits_valid_id, :],
                #                                                verts=new_inv[dhits_valid_id, :],
                #                                                d_thre=0.008, wei=dhits_weights[dhits_valid_id])
                loss_collision_d, sdf = compute_loss_collision(base_verts=dhits_pnts,
                                                               base_norms=dhits_norms,
                                                               verts=new_inv,
                                                               d_thre=0.008, wei=dhits_weights)
                fit_sdf = sdf[fit_focus]
                loss_fitness = compute_feats_loss(fit_sdf, gt_sdf_focus, dhits_weights[fit_focus])

            else:
                fit_sdf = sdf[bnn_fit_focus]
                loss_fitness = compute_feats_loss(fit_sdf, gt_bnn_sdf_focus, hit_face_wei[bnn_fit_focus])
                loss_collision_d = 0.

            loss_collision = coll_alpha * loss_collision_s + (1 - coll_alpha) * loss_collision_d

            loss = 10 * loss_collision + 1e2 * loss_fitness + \
                   1. * loss_bend + 1 * loss_curve + 0.5 * loss_laplacian + 1e-5 * loss_normal + \
                   1 * loss_k + 1e-2 * loss_w

            if itt % 500 == 0:
                print(itt, loss.item(), loss_bend.item(), loss_curve.item(), loss_laplacian.item(),
                      loss_collision.item(), loss_k.item(), loss_normal.item(), loss_w.item(), loss_fitness.item())
                #save_obj('../test/inv_g_' + str(itt) + '.obj', new_inv.detach().cpu().numpy(), faces=garment_faces)
            if itt == num_iter:
                # save_obj('../test/' + tar_avatar + '/inv_' + str(frameID) + '.obj',
                #          new_inv.detach().cpu().numpy(), faces=garment_faces)
                save_obj('../rst/' + tar_avatar + '/' + dress_name + '/' + pose_name + '/ours_plain/' + str(frameID - 1) + '.obj',
                         new_inv.detach().cpu().numpy(), faces=garment_faces)
                print('frame_', frameID, '...Done.')
                # s_h = new_inv[valid_abk_ids, :]
                # s_t = hit_body_verts
                # save_segment_obj('../test/intersection_mapping.obj', seg_head=s_h.detach().cpu().numpy(),
                #                  seg_tail=s_t.detach().cpu().numpy())

                break

            loss.backward()
            opt.step()
























