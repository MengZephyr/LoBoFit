import numpy as np
from collections import defaultdict, deque
from sklearn import neighbors
import torch

def faces_to_edges_and_adjacency(faces):
    edges = dict()
    for fidx, face in enumerate(faces):
        for i, v in enumerate(face):
            nv = face[(i + 1) % len(face)]
            edge = tuple(sorted([v, nv]))
            if not edge in edges:
                edges[edge] = []
            edges[edge] += [fidx]
    face_adjacency = []
    face_adjacency_edges = []
    curve_edges = []
    for edge, face_list in edges.items():
        if len(face_list) < 2:
            curve_edges += [edge]
            continue
        for i in range(len(face_list) - 1):
            for j in range(i + 1, len(face_list)):
                face_adjacency += [[face_list[i], face_list[j]]]
                face_adjacency_edges += [edge]
    edges = np.array([list(edge) for edge in edges.keys()], np.int32)
    face_adjacency = np.array(face_adjacency, np.int32)
    face_adjacency_edges = np.array(face_adjacency_edges, np.int32)
    curve_edges = np.array(curve_edges, np.int32)

    return edges, face_adjacency, face_adjacency_edges, curve_edges


def extract_cycle_edge_chains(edges, check_degree2=True):
    '''
    edges: should be edges belonging to meshes with no adjacent
    '''
    E = np.asarray(edges, dtype=np.int64)
    if E.ndim != 2 or E.shape[1] != 2:
        raise ValueError("edges must be shape (C,2)")
    if len(E) == 0:
        return 0, []

    adj = defaultdict(list)
    for a, b in E:
        if a == b:
            raise ValueError("self-loop is not allowed")
        adj[a].append(b)
        adj[b].append(a)

    if check_degree2:
        bad = [v for v, ns in adj.items() if len(ns) != 2]
        if bad:
            raise ValueError(f"Non-cycle vertices (deg != 2): {bad[:10]} ...")

    undirected = set((int(a), int(b)) if a < b else (int(b), int(a)) for a, b in E)
    visited = set()

    curves_edges = []

    while undirected - visited:
        a, b = next(iter(undirected - visited))
        comp = set()
        q = deque([a])
        while q:
            v = q.popleft()
            if v in comp:
                continue
            comp.add(v)
            for u in adj[v]:
                e = (v, u) if v < u else (u, v)
                if e in undirected:
                    q.append(u)

        start = min(comp)
        n1, n2 = adj[start]
        first_next = min(n1, n2)

        prev, curr = start, first_next
        edge_chain = []
        while True:
            e_und = (prev, curr) if prev < curr else (curr, prev)
            visited.add(e_und)
            edge_chain.append([prev, curr])

            if curr == start:
                break

            a1, a2 = adj[curr]
            nxt = a1 if a1 != prev else a2
            prev, curr = curr, nxt

        curves_edges.append(np.asarray(edge_chain, dtype=np.int64))

    return len(curves_edges), curves_edges

def kdTree_nearest_neighbor(query_verts, tree_verts):
    vtrees = neighbors.KDTree(tree_verts)
    _, vInd = vtrees.query(query_verts, k=1)
    neiList = [i[0] for i in vInd]
    return neiList


def compute_face_normals(verts, vfaceIDs):
    fv12 = verts[vfaceIDs[:, 2], :] - verts[vfaceIDs[:, 1], :]
    fv10 = verts[vfaceIDs[:, 0], :] - verts[vfaceIDs[:, 1], :]
    faces_normals = torch.cross(fv12, fv10, dim=1)
    faces_normals = torch.nn.functional.normalize(faces_normals, eps=1e-12, dim=1)
    return faces_normals


def compute_face_centers(verts, vfaceIDs):
    fv0 = verts[vfaceIDs[:, 0], :]
    fv1 = verts[vfaceIDs[:, 1], :]
    fv2 = verts[vfaceIDs[:, 2], :]
    fc = (fv0+fv1+fv2)/torch.tensor(3).to(verts.device)
    return fc


def compute_feats_face_centers(feats:torch.Tensor, vfaceIDs):
    fv0 = feats[vfaceIDs[:, 0], ...]
    fv1 = feats[vfaceIDs[:, 1], ...]
    fv2 = feats[vfaceIDs[:, 2], ...]
    fc = (fv0 + fv1 + fv2) / torch.tensor(3).to(feats.device)
    return fc

def compute_mesh_areas(verts, faces, eps=1e-8):
    v0 = verts[faces[:, 0], :]
    v1 = verts[faces[:, 1], :]
    v2 = verts[faces[:, 2], :]

    e1 = v1 - v0 # [F, 3]
    e2 = v2 - v0 # [F, 3]
    ec = torch.cross(e1, e2, dim=-1)
    area = 0.5 * torch.linalg.norm(ec, dim=-1).clamp_min(eps)

    return area

def triangle_centroids(V, F):
    v0, v1, v2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    C = (v0 + v1 + v2) / 3.0
    N = torch.nn.functional.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
    A = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)  # area
    return C, N, A  # (M,3),(M,3),(M,)

def np_feats_interpolation(inter_faceID, inter_faceUV, faces, feats):
    face_feat0 = feats[faces[inter_faceID, 0], :]
    face_feat1 = feats[faces[inter_faceID, 1], :]
    face_feat2 = feats[faces[inter_faceID, 2], :]
    u, v = inter_faceUV[:, 0], inter_faceUV[:, 1]
    w = 1.0 - u - v
    q_feat = face_feat0 * w[:, None] + face_feat1 * u[:, None] + face_feat2 * v[:, None]
    return q_feat


def sample_from_faces(vertarray: torch.Tensor, sampleInFaceVID: torch.Tensor, sampleInFaceABC: torch.Tensor):
    Vert0 = vertarray[sampleInFaceVID[:, 0], :].unsqueeze(1)  # Coarse_Verts_N * 1 * d
    Vert1 = vertarray[sampleInFaceVID[:, 1], :].unsqueeze(1)  # Coarse_Verts_N * 1 * d
    Vert2 = vertarray[sampleInFaceVID[:, 2], :].unsqueeze(1)  # Coarse_Verts_N * 1 * d
    FaceVerts = torch.cat([Vert0, Vert1, Vert2], dim=1)  # Coarse_Verts_N * d * d
    SampleVerts = torch.matmul(sampleInFaceABC.unsqueeze(1), FaceVerts).squeeze(1)  # Coarse_Verts_N * d
    return SampleVerts


def dihedral_angle_adjacent_faces(normals, adjacency):
    normals0 = normals[adjacency[:, 0]]
    normals1 = normals[adjacency[:, 1]]
    cos = torch.einsum("ab,ab->a", normals0, normals1)
    sin = torch.norm(torch.cross(normals0, normals1, dim=-1), dim=-1)
    theta = torch.arctan2(sin, cos)
    return theta


def edge_curve_adj_cos(num_chains, c_chains, verts, eps=1.e-12):
    device = verts.device
    chain_adj_cos = []
    for i in range(num_chains):
        ch = torch.as_tensor(c_chains[i], device=device, dtype=torch.long)

        e = verts[ch[:, 0]] - verts[ch[:, 1]]  # [E,3]
        e_len = e.norm(dim=-1)  # [E]
        e_unit = e / (e_len.unsqueeze(-1) + eps)  # [E,3]

        e_unit_next = torch.roll(e_unit, shifts=-1, dims=0)  # [E,3]
        e_adj_cos = (e_unit * e_unit_next).sum(dim=-1)  # [E]
        chain_adj_cos.append(e_adj_cos)

    return chain_adj_cos