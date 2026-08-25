"""Physical-optics surface-integral scattering, ported from Face3D_clean's
``FieldCalculation.py`` (the verified solver) with three changes:

  1. **batched** over meshes — all samples share one face array (fixed
     topology from ``mesh3d``), so vertices stack as (B, Nv, 3);
  2. **chunked** over faces — bounded memory instead of the original's
     ~10 GB intermediates;
  3. the face-independent direct term psi0 (horn -> plane) is computed once
     by ``horn_to_plane`` instead of once per sample.

The Rayleigh-Sommerfeld kernel, the horn aperture model, the visibility
masks, and the psi1/psi2 reflection signs are copied verbatim; visibility is
applied as a 0/1 multiplier instead of fancy indexing (numerically
identical), which is what makes batching possible.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Verbatim helpers from FieldCalculation.py (batched where noted)
# ---------------------------------------------------------------------------
def surface_geometry(v: torch.Tensor, f: torch.Tensor):
    """Per-face centers, areas, outward normals.

    v: (Nv, 3) or (B, Nv, 3); f: (Nf, 3).
    Returns center (..., Nf, 3), area (..., Nf), normal (..., Nf, 3).
    """
    points = v[..., f, :]                       # (..., Nf, 3, 3)
    center = points.mean(dim=-2)
    ab = points[..., 1, :] - points[..., 0, :]
    ac = points[..., 2, :] - points[..., 0, :]
    product = torch.cross(ab, ac, dim=-1)
    area2 = torch.sqrt(torch.sum(product**2, dim=-1))   # 2 * area
    normal = product / area2.unsqueeze(-1)
    return center, area2 / 2, normal


def incident_direction(theta_inc: float, phi_inc: float = 0.0) -> torch.Tensor:
    return torch.tensor(
        [
            np.sin(theta_inc) * np.cos(phi_inc),
            np.sin(theta_inc) * np.sin(phi_inc),
            np.cos(theta_inc),
        ],
        dtype=torch.float32,
    )


def aperture_field(size_ant, dist_ant, resol_ant, theta_inc, wvl, k0):
    """Horn / open-aperture source field (verbatim Face3D ``_aperture_field``)."""
    if len(size_ant) > 3:
        a_aptr, b_aptr = size_ant[:2]
        a_wvg, b_wvg = size_ant[2:4]
        l_horn = size_ant[-1]
        r_e = a_aptr * l_horn / (a_aptr - a_wvg)
        r_h = b_aptr * l_horn / (b_aptr - b_wvg)
        xz_ant, y_ant = torch.meshgrid(
            torch.linspace(-a_aptr / 2, a_aptr / 2, resol_ant),
            torch.linspace(-b_aptr / 2, b_aptr / 2, resol_ant),
            indexing="ij",
        )
        z_ant = dist_ant * np.cos(theta_inc) + xz_ant * np.sin(theta_inc)
        x_ant = dist_ant * np.sin(theta_inc) - xz_ant * np.cos(theta_inc)
        beta_wvg = k0 * np.sqrt(1 - (wvl / (2 * a_wvg)) ** 2)
        amplitude = torch.cos(np.pi * xz_ant / a_aptr)
        psi_at_ant = amplitude * torch.exp(0.5j * beta_wvg * (xz_ant**2 / r_e + y_ant**2 / r_h))
    else:
        a_aptr, b_aptr = size_ant
        xz_ant, y_ant = torch.meshgrid(
            torch.linspace(-a_aptr / 2, a_aptr / 2, resol_ant),
            torch.linspace(-b_aptr / 2, b_aptr / 2, resol_ant),
            indexing="ij",
        )
        z_ant = dist_ant * np.cos(theta_inc) + xz_ant * np.sin(theta_inc)
        x_ant = dist_ant * np.sin(theta_inc) - xz_ant * np.cos(theta_inc)
        psi_at_ant = torch.cos(2 * np.pi * xz_ant / (2 * a_aptr))

    return psi_at_ant, x_ant, y_ant, z_ant, a_aptr, b_aptr


def rs_kernel(R: torch.Tensor, cos_term: torch.Tensor, wvl: float, k0: float) -> torch.Tensor:
    """Exact Rayleigh-Sommerfeld I kernel: (1/wvl)(1/(k0 R) - 1j)(cosθ/R) e^{jk0R}.

    ``cos_term`` is the projected distance (point2plane / obj2ant_dist), so the
    full geometric factor is cos_term / R**2, exactly as in Face3D.
    """
    return (1 / wvl) * (1 / (k0 * R) - 1j) * (cos_term / R**2) * torch.exp(1j * k0 * R)


# ---------------------------------------------------------------------------
# Direct horn -> plane term (face independent — compute once, cache)
# ---------------------------------------------------------------------------
def horn_to_plane(X, Y, H, wvl, theta_inc, size_ant, dist_ant, resol_ant) -> torch.Tensor:
    """psi0 on the observation plane; identical to Face3D's psi0_ms term."""
    k0 = 2 * np.pi / wvl
    device = X.device
    d = incident_direction(theta_inc).to(device)

    psi0_ant, x_ant, y_ant, z_ant, a_aptr, b_aptr = aperture_field(
        size_ant, dist_ant, resol_ant, theta_inc, wvl, k0
    )
    psi0_ant = psi0_ant.to(device)
    x_ant, y_ant, z_ant = x_ant.to(device), y_ant.to(device), z_ant.to(device)

    r = resol_ant
    Xa = x_ant.reshape(r, r, 1, 1)
    Ya = y_ant.reshape(r, r, 1, 1)
    Za = z_ant.reshape(r, r, 1, 1)
    r1, r2 = X.shape[1], X.shape[2]
    Xp = X.reshape(1, 1, r1, r2)
    Yp = Y.reshape(1, 1, r1, r2)

    R = torch.sqrt((Xp - Xa) ** 2 + (Yp - Ya) ** 2 + (H - Za) ** 2)
    point2plane = -((Xp - Xa) * d[0] + (Yp - Ya) * d[1] + (H - Za) * d[2])
    point2plane = point2plane * (point2plane > 0)
    kernel = rs_kernel(R, point2plane, wvl, k0)

    dS_ant = a_aptr * b_aptr / (resol_ant - 1) ** 2
    return (psi0_ant.reshape(r, r, 1, 1) * dS_ant * kernel).sum(dim=(0, 1))


# ---------------------------------------------------------------------------
# Optional ray-cast shadowing (V6 check; off by default)
# ---------------------------------------------------------------------------
def raycast_shadow_mask(
    v_occ: torch.Tensor,
    f_occ: torch.Tensor,
    center: torch.Tensor,
    src_point: torch.Tensor,
    ray_mask: torch.Tensor | None = None,
    budget: int = 2**24,
    eps: float = 1e-6,
    min_t: float = 3.0,
) -> torch.Tensor:
    """1.0 where the face centroid sees ``src_point``, 0.0 where occluded.

    Möller-Trumbore centroid->source occlusion test. The occluder mesh
    (v_occ, f_occ) may be COARSER than the mesh whose centroids are tested —
    shadow boundaries are smooth at the λ/2 scale, and coarse occluders cut
    the O(rays x tris) cost dramatically. ``ray_mask`` limits ray casting to
    faces that matter (e.g. those passing back-face culling); untested faces
    return 1.0. ``budget`` bounds rays_per_chunk x n_tris for memory safety.

    ``min_t`` (mm) ignores intersections closer than this along the ray:
    facet chords sag inside the true convex surface, so horizon-grazing rays
    otherwise clip their own neighboring facets (verified against the 2D
    line-of-sight test, which shows those rays are NOT blocked).

    The artifact lives at the chord-sagitta scale, d^2/(8R) — MEASURED at
    t <= 0.021 mm (lam=5) / 0.037 mm (lam=8) on the production geometry, about
    1/100 of a facet — while real crater walls sit at t >= 0.3 mm. Anything
    much above ~0.3 mm therefore discards real physics as well as the
    artifact. This module stays config-free (every physical quantity arrives
    as an argument); callers pass ``config.SHADOW_MIN_T``, and the V0c gate
    asserts this default still equals it so the two cannot drift.
    """
    device = v_occ.device
    tri = v_occ[f_occ]                             # (Nt, 3, 3)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    n_tris = tri.shape[0]
    n_faces = center.shape[0]
    visible = torch.ones(n_faces, device=device)

    idx = torch.arange(n_faces, device=device) if ray_mask is None else torch.nonzero(ray_mask, as_tuple=True)[0]
    if idx.numel() == 0:
        return visible
    origins = center[idx]
    dir_full = src_point.reshape(1, 3) - origins
    dist_full = dir_full.norm(dim=1, keepdim=True)
    dir_unit = dir_full / dist_full

    chunk_rays = max(1, budget // max(1, n_tris))
    for start in range(0, idx.numel(), chunk_rays):
        stop = min(start + chunk_rays, idx.numel())
        o = origins[start:stop] + dir_unit[start:stop] * 1e-3   # lift off the surface
        dvec = dir_unit[start:stop]                             # (C, 3)
        maxt = (dist_full[start:stop, 0] - 2e-3)

        p = torch.cross(dvec.unsqueeze(1), e2.unsqueeze(0), dim=2)      # (C, Nt, 3)
        det = (e1.unsqueeze(0) * p).sum(dim=2)
        valid = det.abs() > eps
        inv_det = torch.where(valid, 1.0 / det, torch.zeros_like(det))
        tvec = o.unsqueeze(1) - tri[:, 0].unsqueeze(0)
        u = (tvec * p).sum(dim=2) * inv_det
        del p
        q = torch.cross(tvec, e1.unsqueeze(0), dim=2)
        del tvec
        w = (dvec.unsqueeze(1) * q).sum(dim=2) * inv_det
        t = (e2.unsqueeze(0) * q).sum(dim=2) * inv_det
        del q
        hit = valid & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > min_t) & (t < maxt.unsqueeze(1))
        visible[idx[start:stop]] = (~hit.any(dim=1)).float()
        del u, w, t, hit, valid, inv_det, det
    return visible


# ---------------------------------------------------------------------------
# Batched, chunked scattering solver (psi1, psi2)
# ---------------------------------------------------------------------------
def scattered_fields(
    v: torch.Tensor,
    f: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    H: float,
    wvl: float,
    theta_inc: float,
    size_ant,
    dist_ant: float,
    resol_ant: int,
    chunk_faces: int = 1024,
    shadow: str = "none",
    shadow_occluders: tuple[torch.Tensor, torch.Tensor] | None = None,
    shadow_min_t: float | None = None,
    compute_psi2: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """psi1 (single bounce) and psi2 (double bounce) on the observation plane.

    v: (Nv, 3) or (B, Nv, 3); f: (Nf, 3); X, Y: (1, r1, r2) grids at height H.
    ``chunk_faces`` is the total number of face-slots processed per chunk
    *across the batch* (the per-chunk face count is chunk_faces // B), so
    memory stays bounded regardless of batch size.
    ``shadow_min_t`` overrides the ray-cast self-hit guard (mm along the ray);
    None keeps ``raycast_shadow_mask``'s default. Callers that follow the
    project config pass ``config.SHADOW_MIN_T``.
    Returns psi1, psi2 with shape (B, r1, r2) complex64 (psi2 zeros when
    compute_psi2=False).
    """
    single = v.dim() == 2
    if single:
        v = v.unsqueeze(0)
    B = v.shape[0]
    device = v.device
    k0 = 2 * np.pi / wvl
    r1, r2 = X.shape[1], X.shape[2]
    r = resol_ant

    d = incident_direction(theta_inc).to(device)
    dir_front = torch.tensor([0.0, 0.0, 1.0], device=device)

    center, area, normal = surface_geometry(v, f)          # (B,Nf,3), (B,Nf), (B,Nf,3)
    n_faces = center.shape[1]

    front = (normal @ dir_front > 0)
    left = (normal @ d > 0)
    vis1 = (front & left).float()                          # faces contributing to psi1
    vis_ant = left.float()                                 # faces contributing to psi1_ant

    if shadow == "raycast":
        src_point = torch.tensor(
            [dist_ant * np.sin(theta_inc), 0.0, dist_ant * np.cos(theta_inc)],
            dtype=torch.float32, device=device,
        )
        for b in range(B):
            if shadow_occluders is not None:
                v_occ, f_occ = shadow_occluders
                v_occ_b = v_occ[b] if v_occ.dim() == 3 else v_occ
            else:
                v_occ_b, f_occ = v[b], f
            kw_mt = {} if shadow_min_t is None else {"min_t": shadow_min_t}
            los = raycast_shadow_mask(v_occ_b, f_occ, center[b], src_point,
                                      ray_mask=left[b], **kw_mt)
            vis1[b] = vis1[b] * los
            vis_ant[b] = vis_ant[b] * los
    elif shadow != "none":
        raise ValueError(f"Unknown shadow mode: {shadow}")

    psi0_ant, x_ant, y_ant, z_ant, a_aptr, b_aptr = aperture_field(
        size_ant, dist_ant, resol_ant, theta_inc, wvl, k0
    )
    psi0_ant = psi0_ant.to(device)
    x_ant, y_ant, z_ant = x_ant.to(device), y_ant.to(device), z_ant.to(device)
    dS_ant = a_aptr * b_aptr / (resol_ant - 1) ** 2

    Xa = x_ant.reshape(1, 1, r, r)
    Ya = y_ant.reshape(1, 1, r, r)
    Za = z_ant.reshape(1, 1, r, r)

    Xp = X.reshape(1, 1, r1, r2)
    Yp = Y.reshape(1, 1, r1, r2)

    chunk = max(1, chunk_faces // B)
    psi1 = torch.zeros(B, r1, r2, dtype=torch.complex64, device=device)
    psi1_ant = torch.zeros(B, r, r, dtype=torch.complex64, device=device)

    for start in range(0, n_faces, chunk):
        stop = min(start + chunk, n_faces)
        c = center[:, start:stop]                          # (B, C, 3)
        nrm = normal[:, start:stop]
        dS = area[:, start:stop]

        Xo = c[:, :, 0].reshape(B, -1, 1, 1)
        Yo = c[:, :, 1].reshape(B, -1, 1, 1)
        Zo = c[:, :, 2].reshape(B, -1, 1, 1)

        # --- antenna -> face (illumination of every face in the chunk)
        R_ao = torch.sqrt((Xo - Xa) ** 2 + (Yo - Ya) ** 2 + (Zo - Za) ** 2)
        obj2ant = -(Xo - Xa) * np.sin(theta_inc) - (Zo - Za) * np.cos(theta_inc)
        kernel = rs_kernel(R_ao, obj2ant, wvl, k0)
        psi0_face = (psi0_ant.reshape(1, 1, r, r) * dS_ant * kernel).sum(dim=(2, 3))  # (B, C)
        del R_ao, obj2ant, kernel

        # --- face -> plane (psi1); "-" is the 180° phase flip on reflection
        R = torch.sqrt((Xp - Xo) ** 2 + (Yp - Yo) ** 2 + (H - Zo) ** 2)
        point2plane = (
            (Xp - Xo) * nrm[:, :, 0].reshape(B, -1, 1, 1)
            + (Yp - Yo) * nrm[:, :, 1].reshape(B, -1, 1, 1)
            + (H - Zo) * nrm[:, :, 2].reshape(B, -1, 1, 1)
        )
        point2plane = point2plane * (point2plane > 0)
        kernel = rs_kernel(R, point2plane, wvl, k0)
        weight = (vis1[:, start:stop] * dS * psi0_face)     # (B, C) complex
        psi1 = psi1 - torch.einsum("bc,bcxy->bxy", weight + 0j, kernel)
        del R, point2plane, kernel

        if compute_psi2:
            # --- face -> antenna aperture (psi1_ant)
            R = torch.sqrt((Xa - Xo) ** 2 + (Ya - Yo) ** 2 + (Za - Zo) ** 2)
            point2plane = (
                (Xa - Xo) * nrm[:, :, 0].reshape(B, -1, 1, 1)
                + (Ya - Yo) * nrm[:, :, 1].reshape(B, -1, 1, 1)
                + (Za - Zo) * nrm[:, :, 2].reshape(B, -1, 1, 1)
            )
            point2plane = point2plane * (point2plane > 0)
            kernel = rs_kernel(R, point2plane, wvl, k0)
            weight = (vis_ant[:, start:stop] * dS * psi0_face)
            psi1_ant = psi1_ant - torch.einsum("bc,bcxy->bxy", weight + 0j, kernel)
            del R, point2plane, kernel

    psi2 = torch.zeros_like(psi1)
    if compute_psi2:
        # --- antenna aperture -> plane (batch-independent kernel)
        Xa4 = x_ant.reshape(r, r, 1, 1)
        Ya4 = y_ant.reshape(r, r, 1, 1)
        Za4 = z_ant.reshape(r, r, 1, 1)
        R = torch.sqrt((Xp.reshape(1, 1, r1, r2) - Xa4) ** 2
                       + (Yp.reshape(1, 1, r1, r2) - Ya4) ** 2
                       + (H - Za4) ** 2)
        point2plane = -(
            (Xp.reshape(1, 1, r1, r2) - Xa4) * d[0]
            + (Yp.reshape(1, 1, r1, r2) - Ya4) * d[1]
            + (H - Za4) * d[2]
        )
        point2plane = point2plane * (point2plane > 0)
        kernel = rs_kernel(R, point2plane, wvl, k0)         # (r, r, r1, r2)
        psi2 = -dS_ant * torch.einsum(
            "brs,rsxy->bxy", psi1_ant, kernel
        )

    if single:
        return psi1.squeeze(0), psi2.squeeze(0)
    return psi1, psi2
