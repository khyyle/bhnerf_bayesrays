from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from typing import Callable
from pathlib import Path
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from mpl_toolkits.axes_grid1 import make_axes_locatable
import bhnerf
import bhnerf.network
import importlib
importlib.reload(bhnerf.network)
import bhnerf.network
from bhnerf import constants as consts
from ehtim.image import make_square
from ehtim.imaging import imager_utils as iu
from astropy import units

def build_A_per_frame(obs, t_frames, image_fov_rad, image_size, dtype='vis', pol='I'):
    """
    Args:
        obs: Obsdata object for the black hole
        t_frames: (nt,) astropy quantity array of times for each frame
        image_fov_rad: image field of view in radians
        image_size: image size in pixels
        dtype: chisqdata type
        pol: polarization type.
    
    Returns
        target: (nt, Nvis_t, Npol)
        sigma: (nt, Nvis_t, Npol)
        A: (nt, Nvis_t, image_size**2)
    """
    dt_sec = (t_frames[-1] - t_frames[0]).to('s').value / (len(t_frames) + 1)
    obs_list = obs.split_obs(t_gather=dt_sec)
    prior = make_square(obs, image_size, image_fov_rad)
    chisq_fn = getattr(iu, f'chisqdata_{dtype}')

    targ, sig, A_all = [], [], []
    for ob in obs_list:
        t, s, A = chisq_fn(ob, prior, mask=[], pol=pol)
        targ.append(np.asarray(t))
        sig.append(np.asarray(s))
        A_all.append(np.asarray(A))
    return (np.stack(targ), np.stack(sig), np.stack(A_all))

def make_forward_model(predictor_apply, predictor_params, t_frame, ray_args, A):
    t_units = t_frame.unit
    t_frame    = t_frame
    Omega      = ray_args['Omega']
    g          = ray_args['g']
    dtau       = ray_args['dtau']
    Sigma      = ray_args['Sigma']
    t_geos     = ray_args['t_geos']
    J          = ray_args['J']
    t_start    = ray_args['t_start_obs']
    t_inj      = ray_args['t_injection']
    t_units    = t_units 
    
    @jax.jit
    def forward_model(coords):
        x, y, z = jnp.moveaxis(coords, -1, 0)
        coords_list = [x, y, z]

        I_img = bhnerf.network.image_plane_prediction(
            predictor_params, predictor_apply,
            t_frames          = t_frame,
            coords            = coords_list,
            Omega             = Omega,
            J                 = J,
            g                 = g,
            dtau              = dtau,
            Sigma             = Sigma,
            t_start_obs       = t_start,
            t_geos            = t_geos,
            t_injection       = t_inj,
            t_units           = t_units,
        )
        imvec = I_img.reshape(-1)
        vis = (A @ imvec)
        return vis
    return forward_model

def interpolation_check_scalar(volume3d: jnp.ndarray, fov: float, title: str = ""):
    """volume3d must be (R,R,R). No channels added; output remains (R,R,R)."""

    def trilinear_scalar(coords_unit: jnp.ndarray, grid3d: jnp.ndarray) -> jnp.ndarray:
        """coords_unit: (R,R,R,3) in [0,1); grid3d: (R,R,R) -> (R,R,R)"""
        grid4 = grid3d[..., None]
        out4  = trilinear(coords_unit, grid4)
        return out4[..., 0]

    def mapcoords_scalar(coords_unit: jnp.ndarray, grid3d: jnp.ndarray) -> jnp.ndarray:
        """coords_unit: (R,R,R,3) in [0,1); grid3d: (R,R,R) -> (R,R,R)"""
        R = grid3d.shape[0]
        xi = coords_unit[..., 0] * (R - 1)
        yi = coords_unit[..., 1] * (R - 1)
        zi = coords_unit[..., 2] * (R - 1)
        idx = jnp.stack([xi, yi, zi], axis=0)
        return jax.scipy.ndimage.map_coordinates(grid3d, idx, order=1, mode='nearest')

    def _coords_unit_identity(R: int) -> jnp.ndarray:
        eps = 1e-7
        xs = jnp.linspace(0.0, 1.0 - eps, R)
        ys = jnp.linspace(0.0, 1.0 - eps, R)
        zs = jnp.linspace(0.0, 1.0 - eps, R)
        return jnp.stack(jnp.meshgrid(xs, ys, zs, indexing='ij'), axis=-1)
    import matplotlib.pyplot as plt
    if volume3d.ndim != 3:
        raise ValueError("Pass a scalar volume shaped (R, R, R).")

    R = volume3d.shape[0]
    coords = _coords_unit_identity(R)

    out_tri = trilinear_scalar(coords, volume3d)
    out_map = mapcoords_scalar(coords, volume3d)

    e_tri = float(jnp.max(jnp.abs(out_tri - volume3d)))
    e_map = float(jnp.max(jnp.abs(out_map - volume3d)))
    e_tm  = float(jnp.max(jnp.abs(out_tri - out_map)))

    print(f"[{title}] shape={tuple(volume3d.shape)}")
    print("  max |trilinear - original|   :", e_tri)
    print("  max |map_coords - original|  :", e_map)
    print("  max |trilinear - map_coords| :", e_tm)

    zc = R // 2
    orig = np.asarray(volume3d[:, :, zc])
    tri  = np.asarray(out_tri[:, :, zc])
    mco  = np.asarray(out_map[:, :, zc])

    fig, axs = plt.subplots(2, 3, figsize=(12, 8)); axs = axs.ravel()
    im0 = axs[0].imshow(orig, origin='lower'); axs[0].set_title('Original [z mid]')
    im1 = axs[1].imshow(tri,  origin='lower'); axs[1].set_title('Trilinear [z mid]')
    im2 = axs[2].imshow(mco,  origin='lower'); axs[2].set_title('MapCoords [z mid]')
    im3 = axs[3].imshow(tri - orig, origin='lower'); axs[3].set_title('Tri - Orig (diff)')
    im4 = axs[4].imshow(mco - orig, origin='lower'); axs[4].set_title('Map - Orig (diff)')
    im5 = axs[5].imshow(tri - mco,  origin='lower');  axs[5].set_title('Tri - Map (diff)')
    for ax in axs: ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im0, ax=[axs[0], axs[1], axs[2]], shrink=0.75)
    fig.colorbar(im3, ax=[axs[3], axs[4], axs[5]], shrink=0.75)
    plt.tight_layout(); plt.show()

    import bhnerf.visualization as vis
    vis.ipyvolume_3d(np.asarray(volume3d), fov, level=[0.0, 0.2, 0.7])
    vis.ipyvolume_3d(np.asarray(out_tri),  fov, level=[0.0, 0.2, 0.7])
    vis.ipyvolume_3d(np.asarray(out_map),  fov, level=[0.0, 0.2, 0.7])

    return np.asarray(out_tri), np.asarray(out_map)

def trilinear_old(coords: jnp.ndarray, grid: jnp.ndarray, variance: bool = False) -> jnp.ndarray:
    """
    Trilinear interpolation into grid at normalized 3-D *coords*.

    Parameters
    ----------
    coords : [..., 3] float32 in [0, 1]^3
    grid   : [gx, gy, gz, C] values at voxel corners
             - For values (e.g., emission): store values in C channels.
             - For variances (e.g., σ_x^2, σ_y^2, σ_z^2): store *variances*
               in C channels. This function will combine corners with squared
               barycentric weights when `variance=True`.

    Returns
    -------
    out : [..., C]
      Interpolated values if variance=False;
      Interpolated variances if variance=True (already in σ^2 units).
    """
    def _index_and_weights(u, size):
        # keep strictly inside so i1 exists
        u = jnp.clip(u, 0.0, 1.0 - 1e-7)
        x = u * (size - 1)
        i0 = jnp.floor(x).astype(jnp.int32)
        i1 = i0 + 1
        w1 = x - i0
        w0 = 1.0 - w1
        return i0, i1, w0, w1

    gx, gy, gz, _ = grid.shape

    i0,i1,wx0,wx1 = _index_and_weights(coords[...,0], gx)
    j0,j1,wy0,wy1 = _index_and_weights(coords[...,1], gy)
    k0,k1,wz0,wz1 = _index_and_weights(coords[...,2], gz)

    w000 = wx0*wy0*wz0 
    w100 = wx1*wy0*wz0
    w010 = wx0*wy1*wz0
    w110 = wx1*wy1*wz0
    w001 = wx0*wy0*wz1
    w101 = wx1*wy0*wz1
    w011 = wx0*wy1*wz1
    w111 = wx1*wy1*wz1

    weights = jnp.stack([w000,w100,w010,w110,w001,w101,w011,w111], axis=-1)
    
    # Var = SUM((weight_i)^2 Var_i)
    if variance: # interpolate variance w squared weights--technically statistically correct
        weights = weights ** 2

    def gather(ii, jj, kk):
        return grid[ii, jj, kk]

    c000 = gather(i0,j0,k0)
    c100 = gather(i1,j0,k0)
    c010 = gather(i0,j1,k0)
    c110 = gather(i1,j1,k0)
    c001 = gather(i0,j0,k1)
    c101 = gather(i1,j0,k1)
    c011 = gather(i0,j1,k1)
    c111 = gather(i1,j1,k1)

    corners = jnp.stack([c000,c100,c010,c110,c001,c101,c011,c111], axis=-2)
    out = jnp.sum(corners * weights[..., None], axis=-2)
    return out

def trilinear(coords_unit: jnp.ndarray, grid: jnp.ndarray, variance=None) -> jnp.ndarray:
    nx, ny, nz, C = grid.shape
    # move to index space
    xi = coords_unit[..., 0] * (nx - 1) 
    yi = coords_unit[..., 1] * (ny - 1) 
    zi = coords_unit[..., 2] * (nz - 1)
    
    idx = jnp.stack([xi, yi, zi], axis=0)
    grid_c_first = jnp.moveaxis(grid, -1, 0)
    def interp_one(g):
        return jax.scipy.ndimage.map_coordinates(g, idx, order=1, cval=0.)
    out = jax.vmap(interp_one, in_axes=0)(grid_c_first)
    return jnp.moveaxis(out, 0, -1)

def flatten(tree):
    return jnp.concatenate([t.ravel() for t in jax.tree_util.tree_leaves(tree)])

class DeformationGrid(nn.Module):
    resolution: tuple[int, ...]

    @nn.compact
    def __call__(self, coords):
        theta = self.param('theta', nn.initializers.zeros, self.resolution + (coords.shape[-1],))
        return trilinear(coords, theta)



# brandon notes
# for first frame, compute the movie, compute the measurements at teach t, compute the loss at each timestep, and then propagate back to t=0
# we have a gradient that is a sum of all gradients from each time step

# for ex 1st frame, can make a movie using the coordinate changes, but to get back to first frame, do inverse of operation it takes to get from 2nd to 1st frame and repeat bayesrays?

class BayesRaysUncertaintyMapper():
    def __init__(self, predictor_apply: Callable, predictor_params, raytracing_args: dict, t_frames: jnp.ndarray, 
                 frames_to_include: list[int], A, sigma, fov, grid_res: tuple=(64, 64, 64), lam: float=1e-4/(64**3), mode: str='eht'):
        assert mode in ('image', 'eht'), "mode must be either 'image' or 'eht'"
        self.mode = mode
        self.P = 3 * grid_res[0] * grid_res[1] * grid_res[2]
        self.grid_res = grid_res
        self.lam = lam

        #predictor and deformation grid
        self.pred_apply = predictor_apply
        self.pred_params = jax.tree_util.tree_map(jnp.array, predictor_params)
        self.def_grid = DeformationGrid(grid_res)
        self.def_params = self.def_grid.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))['params']

        # raytracing and timing
        self.rt_args = raytracing_args
        self.t_frames = t_frames
        t_units = t_frames.unit
        
        # build coords and normalized
        self.coords = jnp.stack(raytracing_args["coords"], axis=-1) # stack channels along last axis to make deformation grid math simple
        self.coords_unit = (self.coords/(fov/2.0) + 1.0) * 0.5 #NOTE: attempt 1 for building coords. see ref_code.py for other attempts
        self.coords_unit = jnp.clip(self.coords_unit, 0.0, 1.0 - 1e-7)
        xyz_half = jnp.max(jnp.abs(self.coords), axis=(0,1,2))
        self.voxel_world = (2.0 * xyz_half) / (jnp.array(self.grid_res) - 1.0)

        print('==== COORDINATE DEBUG ====')
        print(" Clipped frac per axis:", np.array(((self.coords_unit <= 1e-6) | (self.coords_unit >= 1-1e-6)).mean(axis=(0,1,2,3))))
        print(" Coords (bayesrays):", self.coords.shape)
        print(" Coords min and max:", self.coords.max(), self.coords.min())
        print(" Coords unit min, max:", self.coords_unit.max(axis=(0,1,2,3)), self.coords_unit.min(axis=(0,1,2,3)))
        print(" Coords unit shape:", self.coords_unit.shape, "Coords shape:", self.coords.shape)

        # Restrict forward model to selected frames
        t_list = [t_frames[i] for i in frames_to_include]

        if self.mode == 'eht':
            if A is None:
                raise ValueError("EHT mode requires framewise DFT matrix, A")
            A_list = [jnp.asarray(A[i]) for i in frames_to_include]
            self.nvis_per_frame = [Af.shape[0] for Af in A_list]
            self.sigma = sigma
        elif self.mode == 'image':
            _, H, W, _ = raytracing_args['coords'].shape
            npix = H*W
            frames_idx = np.asarray(frames_to_include)
            T = len(frames_idx)
            if (sigma is None) or np.isscalar(sigma):
                val = float(sigma) if np.isscalar(sigma) else 1.0
                self.sigma = jnp.full((T, npix), val)
            else:
                sigma = jnp.asarray(sigma)
                self.sigma = sigma[frames_idx].reshape(T, npix)
            
            self.nvis_per_frame = len(frames_to_include) * [H * W]
        self.nvis = int(sum(self.nvis_per_frame))

        Omega  = raytracing_args['Omega']
        g      = raytracing_args['g']
        dtau   = raytracing_args['dtau']
        Sigma  = raytracing_args['Sigma']
        t_geos = raytracing_args['t_geos']
        J      = raytracing_args['J']
        t_start= raytracing_args['t_start_obs']
        t_inj  = raytracing_args['t_injection']
        
        # build framewise forward models
        self.fm_per_frame = []
        for i, t_f in enumerate(t_list):
            A_f = None if self.mode == 'image' else A_list[i]
            @jax.jit
            def fm(coords, *, t_f=t_f, A_f=A_f):
                coords_moved = jnp.moveaxis(coords, -1, 0) # put axes back in order bhnerf uses
                img = bhnerf.network.image_plane_prediction(
                    predictor_params, predictor_apply,
                    t_frames    = t_f,
                    coords      = [coords_moved[0], coords_moved[1], coords_moved[2]],
                    Omega       = Omega,
                    J           = J,
                    g           = g,
                    dtau        = dtau,
                    Sigma       = Sigma,
                    t_start_obs = t_start,
                    t_geos      = t_geos,
                    t_injection = t_inj,
                    t_units     = t_units,
                )
                imvec = img.reshape(-1)
                if mode == 'image':
                    return imvec
                return A_f @ imvec
            self.fm_per_frame.append(fm)
        
        if self.mode == 'eht': assert A[frames_to_include].shape[0] == sigma[frames_to_include].shape[0] == len(frames_to_include), f'{A.shape}, {sigma.shape}, {len(frames_to_include)}'
        assert self.coords.shape[0] == self.coords_unit.shape[0] == len(self.t_frames), f'{self.coords.shape}, {self.coords_unit.shape}, {len(t_frames)})'
        assert len(self.fm_per_frame) == len(self.t_frames[frames_to_include]), f'{len(self.fm_per_frame)}, {self.t_frames[frames_to_include]}'
    
    def param_occupancy(self, fov, eight_corners=True):
        R = np.array(self.grid_res, dtype=np.int32)

        half = np.array([fov, fov, fov], dtype=np.float32) * 0.5
        in_vol = np.all(
            (np.asarray(self.coords) >= -half) & (np.asarray(self.coords) <= +half),
            axis=-1
        )

        u = np.asarray(self.coords_unit)[in_vol]

        occ = np.zeros(tuple(R), dtype=np.float32)
        if not eight_corners:
            i = np.floor(u[:, 0] * (R[0]-1)).astype(np.int32)
            j = np.floor(u[:, 1] * (R[1]-1)).astype(np.int32)
            k = np.floor(u[:, 2] * (R[2]-1)).astype(np.int32)
            np.add.at(occ, (i, j, k), 1.0)
            return occ

        x = u[:,0]*(R[0]-1); i0 = np.floor(x).astype(np.int32); wx = x - i0
        y = u[:,1]*(R[1]-1); j0 = np.floor(y).astype(np.int32); wy = y - j0
        z = u[:,2]*(R[2]-1); k0 = np.floor(z).astype(np.int32); wz = z - k0
        i1, j1, k1 = i0+1, j0+1, k0+1

        w000=(1-wx)*(1-wy)*(1-wz); w100=wx*(1-wy)*(1-wz)
        w010=(1-wx)*wy*(1-wz);     w110=wx*wy*(1-wz)
        w001=(1-wx)*(1-wy)*wz;     w101=wx*(1-wy)*wz
        w011=(1-wx)*wy*wz;         w111=wx*wy*wz

        for (ii,jj,kk,ww) in [(i0,j0,k0,w000),(i1,j0,k0,w100),(i0,j1,k0,w010),(i1,j1,k0,w110),
                            (i0,j0,k1,w001),(i1,j0,k1,w101),(i0,j1,k1,w011),(i1,j1,k1,w111)]:
            np.add.at(occ, (ii, jj, kk), ww)
        return occ

    def get_covariance(self, H:jnp.array, mask_by_occpancy=False, fov=None):
        print(f"computing covariance matrix with lambda: {self.lam}")
        if mask_by_occpancy:
            occ = self.param_occupancy(fov)
            H_grid = H.reshape(*self.grid_res, 3)
            H_norm = H_grid / occ[..., None]
            H_norm = jnp.where((occ[..., None] > 0), H_norm, 0.0)
            return 1/(H_norm/self.nvis + 2 * self.lam)
        else:
            return 1/(H/self.nvis + 2 * self.lam)

    def upsample(self, V, resolution=None, squared_weights=False):
        """
        Trilinearly upsample covariance diagonal, V to resolution R
        
        Args:
            V: covariance diagonal
            R: upsample resolution
        Returns:
            covariance grid, shape (3, R, R, R)
        """
        nx, ny, nz = self.grid_res
        grid = V.reshape(nx, ny, nz, 3)
        if resolution is not None and (resolution != nx):
            xs = jnp.linspace(0, 1, resolution)
            coords = jnp.stack(jnp.meshgrid(xs, xs, xs, indexing='ij'), axis=-1)
            grid = trilinear(coords, grid, variance=squared_weights)
        return np.array(grid) 

    def prep_uncertainty_3d(self, hessian, covariance: jnp.ndarray, fov, min_uncertainty=-3, max_uncertainty=6, 
                            log_scale=True, resolution=64, mask=False, squared_weights=False):
        
        grid = self.upsample(covariance, resolution, squared_weights)
        print(grid.shape)
        sigma_vol = np.array(jnp.sqrt(jnp.sum(grid, axis=-1)))
        if log_scale:
            uncertainty = np.log10(sigma_vol + 1e-12)
        uncertainty = np.clip(uncertainty, min_uncertainty, max_uncertainty)
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-12)

        #occ = self.param_occupancy(fov)
        #active_mask = (occ > 0)
        #_mask = active_mask[...]
        #uncertainty = jnp.where(_mask, uncertainty, 0.)
        
        if mask:
            emission_estimate = bhnerf.network.sample_3d_grid(self.pred_apply, self.pred_params, fov=fov, resolution=resolution)
            eps = emission_estimate.max() * 0.02
            mask = (emission_estimate > eps)
            uncertainty = np.where(mask, uncertainty, 0.0)

        return uncertainty

    def render_uncertainty_3d_ipv(self, uncertainty, fov, view_kw=dict(azimuth=30, elevation=25, distance=2.8), 
        level_norm=(0.0, 0.3, 0.6, 0.9), opacity=(0.00, 0.10, 0.45, 0.85), output='', cmap="plasma"):
        import ipyvolume as ipv
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        ipv.figure()
        ipv.view(**view_kw)
        v = ipv.volshow(uncertainty, extent=[(-fov/2, fov/2)]*3, memorder="F", controls=True)
        cmap = cm.get_cmap(cmap)
        rgba = np.array([cmap(l) for l in level_norm], dtype="float32")
        rgba[:, 3] = opacity
        v.tf = ipv.TransferFunction(rgba=rgba, level=level_norm, opacity=opacity)
        ipv.show()

        fig, ax = plt.subplots(figsize=(4, .35))
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
        cb.set_label("normalized sigma")
        plt.show()
        
        if output:
            # TODO
            pass

    def propagate_positional_variance(self, sigma0_var: jnp.ndarray, t_frames, fov_M: float, R: int, spin: float, M: float = 1.0, 
        rot_axis=(0.0, 0.0, 1.0), chunk_pts: int = 32768, include_Omega_grad: bool = False, eps_reg: float = 1e-6) -> jnp.ndarray:
        """
        Rigid default (matches differentiating velocity_warp_coords w.r.t coords):
            Var_t(x) = 3 * σ0^2(w(x;t))

        If include_Omega_grad=True:
            Var_t(x) = σ0^2(w(x;t)) * ||J_w(x,t)^{-1}||_F^2
        """

        side = jnp.linspace(-fov_M/2, fov_M/2, R, dtype=jnp.float32)
        X, Y, Z = jnp.meshgrid(side, side, side, indexing='ij')
        pts = jnp.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        I3 = jnp.eye(3, dtype=jnp.float32)

        if hasattr(t_frames, "__len__") and len(t_frames) > 0 and hasattr(t_frames[0], "unit"):
            base_u = t_frames[0].unit
            tq = jnp.asarray([float((t - t_frames[0]).to_value(base_u)) for t in t_frames], dtype=jnp.float32)
            GMc3 = consts.GM_c3(consts.sgra_mass).to(base_u).value
            tM_all = tq / float(GMc3)
        else:
            tM_all = jnp.asarray(t_frames, dtype=jnp.float32)

        sgn = jnp.sign(spin + 1e-12)
        Ms = jnp.sqrt(M)
        def omega_of_r(r):
            return sgn * Ms / (r**1.5 + spin * Ms)

        def w_point(x: jnp.ndarray, tM: jnp.ndarray) -> jnp.ndarray:
            coords = x.reshape(3, 1, 1, 1)
            r = jnp.linalg.norm(x) + 1e-12
            Om = omega_of_r(r)
            wc = bhnerf.emission.velocity_warp_coords(
                coords=coords,
                Omega=Om,   
                t_frames=tM,
                t_start_obs=0.0,
                t_geos=0.0,
                t_injection=0.0,
                rot_axis=rot_axis,
                M=M,
                t_units=None,
                use_jax=True,
            )
            return wc.reshape(3,)

        grid_sigma = sigma0_var.astype(jnp.float32)
        def sample_sigma0(y_points: jnp.ndarray) -> jnp.ndarray:
            u = (y_points / (fov_M/2.0) + 1.0) * 0.5
            u = jnp.clip(u, 0.0, 1.0)
            return trilinear(u, grid_sigma[..., None])[..., 0]

        def _batch_map(fn, arr, *extra):
            outs = []
            for i in range(0, arr.shape[0], chunk_pts):
                sl = slice(i, min(i + chunk_pts, arr.shape[0]))
                outs.append(jax.vmap(fn)(arr[sl], *extra))
            return jnp.concatenate(outs, axis=0)

        v_warp = lambda xs, tM: _batch_map(lambda p: w_point(p, tM), xs)

        if include_Omega_grad:
            jac_w = jax.jit(jax.jacfwd(w_point, argnums=0))
            def stretch_at_point(x, tM):
                Jw = jac_w(x, tM)
                V  = jnp.linalg.solve(Jw + eps_reg*I3, I3) 
                return jnp.sum(V**2)
            v_stretch = lambda xs, tM: _batch_map(lambda p: stretch_at_point(p, tM), xs)

        vols = []
        for tM in tM_all:
            y_pts = v_warp(pts, tM)
            s0 = sample_sigma0(y_pts)
            if include_Omega_grad:
                stretch = v_stretch(pts, tM)
                var_t = (s0 * stretch).reshape(R, R, R)
            else:
                var_t = (3.0 * s0).reshape(R, R, R)
            vols.append(var_t)

        return jnp.stack(vols, axis=0)

    def build_emission_movie(self, t_frames, fov_M, R, spin, M=1.0):
        Omega3D = omega_grid_kepler(fov_M=fov_M, R=R, spin=spin, M=M)
        vols = []
        for t in t_frames:
            vol = bhnerf.network.sample_3d_grid(
                self.pred_apply,
                self.pred_params,
                t_frame=t, 
                t_start_obs=t_frames[0], 
                Omega=Omega3D,
                fov=fov_M,
                coords=None,
                resolution=R
            )
            vols.append(np.asarray(vol))
        return np.stack(vols, axis=0)
    
    def delta_theta(self, t_f, f_ref=0):
        GMc3 = bhnerf.constants.GM_c3(bhnerf.constants.sgra_mass).to(t_f.unit).value
        t_ref = self.t_frames[f_ref]
        dt = (t_f - t_ref).to_value(self.t_frames.unit)
        return dt / GMc3 * self.rt_args['Omega']

    def _make_fisher_batch_for_frame(self, f: int):
        """Returns a jitted, vmapped fisher diag function for a single frame f.
        Parameters:
        -----------
        f: (int) 
            frame index

        Returns:
        fisher_diag_ray_f: (JitWrapped) 
            The fisher gradient function for frame f, vmapped over rays.
        """
        
        fm_f = self.fm_per_frame[f]
        voxel_world = self.voxel_world
        #dtheta = self.delta_theta(self.t_frames[f], f_ref=0)
        #Rinv_f = bhnerf.utils.rotation_matrix([0, 0, -1], -dtheta, use_jax=True)

        def _render_one_ray_f(def_params, ray_idx):
            offsets = self.def_grid.apply({"params": def_params}, self.coords_unit)
            #ffsets_T = jnp.moveaxis(offsets_ref, -1, 0)
            #offsets_preT = jnp.sum(Rinv_f * offsets_T, axis=1)
            #offsets_pre = jnp.moveaxis(offsets_preT, 0, -1)

            coords_def = self.coords + offsets * self.voxel_world
            vis_f = fm_f(coords_def)
            return vis_f[ray_idx]

        def _fisher_diag_ray_f(ray_idx, sigma_k):

            if self.mode == 'eht':
                w = 1.0 / (sigma_k / jnp.sqrt(2.0))
                def im_fn(dp): return jnp.imag(_render_one_ray_f(dp, ray_idx))
                def re_fn(dp): return jnp.real(_render_one_ray_f(dp, ray_idx))
                g_re = flatten(jax.grad(re_fn)(self.def_params))
                g_im = flatten(jax.grad(im_fn)(self.def_params))
                row  = (w**2) * (g_re**2 + g_im**2)
                active = (jnp.max(g_re**2 + g_im**2) > 1e-20).astype(jnp.int32)
            elif self.mode == 'image':
                w = 1.0 / sigma_k
                def normal_fn(dp): return _render_one_ray_f(dp, ray_idx)
                g_im = flatten(jax.grad(normal_fn)(self.def_params))
                row  = w**2 * g_im**2
                active = (jnp.max(g_im) > 1e-20).astype(jnp.int32)
            return row, active

        return jax.jit(jax.vmap(_fisher_diag_ray_f, in_axes=(0, 0)))

    def compute_hessian_diag(self, sigma: jnp.ndarray, batch_size: int = 256):
        """
        Compute the diagonal of the Hessian matrix (MAP objective) across all frames

        Parameters:
        ----------
        sigma: jnp.ndarray
            shape (N_frames, Nvis), noise per visibility for each frame
        batch_size: int
            the number of rays to vmap over at once
        visibility: bool
            Whether image observations are fourier transformed (imag+real valued) or not (real valued)

        Returns:
        -------
        H: jnp.ndarray 
            Shape (P,) the diagonal of the hessian, where P is the number of parameters the deformation grid
        R_eff: int 
            the number of active rays (have non-zero gradient which intersect w/the bh volume)
        """
        H = jnp.zeros((self.P,), dtype=jnp.float32)
        R_eff = 0

        for f in tqdm(range(len(self.fm_per_frame)), desc='frame'):
            nvis_f = self.nvis_per_frame[f]
            if f % 2 == 0: 
                print('frame {} visibilities: {}'.format(f, nvis_f))
            
            _fisher_batch = self._make_fisher_batch_for_frame(f)
            for start in tqdm(range(0, nvis_f, batch_size), desc='iteration', leave=False):
                end = min(start + batch_size, nvis_f)
                ray_idx = jnp.arange(start, end)
                rows, active = _fisher_batch(ray_idx, self.sigma[f, ray_idx])
                
                H += jnp.sum(rows, axis=0)
                R_eff += int(jnp.sum(active))
        
        R_eff = max(R_eff, 1)
        return H, R_eff
    def prepare_quiver_vectors(self, V, stride=None, mask=None,
                           pct=(5, 99.5), gamma=0.9, eps=1e-12):
        nx, ny, nz = self.grid_res
        V_grid = V.reshape((nx, ny, nz, 3))
        sx, sy, sz = V_grid[..., 0], V_grid[..., 1], V_grid[..., 2]

        if stride is None:
            stride = (max(1,nx//32), max(1,ny//32), max(1,nz//32))
        L = np.sqrt(sx*sx + sy*sy + sz*sz) + eps

        M = np.ones_like(L, dtype=bool)
        if mask is not None:
            M &= np.asarray(mask) > 0
        Ms = M[::stride[0], ::stride[1], ::stride[2]]
        Ls = L[::stride[0], ::stride[1], ::stride[2]][Ms]

        logLs = np.log1p(Ls)
        lo, hi = np.nanpercentile(logLs, pct)
        if not np.isfinite(lo): lo = np.nanmin(logLs)
        if not np.isfinite(hi) or hi <= lo: hi = lo + 1e-6

        S = np.clip((np.log1p(L) - lo) / (hi - lo), 0.0, 1.0) ** gamma

        Ux = (sx / L) * S
        Uy = (sy / L) * S
        Uz = (sz / L) * S

        Sd = S[::stride[0], ::stride[1], ::stride[2]][Ms]
        print("S percentiles on drawn subset:", np.nanpercentile(Sd, [0,25,50,75,90,99,100]))
        return Ux, Uy, Uz

def prepare_unc_for_render(sigma_vol, mask, min_uncertainty=-3.0, max_uncertainty=6.0):
    """
    Log scales the uncertainty, clips it between min and max uncertainty, and masks it where the network has 0 emission.

    Args:
        sigma_vol: per voxel uncertainty (3, res, res, res)
    Returns:
        the clipped, prepared, and masked uncertainty 
    """
    nt = sigma_vol.shape[0]
    outs = []
    
    for i in range(nt):
        uncertainty = np.log10(sigma_vol[i] + 1e-12)
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-12)
        uncertainty = np.where(mask[i], uncertainty, 0.0)
        outs.append(uncertainty.astype(np.float32))
    return np.stack(outs, axis=0)

def build_dynamic_mask(movie, relative_threshold=0.1):
    per_frame_max = movie.max(axis=(1, 2, 3), keepdims=True)
    return (movie > (per_frame_max*relative_threshold))

def omega_grid_kepler(fov_M: float, R: int, spin: float, M: float = 1.0) -> np.ndarray:
    """Ω(r) on a regular (R,R,R) cube in geometric units."""
    side = np.linspace(-fov_M/2, fov_M/2, R, dtype=np.float32)
    X, Y, Z = np.meshgrid(side, side, side, indexing='ij')
    r = np.sqrt(X*X + Y*Y + Z*Z) + 1e-12
    sgn = np.sign(spin + 1e-12)
    Ms  = np.sqrt(M).astype(np.float32)
    return (sgn * Ms / (r**1.5 + spin * Ms)).astype(np.float32)  

