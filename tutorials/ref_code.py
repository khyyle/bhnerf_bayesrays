# NOTE: attempt 1 with coordinates. originally the uncertainty blob was in the corners of the 2D map and the 3d reconstruction was just a 
# flat blob.
self.coords_unit = (self.coords/(fov/2.0) + 1.0) * 0.5
self.coords_unit = jnp.clip(self.coords_unit, 0.0, 1.0 - 1e-7)
xyz_half = jnp.max(jnp.abs(self.coords), axis=(0,1,2))
self.voxel_world = (2.0 * xyz_half) / (jnp.array(self.grid_res) - 1.0)

# NOTE: attempt 2. this assumes all coords are within z_width. If they aren't they get clipped to [0,1] range.
# I suspect this is why the uncertainty looked like a peanut shape using these unit_coords

#extents = jnp.array([fov/2, fov/2, z_width], dtype=jnp.float32)
#self.coords_unit = (self.coords / extents + 1.0) * 0.5
#self.coords_unit = jnp.clip(self.coords_unit, 0.0, 1.0-1e-7)

# NOTE: attempt 3. here, we fit our 'coord box' around the actual ray cloud, avoiding clipping
# this does not work and returns a flat / empty uncertainty map
#xyz_half = jnp.max(jnp.abs(self.coords), axis=(0,1,2,3))
#margin = 1.02
#extents = xyz_half * margin
#self.coords_unit = (self.coords / extents + 1.0) * 0.5
#self.coords_unit = jnp.clip(self.coords_unit, 0.0, 1.0 - 1e-7)

#NOTE: attempt 3.5
#self.coords_unit, center, half = make_coords_unit(self.coords, margin=1.01, eps=1e-6)

# NOTE: attempt 4 with adjusting the coordinates. still doesnt work
#self.coords_unit, center, half = make_coords_unit_weighted(self.coords, self.rt_args['g'], self.rt_args['dtau'], keep=0.995, margin=1.07, eps=1e-6)
#R = jnp.array(self.grid_res, dtype=jnp.float32)
#self.voxel_world = (2.0 * half) / (R - 1.0)

# NOTE: attempt 5
'''
def build_unit_box_weighted(coords, g, dtau, Sigma=None,
                    qlo=0.02, qhi=0.98, margin=1.02):
    w = g * dtau if (Sigma is None or jnp.isscalar(Sigma)) else (g * dtau * Sigma)
    w = w.ravel()
    support = jnp.sum(w > 0) / (w.size + 1e-12)

    # Fallback to unweighted min/max if weights are too concentrated
    if support < 1e-3:   # tune threshold if needed
        reduce_axes = tuple(range(coords.ndim - 1))
        xyz_min = jnp.min(coords, axis=reduce_axes)
        xyz_max = jnp.max(coords, axis=reduce_axes)
        center  = 0.5 * (xyz_min + xyz_max)
        half    = jnp.maximum(0.5 * (xyz_max - xyz_min), 1e-6) * margin
    else:
        def _wq(x, w, q):
            idx = jnp.argsort(x); x = x[idx]; w = w[idx]
            c = jnp.cumsum(w); c = c / (c[-1] + 1e-12)
            return jnp.interp(q, c, x)

        x = coords[...,0].ravel(); y = coords[...,1].ravel(); z = coords[...,2].ravel()
        w = w / (jnp.sum(w) + 1e-12)
        lo = jnp.array([_wq(x,w,qlo), _wq(y,w,qlo), _wq(z,w,qlo)])
        hi = jnp.array([_wq(x,w,qhi), _wq(y,w,qhi), _wq(z,w,qhi)])
        center = 0.5*(lo+hi); half = jnp.maximum(0.5*(hi-lo), 1e-6) * margin

    u = ((coords - center) / half) * 0.5 + 0.5
    u = jnp.clip(u, 1e-6, 1.0 - 1e-6)
    return u, center, half


self.coords_unit, center, half = build_unit_box_weighted(
    self.coords, self.rt_args['g'], self.rt_args['dtau'],
    Sigma=self.rt_args.get('Sigma', None),
    qlo=0.01, qhi=0.99, margin=1.02
))


R = jnp.array(self.grid_res, dtype=jnp.float32)
self.voxel_world = (2.0 * half) / (R - 1.0)
'''

# NOTE: attempt 6
def make_coords_unit_like_emission_grid(coords, fov_M, grid_res):
    fov = jnp.array([fov_M, fov_M, fov_M], dtype=jnp.float32)
    npix = jnp.array(grid_res, dtype=jnp.float32)
    img_idx = bhnerf.utils.world_to_image_coords(coords, fov=fov, npix=npix, use_jax=True)
    u = img_idx / (npix - 1.0)
    u = jnp.clip(u, 1e-6, 1.0 - 1e-6)
    voxel_world = fov / (npix - 1.0)
    return u, voxel_world
#self.coords_unit, self.voxel_world = make_coords_unit_like_emission_grid(
#    self.coords, fov_M=fov, grid_res=self.grid_res
#)

"""
def _render_one(self, def_params, k):
        offsets = self.def_grid.apply({"params": def_params}, self.coords_unit)
        coords_deformed = self.coords + offsets * self.voxel_world
        vis = self.forward_model(coords_deformed)
        return vis[k]
    
    def _fisher_diag_row(self, k, sigma_k):
        def re_fn(def_p): return jnp.real(self._render_one(def_p, k))
        def im_fn(def_p): return jnp.imag(self._render_one(def_p, k))

        g_re = flatten(jax.grad(re_fn)(self.def_params))
        g_im = flatten(jax.grad(im_fn)(self.def_params))

        factor = 1.0 / (sigma_k / jnp.sqrt(2.0))
        gi2 = g_re**2 + g_im**2

        row = factor**2 * (g_re**2 + g_im**2)
        active = (jnp.max(gi2) > 1e-20).astype(jnp.int32)

        return row, active
    
    def compute_hessian_diag(self, sigma: jnp.ndarray, batch_size: int = 256):
        sigma : (N_vis,)  noise per visibility (same epoch)
        H = jnp.zeros((self.P,), dtype=jnp.float32)
        R_eff = 0

        _fisher_batch = jax.jit(jax.vmap(self._fisher_diag_row, in_axes=(0,0)))
        for start in tqdm(range(0, self.nvis, batch_size), desc='iteration'):
            end = min(start + batch_size, self.nvis)
            idx = jnp.arange(start, end)
            rows, active = _fisher_batch(idx, sigma[idx])
            H += jnp.sum(rows, axis=0)
            R_eff += int(jnp.sum(active))
        
        R_eff = max(R_eff, 1)
        return H, R_eff
"""
"""
#NOTE: attempt 7:
def make_coords_unit_emission_constrained(coords, fov_xy, grid_res,
                                        g=None, dtau=None,
                                        qlo=0.01, qhi=0.99,
                                        margin=1.10,
                                        half_min=(1.0, 1.0, 0.75)):  # M units
    """
    - x,y use the known ±fov_xy/2 cube.
    - z extent/center are estimated from the subset of samples with |x|,|y| <= fov_xy/2,
    using weighted quantiles (g*dtau) if provided.
    """
    import numpy as np
    coords_np = np.asarray(coords)            # (Nx,Ny,Ns,3)
    x, y, z = [coords_np[..., i] for i in range(3)]

    # restrict to the x–y footprint of the emission cube
    in_xy = (np.abs(x) <= fov_xy/2) & (np.abs(y) <= fov_xy/2)
    z_sel = z[in_xy]
    if (g is not None) and (dtau is not None):
        w = np.asarray(g * dtau)[in_xy].reshape(-1)
        w = np.clip(w, 0, None); 
        if w.sum() > 0: w = w / (w.sum() + 1e-12)
    else:
        w = None

    # robust z center/half from quantiles
    if z_sel.size > 0:
        z_flat = z_sel.reshape(-1)
        if (w is not None) and (w.size == z_flat.size):
            idx = np.argsort(z_flat)
            z_sorted = z_flat[idx]; w_sorted = w[idx]
            cdf = np.cumsum(w_sorted); cdf /= (cdf[-1] + 1e-12)
            z_lo = np.interp(qlo, cdf, z_sorted)
            z_hi = np.interp(qhi, cdf, z_sorted)
        else:
            z_lo, z_hi = np.quantile(z_flat, [qlo, qhi])
        z_ctr  = 0.5*(z_lo + z_hi)
        z_half = max(0.5*(z_hi - z_lo)*margin, half_min[2])
    else:
        # fallback
        z_ctr, z_half = 0.0, max(fov_xy/2, half_min[2])

    # x,y center/half are fixed by the known FOV
    x_ctr = y_ctr = 0.0
    x_half = max(fov_xy/2, half_min[0])
    y_half = max(fov_xy/2, half_min[1])

    ctr  = np.array([x_ctr, y_ctr, z_ctr], dtype=np.float32)
    half = np.array([x_half, y_half, z_half], dtype=np.float32)

    u = ((coords_np - ctr) / half) * 0.5 + 0.5
    u = np.clip(u, 1e-6, 1.0 - 1e-6)

    R = np.array(grid_res, dtype=np.float32)
    voxel_world = (2.0 * half) / (R - 1.0)
    return jnp.asarray(u), jnp.asarray(voxel_world), jnp.asarray(ctr), jnp.asarray(half)

self.coords_unit, self.voxel_world, self.box_center, self.box_half = \
    make_coords_unit_emission_constrained(self.coords, fov_xy=float(fov),
                                        grid_res=self.grid_res,
                                        g=self.rt_args.get('g', None),
                                        dtau=self.rt_args.get('dtau', None))

        
def _q(a): 
    return np.array(np.quantile(np.asarray(a).ravel(), [0.01,0.5,0.99]))
print("u_z quantiles (in all points):", _q(self.coords_unit[...,2]))
print("box_center, box_half:", np.array(self.box_center), np.array(self.box_half))

def make_coords_unit_weighted(coords, g, dtau, keep=0.99, margin=1.01, eps=1e-6):
    weights = g * dtau
    reduce_axes = tuple(range(coords.ndim - 1))
    thr = jnp.quantile(weights, 1.0 - keep)
    mask = weights >= thr
    masked = jnp.where(mask[..., None], coords, jnp.nan)

    xyz_min = jnp.nanmin(masked, axis=reduce_axes)
    xyz_max = jnp.nanmax(masked, axis=reduce_axes)
    center = 0.5 * (xyz_max + xyz_min)
    half = jnp.maximum(0.5 * (xyz_max - xyz_min), eps) * margin

    u = (coords - center) / half
    u = (u + 1.0) * 0.5
    u = jnp.clip(u, eps, 1.0 - eps)
    return u, center, half

def make_coords_unit(coords, margin=1.01, eps=1e-6):
    reduce_axes = tuple(range(coords.ndim - 1))
    xyz_min = jnp.min(coords, axis=reduce_axes)
    xyz_max = jnp.max(coords, axis=reduce_axes)
    center = 0.5 * (xyz_max + xyz_min)
    half = 0.5 * (xyz_max - xyz_min)
    half = jnp.maximum(half, eps) * margin

    u = (coords - center) / half
    u = (u + 1.0) * 0.5
    u = jnp.clip(u, eps, 1.0 - eps)

    return u, center, half
"""

"""def sigma_volume(V, upres=64):
            """upsample and normalize the uncertainty map"""
            nx, ny, nz = self.grid_res
            grid = V.reshape((nx, ny, nz, 3))

            if upres > nx:
                print('upsampling')
                xs = jnp.linspace(0, 1, upres)
                ys = jnp.linspace(0, 1, upres)
                zs = jnp.linspace(0, 1, upres)

                coords = jnp.stack(jnp.meshgrid(xs, ys, zs, indexing='ij'), axis=-1)
                grid = trilinear(coords, grid, variance=squared_weights)
            return np.array(jnp.sqrt(jnp.sum(grid, axis=-1)))"""