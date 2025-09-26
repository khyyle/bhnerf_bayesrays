import bhnerf
import bhnerf.constants as consts
import numpy as np
import os
from datetime import datetime
from astropy import units
import ehtim as eh
from bhnerf.optimization import LogFn
import sys
import xarray as xr

def shift_z_xr(da: xr.DataArray, dz: float) -> xr.DataArray:
    if abs(dz) < 1e-12: 
        return da
    x, y, z = np.meshgrid(da.x, da.y, da.z, indexing='ij')
    coords = np.stack([x, y, z-dz], axis=-1)
    data = bhnerf.emission.interpolate_coords(da, coords)  
    return xr.DataArray(data, coords=da.coords, dims=da.dims, attrs=da.attrs)

def multi_hotspot_emission_xr(spin, fov_M, resolution=(64,64,64), orbit_radius=5.5, blobs=(), flux_scale=0.1):
    """
    blobs: list of dicts with keys:
      ang_deg (required), amp (default 1.0), std (scalar or (sx,sy,sz), default 0.7), dz (default 0.0)
    All built with bhnerf.emission.generate_hotspot_xr; optional dz uses interpolate_coords.
    """
    r_isco = bhnerf.constants.isco_pro(spin)
    total = None
    amp_sum = 0.0

    for b in blobs:
        ang  = np.deg2rad(b['ang_deg'])
        amp  = float(b.get('amp', 1.0))
        std  = b.get('std', 0.7)
        dz   = float(b.get('dz', 0.0))

        da = bhnerf.emission.generate_hotspot_xr(
            resolution=resolution,
            rot_axis=[0.0, 0.0, 1.0],
            rot_angle=ang,
            orbit_radius=float(orbit_radius),
            std=std, r_isco=r_isco,
            fov=(fov_M, 'GM/c^2'),
            normalize=True,
        )

        if abs(dz) > 0.0:
            da = shift_z_xr(da, dz)

        total = da*amp if total is None else (total + da*amp)
        amp_sum += amp

    return total * (flux_scale / max(amp_sum, 1e-12))
fov_M = 16.0
spin = 0.2
inclination = np.deg2rad(60.0)
nt = 64

array = 'ngEHT'
flux_scale = 0.1
tstart = 2.0 * units.hour
tstop = tstart + 40.0 * units.min

geos = bhnerf.kgeo.image_plane_geos(
    spin, inclination, 
    num_alpha=64, num_beta=64,
    alpha_range=[-fov_M/2, fov_M/2], 
    beta_range=[-fov_M/2, fov_M/2]
)
Omega = np.sign(spin + np.finfo(float).eps) * np.sqrt(geos.M) / (geos.r**(3/2) + geos.spin * np.sqrt(geos.M))
t_injection = -float(geos.r_o)

three_spot = [
    {'ang_deg': 25,  'amp': 1.00, 'std': (0.7, 0.7, 0.7), 'dz': +2},  # bright, upper-right-ish
    {'ang_deg': 200, 'amp': 0.85, 'std': (0.85,0.65,0.7), 'dz': -4},  # close pair, side A
    {'ang_deg': 250, 'amp': 0.80, 'std': (0.85,0.65,0.7), 'dz': -5},  # close pair, side A
]

emission_0 = multi_hotspot_emission_xr(
    spin=spin, 
    fov_M=fov_M, 
    orbit_radius=5.5,
    blobs=three_spot, 
    flux_scale=0.1
)
obs_params = {
    'mjd': 57581,
    'timetype': 'GMST',
    'nt': nt,
    'tstart': tstart.to('hr').value,
    'tstop': tstop.to('hr').value,
    'tint': 30.0,
    'array': eh.array.load_txt('../eht_arrays/{}.txt'.format(array))
}
obs_empty = bhnerf.observation.empty_eht_obs(**obs_params)
fov_rad = (fov_M * consts.GM_c2(consts.sgra_mass) / consts.sgra_distance.to('m')) * units.rad
psize = fov_rad.value / geos.alpha.size
obs_args = {'psize': psize, 'ra': obs_empty.ra, 'dec': obs_empty.dec, 'rf': obs_empty.rf, 'mjd': obs_empty.mjd}

t_frames = np.linspace(tstart, tstop, nt)
image_plane = bhnerf.emission.image_plane_dynamics(emission_0, geos, Omega, t_frames, t_injection)
movie = eh.movie.Movie(image_plane, times=t_frames.value, **obs_args)
obs = bhnerf.observation.observe_same(movie, obs_empty, ttype='direct', seed=None)

"""
Optimize network paremters to recover the 3D emission (as a continuous function) from observations 
Note that logging is done using tensorboardX. To view the tensorboard (from the main directory):
    `tensorboard --logdir runs`
"""
batchsize = 6
z_width = fov_M            # maximum disk width [M]
rmax = (fov_M / 2) + 2     # maximum recovery radius 
rmin = float(geos.r.min()) # minimum recovery radius

SEED = int(sys.argv[1])
hparams = {'num_iters': 8000, 'lr_init': 1e-4, 'lr_final': 1e-6, 'seed': SEED}

# Checkpointing
current_time = datetime.now().strftime('%Y-%m-%d.%H:%M:%S')
runname = 'tutorial4/recovery.vis.{}'.format(current_time)

# Observation parameters 
chisqdata = eh.imaging.imager_utils.chisqdata_vis
train_step = bhnerf.optimization.TrainStep.eht(t_frames, obs, movie.fovx(), movie.xdim, chisqdata)

# Optimization
predictor = bhnerf.network.NeRF_Predictor(rmax, rmin, rmax, z_width)
raytracing_args = bhnerf.network.raytracing_args(geos, Omega, t_injection, t_frames[0])
optimizer = bhnerf.optimization.Optimizer(hparams, predictor, raytracing_args, checkpoint_dir='../checkpoints/ensemble_3obs/seed{}_{}'.format(SEED, runname))
optimizer.run(batchsize, train_step, raytracing_args)