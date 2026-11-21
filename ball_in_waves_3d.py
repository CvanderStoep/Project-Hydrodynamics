"""
Drijvende bal in golven - 3D

Fysica
------
Water  2D Boussinesq-golfvergelijkingen (Peregrine, vlakke bodem) in x en y:
           d(eta)/dt + div(H u) = bron                         H = h + eta
           (1 - h^2/3 grad div) du/dt = -g grad(eta) - (u.grad)u
       Dispersie zorgt dat golven met kh ~ 1 de juiste snelheid hebben.
       Gestaggerd (Arakawa C) rooster, forward-backward tijdstap. De
       dispersieterm is een gekoppeld, symmetrisch positief-definiet systeem
       voor (u_t, v_t); dat wordt een keer gefactoriseerd (sparse LU).
       Links een golfmaker (zuiger), rechts en aan de zijkanten absorberende
       sponge-lagen (geen reflecties).

Bal    Starre bol met 3 translatievrijheidsgraden (x, y, z). Krachten:
         - zwaartekracht
         - druk op het natte oppervlak, numeriek geintegreerd over de bol:
           hydrostatisch (Archimedes + golfhelling) plus de niet-hydrostatische
           correctie uit dezelfde Boussinesq-benadering
         - toegevoegde massa (Morison)
         - kwadratische weerstand t.o.v. de lokale watersnelheid
         - zachte horizontale verankering, zodat de bal in beeld blijft
       Terugkoppeling naar het water: als de bal beweegt verplaatst hij water.
       Dat volume komt als bron in de massabalans terecht op de plek van de
       waterlijn, zodat de bal zelf ringvormige golven uitzendt (lineaire
       benadering van de stralingsgolven bij op-en-neer bewegen).

Gebruik
-------
    pip install numpy scipy matplotlib pyvista
    python ball_in_waves_3d.py                       # live 3D-animatie (muis: draaien/zoomen)
    python ball_in_waves_3d.py --density 250
    python ball_in_waves_3d.py --amp 0.15 --period 2.5
    python ball_in_waves_3d.py --zscale 5            # golven in beeld sterker uitvergroten (standaard 3)
    python ball_in_waves_3d.py --warmup 0            # ook het binnenlopen van de golffront zien
    python ball_in_waves_3d.py --amp 0 --drop 0.8    # bal laten vallen in stilstaand water: ringen
    python ball_in_waves_3d.py --radiated            # kleur alleen de golven die de bal zelf maakt
    Toetsen in het venster:  d = bal opnieuw laten vallen,  k = kleur wisselen (totaal / alleen bal)
    python ball_in_waves_3d.py --save golven3d.mp4 --seconds 20   # mp4 vraagt imageio-ffmpeg
    python ball_in_waves_3d.py --selftest            # fysica-controles zonder beeld
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.sparse.linalg import splu

G = 9.81          # m/s^2
RHO_W = 1000.0    # kg/m^3


@dataclass
class Params:
    # bassin
    L: float = 28.0            # lengte (x) [m]
    W: float = 16.0            # breedte (y) [m]
    dx: float = 0.2            # roosterafstand [m]
    depth: float = 1.0         # waterdiepte [m]
    # golven
    wave_amp: float = 0.10     # amplitude golfmaker [m]
    wave_period: float = 3.0   # periode [s]
    sponge_x: float = 7.0      # absorberende zone rechts [m]
    sponge_y: float = 4.0      # absorberende zones aan de zijkanten [m]
    sponge_strength: float = 4.0
    nonlinear: bool = True
    # bal
    radius: float = 0.5        # [m]
    density: float = 500.0     # [kg/m^3]; < 1000 = drijft
    x0: float = 10.0
    y0: Optional[float] = None   # None = midden van het bassin
    ca: float = 0.5            # toegevoegde-massacoefficient bol
    cd: float = 0.5            # weerstandscoefficient bol
    mooring_period: float = 20.0  # zachte horizontale veer [s]; 0 = vrij drijvend
    two_way: bool = True       # bal zendt zelf golven uit
    nonhydro: bool = True      # niet-hydrostatische drukcorrectie op de bal
    source_width: float = 2.0  # breedte van de golfbron, in balstralen
    drop: float = 0.0          # bal zoveel meter boven evenwicht loslaten [m]
    # numeriek
    dt: float = 0.02
    steps_per_frame: int = 2


def bilerp(f, ox, oy, dx, dy, px, py):
    """Bilineaire interpolatie van een 2D-veld op positie(s) (px, py)."""
    px = np.asarray(px, float)
    py = np.asarray(py, float)
    fx = (px - ox) / dx
    fy = (py - oy) / dy
    i = np.clip(np.floor(fx).astype(int), 0, f.shape[0] - 2)
    j = np.clip(np.floor(fy).astype(int), 0, f.shape[1] - 2)
    tx = np.clip(fx - i, 0.0, 1.0)
    ty = np.clip(fy - j, 0.0, 1.0)
    return ((1 - tx) * (1 - ty) * f[i, j] + tx * (1 - ty) * f[i + 1, j]
            + (1 - tx) * ty * f[i, j + 1] + tx * ty * f[i + 1, j + 1])


class Water:
    """2D Boussinesq-water. eta in celmiddens, u op x-randen, v op y-randen."""

    def __init__(self, p: Params):
        self.p = p
        nx = self.nx = int(round(p.L / p.dx))
        ny = self.ny = int(round(p.W / p.dx))
        self.dx, self.dy = p.L / nx, p.W / ny
        self.x = (np.arange(nx) + 0.5) * self.dx
        self.y = (np.arange(ny) + 0.5) * self.dy
        self.xf = np.arange(nx + 1) * self.dx
        self.yf = np.arange(ny + 1) * self.dy
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        self.eta = np.zeros((nx, ny))
        self.u = np.zeros((nx + 1, ny))
        self.v = np.zeros((nx, ny + 1))
        self.au = np.zeros_like(self.u)          # versnellingen u_t, v_t
        self.av = np.zeros_like(self.v)
        self.detadt = np.zeros((nx, ny))         # verticale snelheid (stroming)
        self.d2etadt2 = np.zeros((nx, ny))       # verticale versnelling (stroming)
        self.t = 0.0

        self.c = np.sqrt(G * p.depth)
        self.omega = 2 * np.pi / p.wave_period
        k = brentq(lambda k: G * k * k * p.depth / (1 + (k * p.depth) ** 2 / 3)
                   - self.omega ** 2, 1e-6, 50.0)
        self.k = k
        self.c_phase = self.omega / k

        # Absorberende lagen. Rechts (x): dempt alles. Aan de zijkanten (y): dempt
        # alleen de dwarssnelheid v. Een vlakke golf heeft v = 0 en blijft dus
        # ongestoord; zijwaarts lopende golven (van de bal) worden wel gedempt.
        def sponge_x(X):
            return p.sponge_strength * np.clip((X - (p.L - p.sponge_x)) / p.sponge_x, 0, 1) ** 2

        def sponge_y(Y):
            if p.sponge_y <= 0:
                return 0.0 * Y
            return p.sponge_strength * (np.clip((p.sponge_y - Y) / p.sponge_y, 0, 1) ** 2
                                        + np.clip((Y - (p.W - p.sponge_y)) / p.sponge_y, 0, 1) ** 2)

        self.sig_c = sponge_x(self.X)
        Xu, Yu = np.meshgrid(self.xf, self.y, indexing="ij")
        Xv, Yv = np.meshgrid(self.x, self.yf, indexing="ij")
        self.sig_u = sponge_x(Xu)[1:-1]
        self.sig_v = (sponge_x(Xv) + 2.0 * sponge_y(Yv))[:, 1:-1]
        self._build_operator()

    # --- (1 + beta Di^T Di) a = rhs, vooraf gefactoriseerd ----------------
    def _build_operator(self):
        nx, ny, dx, dy = self.nx, self.ny, self.dx, self.dy
        beta = self.p.depth ** 2 / 3.0
        Dx = sp.diags([-np.ones(nx), np.ones(nx)], [0, 1], shape=(nx, nx + 1))
        Dy = sp.diags([-np.ones(ny), np.ones(ny)], [0, 1], shape=(ny, ny + 1))
        D_u = (sp.kron(Dx, sp.identity(ny)) / dx).tocsc()
        D_v = (sp.kron(sp.identity(nx), Dy) / dy).tocsc()
        iu = np.arange((nx + 1) * ny).reshape(nx + 1, ny)
        iv = np.arange(nx * (ny + 1)).reshape(nx, ny + 1)
        cols_u, cols_v = iu[1:-1, :].ravel(), iv[:, 1:-1].ravel()
        Di = sp.hstack([D_u[:, cols_u], D_v[:, cols_v]]).tocsr()
        A = (sp.identity(Di.shape[1]) + beta * (Di.T @ Di)).tocsc()
        self.lu = splu(A)
        left = D_u[:, iu[0, :]]
        self.bvec = beta * (Di.T @ np.asarray(left.sum(axis=1)).ravel())
        self.n_u = cols_u.size

    # --- golfmaker --------------------------------------------------------
    def paddle_u(self, t: float) -> float:
        p = self.p
        ramp = 0.5 * (1 - np.cos(np.pi * min(t / (2 * p.wave_period), 1.0)))
        return ramp * p.wave_amp * self.c_phase / p.depth * np.sin(self.omega * t)

    # --- advectie (u.grad)u, upwind ---------------------------------------
    def _advection(self, u, v):
        dx, dy = self.dx, self.dy
        uc = u[1:-1]
        ux = np.where(uc > 0, uc - u[:-2], u[2:] - uc) / dx
        vbar = 0.25 * (v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:])
        upd = np.pad(u, ((0, 0), (1, 1)), mode="edge")[1:-1]
        uy = np.where(vbar > 0, uc - upd[:, :-2], upd[:, 2:] - uc) / dy
        adv_u = uc * ux + vbar * uy

        vc = v[:, 1:-1]
        vy = np.where(vc > 0, vc - v[:, :-2], v[:, 2:] - vc) / dy
        ubar = 0.25 * (u[:-1, :-1] + u[1:, :-1] + u[:-1, 1:] + u[1:, 1:])
        vpd = np.pad(v, ((1, 1), (0, 0)), mode="edge")[:, 1:-1]
        vx = np.where(ubar > 0, vc - vpd[:-2], vpd[2:] - vc) / dx
        adv_v = ubar * vx + vc * vy
        return adv_u, adv_v

    # --- tijdstap ----------------------------------------------------------
    def step(self, dt: float, source: Optional[np.ndarray] = None) -> None:
        p, dx, dy, h = self.p, self.dx, self.dy, self.p.depth
        eta, u, v = self.eta, self.u, self.v

        # 1. massabehoud
        if p.nonlinear:
            Hu = np.empty_like(u)
            Hu[1:-1] = h + 0.5 * (eta[:-1] + eta[1:])
            Hu[0], Hu[-1] = h + eta[0], h + eta[-1]
            Hv = np.empty_like(v)
            Hv[:, 1:-1] = h + 0.5 * (eta[:, :-1] + eta[:, 1:])
            Hv[:, 0], Hv[:, -1] = h + eta[:, 0], h + eta[:, -1]
            Fu, Fv = Hu * u, Hv * v
        else:
            Fu, Fv = h * u, h * v
        tend = -(Fu[1:] - Fu[:-1]) / dx - (Fv[:, 1:] - Fv[:, :-1]) / dy
        self.d2etadt2 = (tend - self.detadt) / dt
        self.detadt = tend
        new_eta = eta + dt * (tend - self.sig_c * eta)
        if source is not None:
            new_eta += dt * source
        self.eta = eta = new_eta

        # 2. impuls met dispersie
        rhs_u = -G * (eta[1:] - eta[:-1]) / dx - self.sig_u * u[1:-1]
        rhs_v = -G * (eta[:, 1:] - eta[:, :-1]) / dy - self.sig_v * v[:, 1:-1]
        if p.nonlinear:
            adv_u, adv_v = self._advection(u, v)
            rhs_u -= adv_u
            rhs_v -= adv_v
        u0_new = self.paddle_u(self.t + dt)
        a0 = (u0_new - u[0, 0]) / dt
        rhs = np.concatenate([rhs_u.ravel(), rhs_v.ravel()]) - self.bvec * a0
        a = self.lu.solve(rhs)
        au = a[:self.n_u].reshape(self.nx - 1, self.ny)
        av = a[self.n_u:].reshape(self.nx, self.ny - 1)
        u[1:-1] += dt * au
        v[:, 1:-1] += dt * av
        u[0, :], u[-1, :] = u0_new, 0.0
        self.au[1:-1], self.au[0] = au, a0
        self.av[:, 1:-1] = av
        self.t += dt

    # --- interpolatie op positie (x, y) -----------------------------------
    def eta_at(self, x, y):  return bilerp(self.eta, self.dx / 2, self.dy / 2, self.dx, self.dy, x, y)
    def u_at(self, x, y):    return bilerp(self.u, 0.0, self.dy / 2, self.dx, self.dy, x, y)
    def v_at(self, x, y):    return bilerp(self.v, self.dx / 2, 0.0, self.dx, self.dy, x, y)
    def w_at(self, x, y):    return bilerp(self.detadt, self.dx / 2, self.dy / 2, self.dx, self.dy, x, y)
    def ax_at(self, x, y):   return bilerp(self.au, 0.0, self.dy / 2, self.dx, self.dy, x, y)
    def ay_at(self, x, y):   return bilerp(self.av, self.dx / 2, 0.0, self.dx, self.dy, x, y)
    def az_at(self, x, y):   return bilerp(self.d2etadt2, self.dx / 2, self.dy / 2, self.dx, self.dy, x, y)


class Ball:
    """Starre bol; druk geintegreerd over het oppervlak, toegevoegde massa, weerstand."""

    def __init__(self, p: Params):
        self.p = p
        self.R = R = p.radius
        self.volume = 4.0 / 3.0 * np.pi * R ** 3
        self.mass = p.density * self.volume
        y0 = p.W / 2 if p.y0 is None else p.y0
        # evenwichtsdiepgang voor stilstaand water (waterlijn op z = 0)
        rel = np.clip(p.density / RHO_W, 1e-6, 1 - 1e-6)
        hc = brentq(lambda hc: np.pi * hc * hc * (3 * R - hc) / 3 - rel * self.volume,
                    1e-9, 2 * R - 1e-9)
        self.pos = np.array([p.x0, y0, R - hc])
        self.z_eq = float(R - hc)                  # evenwichtshoogte van het middelpunt
        self.home = np.array([p.x0, y0])
        self.vel = np.zeros(3)
        # integratiepunten: ringen van gelijk oppervlak, gelijkmatig in azimut
        # (in stilstaand water is de horizontale kracht dan exact nul)
        n_lat, n_az = 60, 32
        cz = 1 - 2 * (np.arange(n_lat) + 0.5) / n_lat
        az = 2 * np.pi * (np.arange(n_az) + 0.5) / n_az
        CZ, AZ = np.meshgrid(cz, az, indexing="ij")
        sz = np.sqrt(1 - CZ ** 2)
        self.nrm = np.column_stack([(sz * np.cos(AZ)).ravel(), (sz * np.sin(AZ)).ravel(),
                                    CZ.ravel()])
        self.dA = 4 * np.pi * R ** 2 / (n_lat * n_az)
        # afgeleide grootheden, bijgewerkt in forces()
        self.frac = 0.0
        self.V_sub = 0.0
        self.A_wp = 0.0
        self.r_wp = 0.0

    def forces(self, w: Water, amb: Water):
        """w = totaal waterveld (inclusief golven van de bal zelf),
        amb = ongestoorde golf (zonder bal) voor stroming/versnelling bij de bal."""
        p, R, h = self.p, self.R, self.p.depth
        pos, vel = self.pos, self.vel
        ya = 0.5 * amb.p.W                      # het ongestoorde veld is uniform in y

        # druk op de bol: hydrostatisch (totaal veld) + niet-hydrostatische
        # Boussinesq-correctie uit de ongestoorde golf
        P = pos + R * self.nrm
        eta = w.eta_at(P[:, 0], P[:, 1])
        wet = eta > P[:, 2]
        zb = np.clip(P[:, 2] + h, 0.0, h)
        dyn = (float(amb.az_at(pos[0], ya)) * (h * h - zb * zb) / (2 * h)
               if p.nonhydro else 0.0)
        press = RHO_W * np.where(wet, G * (eta - P[:, 2]) + dyn, 0.0)
        press = np.maximum(press, 0.0)
        F = -(press @ self.nrm) * self.dA
        F[2] -= self.mass * G

        # waterlijn, ondergedompeld volume (ongestoorde waterstand bij de bal)
        eta_ref = float(amb.eta_at(pos[0], ya))
        dzw = eta_ref - pos[2]
        hc = float(np.clip(dzw + R, 0.0, 2 * R))
        self.V_sub = np.pi * hc * hc * (3 * R - hc) / 3
        self.frac = self.V_sub / self.volume
        self.r_wp = np.sqrt(max(R * R - dzw * dzw, 0.0)) if abs(dzw) < R else 0.0
        self.A_wp = np.pi * self.r_wp ** 2

        # toegevoegde massa (Morison): het deel van het water dat meebeweegt.
        # Verticaal wordt dat bij terugkoppeling al door de uitgezonden golven
        # (de bron) geleverd, dus daar tellen we het niet nog eens mee.
        m_a = p.ca * RHO_W * self.V_sub * np.array([1.0, 1.0, 0.0 if p.two_way else 1.0])
        a_w = np.array([amb.ax_at(pos[0], ya), amb.ay_at(pos[0], ya),
                        amb.az_at(pos[0], ya)], float)
        F += m_a * a_w

        # kwadratische weerstand t.o.v. de lokale watersnelheid
        u_w = np.array([amb.u_at(pos[0], ya), amb.v_at(pos[0], ya),
                        amb.w_at(pos[0], ya)], float)
        rel_v = vel - u_w
        F -= 0.5 * RHO_W * p.cd * np.pi * R * R * self.frac * np.linalg.norm(rel_v) * rel_v

        # zachte horizontale verankering
        if p.mooring_period > 0:
            F[:2] -= self.mass * (2 * np.pi / p.mooring_period) ** 2 * (pos[:2] - self.home)

        return F, self.mass + m_a

    def step(self, w: Water, amb: Water, dt: float) -> None:
        F, m_eff = self.forces(w, amb)
        self.vel += dt * F / m_eff          # semi-impliciet Euler (symplectisch)
        self.pos += dt * self.vel

    def source(self, w: Water) -> Optional[np.ndarray]:
        """Bron in de massabalans: water dat door de beweging van de bal wordt verplaatst."""
        if not self.p.two_way or self.A_wp <= 0.0:
            return None
        Q = -self.A_wp * self.vel[2]                      # [m^3/s], omlaag = water omhoog
        # zacht (Gaussisch) uitgesmeerd over enkele cellen: voorkomt dat de
        # koppeling korte golven op roosterschaal aanslaat
        sigma = max(self.p.source_width * self.R, 1.5 * w.dx)
        r2 = (w.X - self.pos[0]) ** 2 + (w.Y - self.pos[1]) ** 2
        wgt = np.exp(-0.5 * r2 / sigma ** 2)
        s = wgt.sum() * w.dx * w.dy
        return Q * wgt / s if s > 0 else None


class Simulation:
    def __init__(self, p: Params):
        self.p = p
        self.water = Water(p)
        # ongestoorde golf: dezelfde vergelijkingen in een smal bassin (4 cellen
        # breed, glijdende wanden) = exact 1D; bevat de bal niet
        self.ambient = Water(replace(p, W=4 * p.dx, sponge_y=0.0))
        self.ball = Ball(p)
        self.ball.pos[2] += p.drop

    def drop_ball(self, height: float) -> None:
        """Zet de bal op `height` meter boven zijn evenwichtsstand en laat hem los."""
        b = self.ball
        b.pos[:] = [b.home[0], b.home[1], b.z_eq + height]
        b.vel[:] = 0.0

    def advance(self) -> None:
        p = self.p
        for _ in range(p.steps_per_frame):
            self.ball.step(self.water, self.ambient, p.dt)
            self.water.step(p.dt, self.ball.source(self.water))
            self.ambient.step(p.dt)

    @property
    def t(self) -> float:
        return self.water.t


# =========================================================================
# Visualisatie (PyVista)
# =========================================================================
def _quad_faces(nu: int, nv: int) -> np.ndarray:
    """Quad-cellen voor een (nu x nv) puntenrooster met id = i*nv + j."""
    i, j = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1), indexing="ij")
    a = (i * nv + j).ravel()
    return np.column_stack([np.full(a.size, 4), a, a + nv, a + nv + 1, a + 1]).ravel()


def build_scene(sim: Simulation, off_screen: bool = False, size=(1280, 720),
                shade: Optional[float] = None, radiated: bool = False,
                zscale: Optional[float] = None):
    """shade: versterking van de schaduw op het water (None = automatisch).
    radiated: kleur alleen het deel van het water dat de bal zelf uitzendt.
    zscale: verticale uitvergroting van alle hoogteverschillen t.o.v. rust
            (None = 3 bij golven, 1 zonder golven); de bal beweegt mee."""
    import pyvista as pv
    from matplotlib.colors import LinearSegmentedColormap

    p, w, b = sim.p, sim.water, sim.ball
    K = zscale if zscale is not None else (3.0 if p.wave_amp > 0 else 1.0)
    h = p.depth
    ym = 0.5 * p.W
    xl, xr = max(2.0, p.x0 - 6.0), min(p.L - p.sponge_x, p.x0 + 8.0)
    half_w = min(3.6, 0.5 * p.W - p.sponge_y)
    yl, yr = ym - half_w, ym + half_w
    nxr = int(round((xr - xl) / (w.dx / 2))) + 1
    nyr = int(round((yr - yl) / (w.dy / 2))) + 1
    xs, ys = np.linspace(xl, xr, nxr), np.linspace(yl, yr, nyr)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")

    pl = pv.Plotter(off_screen=off_screen, window_size=list(size))
    pl.set_background("#dcecf7", top="#f7fbff")
    pl.enable_anti_aliasing("ssaa") if off_screen else pl.enable_anti_aliasing()

    # wateroppervlak
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)])
    surf = pv.PolyData(pts, _quad_faces(nxr, nyr))
    surf["eta"] = np.zeros(XX.size)
    cmap = LinearSegmentedColormap.from_list(
        "water", ["#082f66", "#134f9b", "#2b7fcf", "#78bdec", "#f0faff"])
    surf.compute_normals(inplace=True, split_vertices=False, consistent_normals=False,
                         auto_orient_normals=False)
    water_actor = pl.add_mesh(surf, scalars="eta", cmap=cmap, clim=[-0.01, 0.01],
                              show_scalar_bar=False, smooth_shading=True, opacity=0.78,
                              specular=0.5, specular_power=30, name="water")

    # wanden van het "aquarium": doorsnede van het water, volgt het oppervlak
    walls = []

    def wall(axis, val, coords, opacity):
        n = coords.size
        t = np.column_stack([coords, coords]).ravel()
        if axis == "y":      # vlak y = val, loopt langs x
            P = np.column_stack([t, np.full(2 * n, val), np.zeros(2 * n)])
        else:                # vlak x = val, loopt langs y
            P = np.column_stack([np.full(2 * n, val), t, np.zeros(2 * n)])
        P[0::2, 2] = -h
        m = pv.PolyData(P, _quad_faces(n, 2))
        pl.add_mesh(m, color="#2a74c4", opacity=opacity, smooth_shading=False)
        walls.append((m, axis, val, coords))
    wall("y", yr, xs, 0.55)
    wall("x", xr, ys, 0.55)
    wall("y", yl, xs, 0.42)      # voorwand: deels doorzichtig

    # zeebodem met ruitpatroon
    floor = pv.Plane(center=((xl + xr) / 2, ym, -h), direction=(0, 0, 1),
                     i_size=xr - xl, j_size=yr - yl,
                     i_resolution=int(xr - xl), j_resolution=int(yr - yl))
    ci = np.arange(floor.n_cells)
    floor.cell_data["c"] = ((ci // int(yr - yl) + ci % int(yr - yl)) % 2).astype(float)
    pl.add_mesh(floor, scalars="c", cmap=["#b49c6a", "#a38b5b"], show_scalar_bar=False,
                specular=0.0)

    # bal met strepen
    ball = pv.Sphere(radius=b.R, center=(0, 0, 0), theta_resolution=64, phi_resolution=64)
    az = np.arctan2(ball.points[:, 1], ball.points[:, 0])
    ball["stripe"] = (np.floor((az + np.pi) / (2 * np.pi) * 8) % 2).astype(float)
    ball_actor = pl.add_mesh(ball, scalars="stripe", cmap=["#d9402b", "#f6f1e7"],
                             show_scalar_bar=False, smooth_shading=True,
                             specular=0.5, specular_power=30)

    # raster op het water dat met de golven meebuigt: maakt de beweging zichtbaar
    pts_xy, polylines, off = [], [], 0
    for xv in np.arange(np.ceil(xl), xr + 1e-9, 1.0):          # lijnen x = constant
        pts_xy.append(np.column_stack([np.full(nyr, xv), ys]))
        polylines += [nyr, *range(off, off + nyr)]
        off += nyr
    for yv in np.arange(np.ceil(yl), yr + 1e-9, 1.0):          # lijnen y = constant
        pts_xy.append(np.column_stack([xs, np.full(nxr, yv)]))
        polylines += [nxr, *range(off, off + nxr)]
        off += nxr
    LP = np.vstack(pts_xy)
    gridlines = pv.PolyData(np.column_stack([LP, np.zeros(len(LP))]))
    gridlines.lines = np.array(polylines)
    pl.add_mesh(gridlines, color="white", opacity=0.55, line_width=1.3, lighting=False)

    pl.add_light(pv.Light(position=(p.x0 - 6, ym - 6, 12), focal_point=(p.x0, ym, 0),
                          color="white", intensity=0.35))
    # laag invallend licht: laat de hellingen van de golven goed uitkomen
    pl.add_light(pv.Light(position=(p.x0 + 10, ym - 4, 1.8), focal_point=(p.x0, ym, 0),
                          color="white", intensity=0.5))
    if p.wave_amp > 0:      # golven: lage camera, zodat het golfprofiel langs de wanden zichtbaar is
        pl.camera_position = [(p.x0 - 3.0, yl - 5.0, 2.7), (p.x0 + 1.5, ym + 0.2, -0.05), (0, 0, 1)]
        pl.camera.view_angle = 42
    else:                   # geen golven (bal laten vallen): dichter op de ringen, meer van boven
        pl.camera_position = [(p.x0 - 1.2, ym - 6.4, 6.6), (p.x0, ym, -0.1), (0, 0, 1)]
        pl.camera.view_angle = 45

    dxr, dyr = xs[1] - xs[0], ys[1] - ys[0]
    view = {"radiated": radiated, "cmax": 0.002, "slope": 0.02}
    drop_h = p.drop if p.drop > 0 else 0.8
    pl.add_key_event("d", lambda: sim.drop_ball(drop_h))
    pl.add_key_event("k", lambda: view.update(radiated=not view["radiated"]))

    def update_scene():
        e = bilerp(w.eta, w.dx / 2, w.dy / 2, w.dx, w.dy, XX, YY)      # totale waterhoogte
        e_amb = sim.ambient.eta_at(xs, 0.5 * sim.ambient.p.W)           # golf zonder bal
        shown = e - e_amb[:, None] if view["radiated"] else e
        P = surf.points
        P[:, 2] = K * e.ravel()                                         # geometrie (uitvergroot)
        surf.points = P
        Pg = gridlines.points
        Pg[:, 2] = K * bilerp(w.eta, w.dx / 2, w.dy / 2, w.dx, w.dy, LP[:, 0], LP[:, 1]) + 0.01
        gridlines.points = Pg

        # kleurschaal en schaduwversterking passen zich aan de rimpelhoogte aan
        # (statistiek buiten de bal, snel omhoog, langzaam omlaag)
        far = np.hypot(XX - b.pos[0], YY - b.pos[1]) > 1.3 * b.R
        gx, gy = np.gradient(e, dxr, axis=0), np.gradient(e, dyr, axis=1)
        # 92e percentiel: de sterke bult vlak bij de bal verzadigt, zodat de ringen
        # verderop het volle contrast krijgen
        view["slope"] = max(float(np.percentile(np.hypot(gx, gy)[far], 92)),
                            0.985 * view["slope"], 0.01)
        view["cmax"] = max(float(np.percentile(np.abs(shown)[far], 92)),
                           0.985 * view["cmax"], 0.002)
        gain = shade if shade else float(np.clip(0.3 / (K * view["slope"]), 1.0, 12.0))
        nrm = np.column_stack([-gain * K * gx.ravel(), -gain * K * gy.ravel(), np.ones(e.size)])
        nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        surf.point_data["Normals"] = nrm
        surf.GetPointData().SetActiveNormals("Normals")
        surf["eta"] = shown.ravel()
        water_actor.mapper.scalar_range = (-1.2 * view["cmax"], 1.2 * view["cmax"])

        for m, axis, val, coords in walls:
            top = (w.eta_at(coords, np.full(coords.size, val)) if axis == "y"
                   else w.eta_at(np.full(coords.size, val), coords))
            Q = m.points
            Q[1::2, 2] = K * top
            m.points = Q
        # de bal beweegt mee met dezelfde uitvergroting (t.o.v. zijn rustwaterlijn)
        ball_actor.position = (b.pos[0], b.pos[1], b.z_eq + K * (b.pos[2] - b.z_eq))
        mode = "alleen golven van de bal" if view["radiated"] else "totale waterhoogte"
        pl.add_text(f"t = {w.t:5.1f} s    bal: x = {b.pos[0]:5.2f}  y = {b.pos[1]:5.2f}  "
                    f"z = {b.pos[2]:+.2f} m    onder water: {100 * b.frac:3.0f} %\n"
                    f"kleur: {mode}, schaal ±{1000 * view['cmax']:.0f} mm    "
                    f"hoogte ×{K:.1f} uitvergroot    schaduw ×{gain:.1f}    "
                    f"[d] bal laten vallen   [k] kleur wisselen",
                    position="upper_left", font_size=10, color="#26323d", name="info")

    update_scene()
    return pl, update_scene


def run_gui(sim: Simulation, save: Optional[str], seconds: float,
            shade: Optional[float] = None, radiated: bool = False,
            zscale: Optional[float] = None, warmup: float = 0.0) -> None:
    p = sim.p
    frame_dt = p.dt * p.steps_per_frame
    for _ in range(int(warmup / frame_dt)):      # eerst laten inlopen, zodat de golven er al zijn
        sim.advance()
    if save:
        size = (800, 450) if save.endswith(".gif") else (1280, 720)
        pl, update = build_scene(sim, off_screen=True, size=size, shade=shade,
                                 radiated=radiated, zscale=zscale)
        if save.endswith(".gif"):
            pl.open_gif(save, fps=round(1 / frame_dt))
        else:
            pl.open_movie(save, framerate=round(1 / frame_dt))
        for _ in range(int(seconds / frame_dt)):
            sim.advance()
            update()
            pl.write_frame()
        pl.close()
        print(f"opgeslagen: {save}")
        return

    pl, update = build_scene(sim, shade=shade, radiated=radiated, zscale=zscale)
    pl.show(interactive_update=True, auto_close=False)
    t_next = time.perf_counter()
    while not getattr(pl, "_closed", False):
        sim.advance()
        update()
        try:
            pl.update()
        except (AttributeError, RuntimeError):  # venster is gesloten
            break
        t_next += frame_dt                      # probeer real-time te blijven
        pause = t_next - time.perf_counter()
        if pause > 0:
            time.sleep(pause)
        else:
            t_next = time.perf_counter()


# =========================================================================
# Zelftest
# =========================================================================
def selftest() -> None:
    p = Params()
    sim = Simulation(p)
    w, b = sim.water, sim.ball
    print(f"rooster {w.nx} x {w.ny}, onbekenden in dispersiesysteem: {w.lu.shape[0]}")

    for dens in (250.0, 500.0, 750.0):
        q = Params(density=dens, wave_amp=0.0)
        s = Simulation(q)
        z_start = s.ball.pos[2]
        q.steps_per_frame = 1
        for _ in range(int(15 / q.dt)):
            s.advance()
        print(f"dichtheid {dens:4.0f}: z start {z_start:+.4f}  na 15 s {s.ball.pos[2]:+.4f}  "
              f"onderwater {100 * s.ball.frac:.1f}% (exact {dens / 10:.1f}%)  |v|={np.linalg.norm(s.ball.vel):.1e}")

    p.steps_per_frame = 1
    t0 = time.perf_counter()
    n = 500
    for _ in range(n):
        sim.advance()
    print(f"rekentijd per tijdstap: {1000 * (time.perf_counter() - t0) / n:.1f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description="Drijvende bal in golven (3D)")
    ap.add_argument("--density", type=float, help="dichtheid bal [kg/m3] (water = 1000)")
    ap.add_argument("--amp", type=float, help="golfamplitude [m]")
    ap.add_argument("--period", type=float, help="golfperiode [s]")
    ap.add_argument("--radius", type=float, help="straal bal [m]")
    ap.add_argument("--drop", type=float, help="laat de bal zoveel m boven evenwicht los (met --amp 0: ringgolven)")
    ap.add_argument("--zscale", type=float,
                    help="verticale uitvergroting van de golven in beeld (standaard 3 met golven, 1 zonder)")
    ap.add_argument("--warmup", type=float,
                    help="seconden vooruitrekenen voor de start, zodat de golven er al zijn (standaard 10)")
    ap.add_argument("--shade", type=float, help="schaduwversterking van het water (standaard automatisch)")
    ap.add_argument("--radiated", action="store_true",
                    help="kleur alleen het deel van het water dat de bal zelf uitzendt")
    ap.add_argument("--linear", action="store_true", help="schakel niet-lineaire termen uit")
    ap.add_argument("--free", action="store_true", help="geen verankering: bal drijft vrij weg")
    ap.add_argument("--one-way", action="store_true", help="bal zendt zelf geen golven uit")
    ap.add_argument("--save", type=str, help="sla op als .gif of .mp4 (mp4 vereist imageio-ffmpeg)")
    ap.add_argument("--seconds", type=float, default=20.0, help="duur bij --save")
    ap.add_argument("--selftest", action="store_true", help="fysica-controles zonder beeld")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    p = Params()
    if args.density is not None: p.density = args.density
    if args.amp is not None:     p.wave_amp = args.amp
    if args.period is not None:  p.wave_period = args.period
    if args.radius is not None:  p.radius = args.radius
    if args.drop is not None:    p.drop = args.drop
    if args.linear:              p.nonlinear = False
    if args.free:                p.mooring_period = 0.0
    if args.one_way:             p.two_way = False

    warmup = args.warmup if args.warmup is not None else (10.0 if p.wave_amp > 0 and p.drop == 0 else 0.0)
    run_gui(Simulation(p), args.save, args.seconds, shade=args.shade, radiated=args.radiated,
            zscale=args.zscale, warmup=warmup)


if __name__ == "__main__":
    main()
