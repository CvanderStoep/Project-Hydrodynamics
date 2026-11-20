"""
Drijvende bal in golven - 2D zijaanzicht (x = horizontaal, z = omhoog)

Fysica
------
Water  Boussinesq-golfvergelijkingen (Peregrine, vlakke bodem), 1D in x:
           d(eta)/dt + d(H u)/dx = 0                     H = h + eta
           (1 - h^2/3 d2/dx2) du/dt = -g d(eta)/dx - u du/dx
       De dispersieterm zorgt dat golven met kh ~ 1 de juiste snelheid
       hebben (gewone ondiep-water-vergelijkingen steilen op tot bores).
       Numeriek: gestaggerd rooster, forward-backward tijdstap, tridiagonale
       solve voor de dispersieterm. Links een golfmaker (zuiger), rechts een
       sponge-laag die golven absorbeert (geen reflecties).

Bal    Starre cilinder (2D), vrijheidsgraden x en z. Krachten:
         - zwaartekracht
         - hydrostatische druk, geintegreerd over het natte oppervlak
           (Archimedes + golfhelling, dus ook de horizontale golfkracht)
         - toegevoegde massa (Morison): het water dat meebeweegt
         - kwadratische weerstand t.o.v. de lokale watersnelheid
         - stralingsdemping: de golven die de bal zelf maakt, lineaire
           benadering (de bal werkt niet terug op het waterveld)
         - zachte horizontale verankering (periode 20 s, veel trager dan de
           golven) zodat de bal niet uit beeld drijft; --free zet die uit

Gebruik
-------
    python ball_in_waves.py                      # live animatie
    python ball_in_waves.py --density 250        # lichtere bal
    python ball_in_waves.py --amp 0.15 --period 2.5
    python ball_in_waves.py --save golven.gif --seconds 20
"""
from __future__ import annotations

import argparse
import itertools
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import diags
from scipy.sparse.linalg import splu

G = 9.81          # m/s^2
RHO_W = 1000.0    # kg/m^3


@dataclass
class Params:
    # water
    L: float = 30.0            # lengte bassin [m]
    nx: int = 300              # aantal cellen (dx = 0.1 m)
    depth: float = 1.0         # waterdiepte [m]
    wave_amp: float = 0.10     # amplitude golfmaker [m]
    wave_period: float = 3.0   # periode [s]
    sponge_len: float = 7.0    # lengte absorberende zone rechts [m]
    sponge_strength: float = 4.0
    nonlinear: bool = True
    # bal (per meter breedte, 2D cilinder)
    radius: float = 0.35       # [m]
    density: float = 500.0     # [kg/m^3]; < 1000 = drijft
    x0: float = 15.0
    z0: float = 0.0            # beginhoogte van het middelpunt
    cd: float = 1.0            # weerstandscoefficient cilinder
    ca: float = 1.0            # toegevoegde-massacoefficient cilinder
    mooring_period: float = 20.0  # zachte horizontale veer [s]; 0 = vrij drijvend
    # numeriek
    dt: float = 0.004
    steps_per_frame: int = 6


class Water:
    """Boussinesq-water op een gestaggerd rooster: eta in cellen, u op randen."""

    def __init__(self, p: Params):
        self.p = p
        self.dx = p.L / p.nx
        self.x = (np.arange(p.nx) + 0.5) * self.dx       # celmiddens (eta)
        self.xf = np.arange(p.nx + 1) * self.dx          # randen (u)
        self.eta = np.zeros(p.nx)
        self.u = np.zeros(p.nx + 1)
        self.dudt = np.zeros(p.nx + 1)                   # horizontale versnelling
        self.detadt = np.zeros(p.nx)                     # verticale snelheid oppervlak
        self.d2etadt2 = np.zeros(p.nx)                   # verticale versnelling oppervlak
        self.t = 0.0
        self.c = np.sqrt(G * p.depth)
        self.omega = 2 * np.pi / p.wave_period
        # fasesnelheid van het model zelf (met dispersie), voor de golfmaker
        k = brentq(lambda k: G * k * k * p.depth / (1 + (k * p.depth) ** 2 / 3)
                   - self.omega ** 2, 1e-6, 50.0)
        self.c_phase = self.omega / k

        s0 =p.L - p.sponge_len
        self.sig_c = p.sponge_strength * np.clip((self.x - s0) / p.sponge_len, 0, 1) ** 2
        self.sig_f = p.sponge_strength * np.clip((self.xf - s0) / p.sponge_len, 0, 1) ** 2

        # (1 - h^2/3 d2/dx2) a = rhs, op de binnenste randen; vooraf gefactoriseerd
        n = p.nx - 1
        self.beta = p.depth ** 2 / 3.0 / self.dx ** 2
        A = diags([-self.beta * np.ones(n - 1),
                   (1 + 2 * self.beta) * np.ones(n),
                   -self.beta * np.ones(n - 1)], [-1, 0, 1], format="csc")
        self.lu = splu(A)

    # --- golfmaker -------------------------------------------------------
    def paddle_u(self, t: float) -> float:
        p = self.p
        ramp = 0.5 * (1 - np.cos(np.pi * min(t / (2 * p.wave_period), 1.0)))
        # massabehoud voor een lopende golf: u = (c_fase / h) * eta
        return ramp * p.wave_amp * self.c_phase / p.depth * np.sin(self.omega * t)

    # --- tijdstap --------------------------------------------------------
    def step(self, dt: float) -> None:
        p, dx = self.p, self.dx
        eta_old, detadt_old = self.eta.copy(), self.detadt.copy()

        # 1. massabehoud
        H = np.empty_like(self.u)
        if p.nonlinear:
            H[1:-1] = p.depth + 0.5 * (self.eta[:-1] + self.eta[1:])
            H[0], H[-1] = p.depth + self.eta[0], p.depth + self.eta[-1]
        else:
            H[:] = p.depth
        flux = H * self.u
        self.eta -= dt / dx * (flux[1:] - flux[:-1])
        self.eta -= dt * self.sig_c * self.eta

        # 2. impuls (met dispersie)
        u = self.u
        rhs = -G * (self.eta[1:] - self.eta[:-1]) / dx - self.sig_f[1:-1] * u[1:-1]
        if p.nonlinear:
            dudx = np.where(u[1:-1] > 0, u[1:-1] - u[:-2], u[2:] - u[1:-1]) / dx  # upwind
            rhs -= u[1:-1] * dudx
        u0_new = self.paddle_u(self.t + dt)
        a0 = (u0_new - u[0]) / dt
        rhs[0] += self.beta * a0
        a = self.lu.solve(rhs)
        self.u[1:-1] += dt * a
        self.u[0] = u0_new
        self.u[-1] = 0.0
        self.dudt[1:-1], self.dudt[0], self.dudt[-1] = a, a0, 0.0

        self.t += dt
        self.detadt = (self.eta - eta_old) / dt
        self.d2etadt2 = (self.detadt - detadt_old) / dt

    # --- interpolatie op positie x --------------------------------------
    def eta_at(self, x):   return np.interp(x, self.x, self.eta)
    def u_at(self, x):     return np.interp(x, self.xf, self.u)
    def w_at(self, x):     return np.interp(x, self.x, self.detadt)
    def ax_at(self, x):    return np.interp(x, self.xf, self.dudt)
    def az_at(self, x):    return np.interp(x, self.x, self.d2etadt2)


class Ball:
    """Starre cilinder met hydrostatische druk, toegevoegde massa en weerstand."""

    def __init__(self, p: Params):
        self.p = p
        self.R = p.radius
        self.mass = p.density * np.pi * self.R ** 2      # [kg per m breedte]
        self.x, self.z = p.x0, p.z0
        self.vx = self.vz = 0.0
        n = 96
        th = np.arange(n) * 2 * np.pi / n
        self._cos, self._sin = np.cos(th), np.sin(th)
        self._ds = 2 * np.pi * self.R / n
        self._dxs = np.linspace(-self.R, self.R, 101)
        self._half = np.sqrt(np.maximum(self.R ** 2 - self._dxs ** 2, 0.0))

    def wet_area(self, water: Water) -> float:
        eta = np.interp(self.x + self._dxs, water.x, water.eta)
        h = np.clip(eta - (self.z - self._half), 0.0, 2 * self._half)
        return float(np.sum(h) * (self._dxs[1] - self._dxs[0]))

    def forces(self, water: Water):
        R, p = self.R, self.p

        # hydrostatische druk op de rand van de cirkel: p = rho g (eta - z)
        px = self.x + R * self._cos
        pz = self.z + R * self._sin
        depth_below = np.interp(px, water.x, water.eta) - pz
        press = RHO_W * G * np.clip(depth_below, 0.0, None)
        Fx = -np.sum(press * self._cos) * self._ds       # druk duwt naar binnen
        Fz = -np.sum(press * self._sin) * self._ds

        A_wet = self.wet_area(water)
        frac = A_wet / (np.pi * R ** 2)

        # toegevoegde massa (Morison): Ca*rho*A*(a_water - a_bal); a_bal-deel
        # zit in m_eff, het water-deel komt hier als kracht
        Fx += p.ca * RHO_W * A_wet * water.ax_at(self.x)
        Fz += p.ca * RHO_W * A_wet * water.az_at(self.x)

        # kwadratische weerstand t.o.v. het water (alleen natte deel)
        rx = self.vx - water.u_at(self.x)
        rz = self.vz - water.w_at(self.x)
        k = 0.5 * RHO_W * p.cd * 2 * R * frac
        Fx -= k * abs(rx) * rx
        Fz -= k * abs(rz) * rz

        # stralingsdemping (lange-golf): b = rho g B^2 / (2c), B = breedte op waterlijn
        dz = np.clip(water.eta_at(self.x) - self.z, -R, R)
        B = 2 * np.sqrt(R ** 2 - dz ** 2)
        Fz -= RHO_W * G * B ** 2 / (2 * water.c) * self.vz

        Fz -= self.mass * G
        m_eff = self.mass + p.ca * RHO_W * A_wet
        if p.mooring_period > 0:   # zachte verankering: houdt de bal in beeld
            Fx -= self.mass * (2 * np.pi / p.mooring_period) ** 2 * (self.x - p.x0)
        return Fx, Fz, m_eff, A_wet

    def step(self, water: Water, dt: float) -> None:
        Fx, Fz, m_eff, _ = self.forces(water)
        self.vx += dt * Fx / m_eff          # semi-impliciet Euler (symplectisch)
        self.vz += dt * Fz / m_eff
        self.x += dt * self.vx
        self.z += dt * self.vz


class Tracers:
    """Passieve deeltjes: laten de waterbeweging zien (puur visueel)."""

    def __init__(self, p: Params, n: int, xmin: float, xmax: float, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.p = p
        self.x = rng.uniform(xmin, xmax, n)
        self.s = rng.uniform(0.05, 0.95, n)   # relatieve hoogte in de waterkolom

    def step(self, water: Water, dt: float) -> None:
        self.x = np.clip(self.x + water.u_at(self.x) * dt, 0.01, self.p.L - 0.01)

    def z(self, water: Water):
        h = self.p.depth
        return -h + self.s * (h + water.eta_at(self.x))


class Simulation:
    def __init__(self, p: Params):
        self.p = p
        self.water = Water(p)
        self.ball = Ball(p)
        self.tracers = Tracers(p, 70, p.x0 - 6, p.x0 + 6)

    def advance(self) -> None:
        p = self.p
        for _ in range(p.steps_per_frame):
            self.ball.step(self.water, p.dt)
            self.water.step(p.dt)
            self.tracers.step(self.water, p.dt)

    @property
    def t(self) -> float:
        return self.water.t


# =========================================================================
# Visualisatie
# =========================================================================
def build_figure(sim: Simulation):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    p = sim.p
    view_w = 12.0
    x_lo, x_hi = p.x0 - view_w / 2, p.x0 + view_w / 2
    vis = (sim.water.x >= x_lo - 0.2) & (sim.water.x <= x_hi + 0.2)
    xv = sim.water.x[vis]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.5),
                                  gridspec_kw={"height_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.09, hspace=0.25)

    # --- zijaanzicht -----------------------------------------------------
    ax.set_facecolor("#eef6fc")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-p.depth - 0.08, 0.75)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.add_patch(Rectangle((x_lo, -p.depth - 0.08), view_w, 0.08, fc="#c9b48a", ec="none", zorder=1))

    ball_patch = Circle((sim.ball.x, sim.ball.z), sim.ball.R,
                        fc="#d9402b", ec="#7a1d10", lw=1.5, zorder=2)
    ax.add_patch(ball_patch)
    water_poly = Polygon(np.zeros((4, 2)), closed=True, fc="#2b7fd0", alpha=0.55, ec="none", zorder=3)
    ax.add_patch(water_poly)
    surface, = ax.plot([], [], color="#1b5fa8", lw=2, zorder=4)
    dots = ax.scatter(sim.tracers.x, sim.tracers.z(sim.water), s=7, c="white",
                      alpha=0.8, zorder=5)
    info = ax.text(0.01, 0.93, "", transform=ax.transAxes, fontsize=9, va="top")

    # --- tijdreeks -------------------------------------------------------
    hist_t = deque(maxlen=int(15 / (p.dt * p.steps_per_frame)))
    hist_ball = deque(maxlen=hist_t.maxlen)
    hist_eta = deque(maxlen=hist_t.maxlen)
    line_eta, = ax2.plot([], [], color="#2b7fd0", lw=1.5, label="wateroppervlak op positie bal")
    line_ball, = ax2.plot([], [], color="#d9402b", lw=1.8, label="hoogte middelpunt bal")
    ax2.set_ylim(-0.3, 0.3)
    ax2.set_xlim(0, 15)
    ax2.set_xlabel("tijd [s]")
    ax2.set_ylabel("hoogte [m]")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    def update(_frame=None):
        sim.advance()
        w, b = sim.water, sim.ball
        ball_patch.center = (b.x, b.z)
        top = np.column_stack([xv, w.eta[vis]])
        bottom = np.array([[xv[-1], -p.depth], [xv[0], -p.depth]])
        water_poly.set_xy(np.vstack([top, bottom]))
        surface.set_data(xv, w.eta[vis])
        dots.set_offsets(np.column_stack([sim.tracers.x, sim.tracers.z(w)]))
        frac = b.wet_area(w) / (np.pi * b.R ** 2)
        info.set_text(f"t = {w.t:5.1f} s     bal x = {b.x:5.2f} m     "
                      f"onder water: {100 * frac:3.0f} %")

        hist_t.append(w.t)
        hist_ball.append(b.z)
        hist_eta.append(w.eta_at(b.x))
        line_ball.set_data(hist_t, hist_ball)
        line_eta.set_data(hist_t, hist_eta)
        ax2.set_xlim(max(0.0, w.t - 15), max(15.0, w.t))
        return ball_patch, water_poly, surface, dots, info, line_ball, line_eta

    update_first = update
    return fig, update_first


def main() -> None:
    ap = argparse.ArgumentParser(description="Drijvende bal in golven")
    ap.add_argument("--density", type=float, help="dichtheid bal [kg/m3] (water = 1000)")
    ap.add_argument("--amp", type=float, help="golfamplitude [m]")
    ap.add_argument("--period", type=float, help="golfperiode [s]")
    ap.add_argument("--radius", type=float, help="straal bal [m]")
    ap.add_argument("--linear", action="store_true", help="schakel niet-lineaire termen uit")
    ap.add_argument("--free", action="store_true",
                    help="geen verankering: de bal drijft vrij weg (golfdrift)")
    ap.add_argument("--save", type=str, help="sla op als .gif of .mp4 (mp4 vereist ffmpeg)")
    ap.add_argument("--seconds", type=float, default=20.0, help="duur bij --save")
    args = ap.parse_args()

    p = Params()
    if args.density is not None: p.density = args.density
    if args.amp is not None:     p.wave_amp = args.amp
    if args.period is not None:  p.wave_period = args.period
    if args.radius is not None:  p.radius = args.radius
    if args.linear:              p.nonlinear = False
    if args.free:                p.mooring_period = 0.0

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    sim = Simulation(p)
    fig, update = build_figure(sim)
    frame_dt = p.dt * p.steps_per_frame

    if args.save:
        n_frames = int(args.seconds / frame_dt)
        anim = FuncAnimation(fig, update, frames=n_frames, cache_frame_data=False)
        anim.save(args.save, fps=round(1 / frame_dt), dpi=90)
        print(f"opgeslagen: {args.save}")
    else:
        anim = FuncAnimation(fig, update, frames=itertools.count(), interval=1000 * frame_dt,
                             blit=False, cache_frame_data=False)
        plt.show()


if __name__ == "__main__":
    main()
