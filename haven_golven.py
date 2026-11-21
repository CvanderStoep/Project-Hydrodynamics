"""
Golfpatronen in een vierkante haven met een kleine opening (PyVista, 3D).

Model: 2D lineaire golfvergelijking  u_tt = c^2 * (u_xx + u_yy) - demping
- Muren van de haven weerkaatsen golven volledig (Neumann-randvoorwaarde).
- Golven komen als vlakke golf uit zee (onderkant) en gaan door de opening naar binnen.
- Aan de buitenranden van het domein zit een "spons" die golven absorbeert.
- Getoond wordt alleen het water in de haven (zee en kust blijven buiten beeld).

Gebruik:
    python haven_golven.py                   # live venster
    python haven_golven.py --save haven.gif  # opslaan als gif

Installeren:  pip install pyvista numpy
"""
import sys
import numpy as np
import pyvista as pv

# ------------------------------------------------------------------ parameters
N = 120                 # gridgrootte (N x N cellen)
C = 1.0                 # golfsnelheid (cellen per tijdseenheid)
SPEED = 0.5            # simulatiesnelheid (1.0 = origineel, kleiner = trager)
DT = 0.5 * SPEED        # tijdstap (CFL: C*DT <= 0.707)
WAVELENGTH = 28.0       # golflengte in cellen (langer = rustiger)
AMPLITUDE = 0.08
OPENING_WIDTH = 10      # breedte van de havenopening in cellen
WALL = 3                # muurdikte in cellen
STEPS_PER_FRAME = 1
FRAMES = 600            # alleen voor --save

# visualisatie
Z_SCALE = 12.0          # verticale uitvergroting van de golven
GRID_EVERY = 4          # om de zoveel cellen een gridlijn op het oppervlak
CMAP = "ocean"          # kleurenschema water
CLIM = (-0.4, 0.4)      # kleurbereik (golfhoogte)
WALL_TOP = 6.0          # bovenkant van de muren (in getekende eenheden)
SIZE = (1100, 800)

# ------------------------------------------------------------------ geometrie
wall = np.zeros((N, N), dtype=bool)          # wall[y, x]
x0, x1 = 20, 100                              # haven links/rechts (buitenkant)
y0, y1 = 40, 108                              # haven onder/boven (buitenkant)
wall[y0:y0 + WALL, x0:x1] = True              # onderste muur (zeezijde)
wall[y1 - WALL:y1, x0:x1] = True              # bovenmuur
wall[y0:y1, x0:x0 + WALL] = True              # linkermuur
wall[y0:y1, x1 - WALL:x1] = True              # rechtermuur
mid = (x0 + x1) // 2
o0, o1 = mid - OPENING_WIDTH // 2, mid + OPENING_WIDTH // 2
wall[y0:y0 + WALL, o0:o1] = False             # opening
water = ~wall

# spons-demping aan de buitenranden
yy, xx = np.mgrid[0:N, 0:N]
edge = np.minimum.reduce([xx, yy, N - 1 - xx, N - 1 - yy]).astype(float)
SPONGE = 12
damping = np.where(edge < SPONGE, 0.08 * ((SPONGE - edge) / SPONGE) ** 2, 0.0)
damping = damping + np.where(water, 0.003, 0.0)   # lichte wrijving in het water

# bron: lijn op y = 18, golft in +y richting
SRC_Y = 18
omega = 2 * np.pi * C / WAVELENGTH

# ------------------------------------------------------------------ simulatie
u = np.zeros((N, N))
u_prev = np.zeros((N, N))
water_f = water.astype(float)
r2 = (C * DT) ** 2
t = 0.0


def laplacian(u):
    """Laplaciaan met reflecterende muren: alleen uitwisseling tussen watercellen."""
    lap = np.zeros_like(u)
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        lap += np.roll(water_f, shift, axis=axis) * (np.roll(u, shift, axis=axis) - u)
    return lap


def step():
    global u, u_prev, t
    lap = laplacian(u)
    u_new = 2 * u - u_prev + r2 * lap - damping * DT * (u - u_prev)
    ramp = min(1.0, t / (2 * 2 * np.pi / omega))        # bron loopt rustig op
    u_new[SRC_Y, :] += AMPLITUDE * ramp * np.sin(omega * t) * 0.25 * SPEED ** 2
    u_new[wall] = 0.0
    u_prev, u = u, u_new
    t += DT


# ------------------------------------------------------------------ visualisatie
# alleen het water binnen de haven tonen
ix0, ix1 = x0 + WALL, x1 - WALL
iy0, iy1 = y0 + WALL, y1 - WALL
nx, ny = ix1 - ix0, iy1 - iy0
X, Y = np.meshgrid(np.arange(ix0, ix1, dtype=float), np.arange(iy0, iy1, dtype=float))

surface = pv.StructuredGrid(X, Y, np.zeros_like(X))
surface["u"] = np.zeros(X.size)

# gridlijnen: ijler raster dat exact op het oppervlak ligt (volle resolutie langs de lijnen)
def _build_gridlines():
    rows = sorted(set(range(0, ny, GRID_EVERY)) | {ny - 1})
    cols = sorted(set(range(0, nx, GRID_EVERY)) | {nx - 1})
    lines = []
    for j in rows:                                   # lijnen in x-richting
        ids = [j + i * ny for i in range(nx)]
        lines += [len(ids)] + ids
    for i in cols:                                   # lijnen in y-richting
        ids = [j + i * ny for j in range(ny)]
        lines += [len(ids)] + ids
    return pv.PolyData(np.array(surface.points), lines=np.array(lines))


gridlines = _build_gridlines()


def update_meshes():
    sub = u[iy0:iy1, ix0:ix1]
    surface.points[:, 2] = (sub * Z_SCALE).ravel(order="F")
    surface["u"] = sub.ravel(order="F")
    gridlines.points[:, 2] = surface.points[:, 2] + 0.15


def make_walls():
    """Muren als grijze blokken rond het water; de opening blijft open."""
    zb, zt = -WALL_TOP, WALL_TOP
    xl, xr = ix0, ix1 - 1               # binnenkant links/rechts
    yb, yt = iy0, iy1 - 1               # binnenkant onder/boven
    xa, xb = x0 - 0.5, x1 - 0.5         # buitenkant
    ya, yc = y0 - 0.5, y1 - 0.5
    boxes = [
        (xa, xl, ya, yc),                       # links
        (xr, xb, ya, yc),                       # rechts
        (xa, xb, yt, yc),                       # boven
        (xa, o0 - 0.5, ya, yb),                 # onder, links van opening
        (o1 - 0.5, xb, ya, yb),                 # onder, rechts van opening
    ]
    return [pv.Box(bounds=(a, b, c, d, zb, zt)) for a, b, c, d in boxes]


def build_plotter(off_screen):
    p = pv.Plotter(window_size=SIZE, off_screen=off_screen)
    p.set_background("white")
    p.add_mesh(surface, scalars="u", cmap=CMAP, clim=CLIM, show_scalar_bar=False,
               smooth_shading=False, specular=0.4)
    p.add_mesh(gridlines, color="#0b2a44", line_width=1.5, opacity=0.6,
               render_lines_as_tubes=False, lighting=False)
    for box in make_walls():
        p.add_mesh(box, color="#9a9a9a", smooth_shading=False)
    cx, cy = (ix0 + ix1) / 2, (iy0 + iy1) / 2
    p.camera_position = [(cx + 20, cy - 115, 85), (cx, cy, 0), (0, 0, 1)]
    p.camera.zoom(1.15)
    return p


def set_title(p):
    p.add_text(f"Golven in een haven    t = {t:6.1f}", position="upper_edge",
               font_size=11, color="black", name="titel")


def main():
    save = "--save" in sys.argv
    update_meshes()
    p = build_plotter(off_screen=save)
    if save:
        out = sys.argv[sys.argv.index("--save") + 1]
        p.open_gif(out, fps=25)
        for _ in range(FRAMES):
            for _ in range(STEPS_PER_FRAME):
                step()
            update_meshes()
            set_title(p)
            p.write_frame()
        p.close()
        print("opgeslagen als", out)
    else:
        p.show(auto_close=False, interactive_update=True)
        while p.render_window is not None:
            for _ in range(STEPS_PER_FRAME):
                step()
            update_meshes()
            set_title(p)
            p.update(stime=15)


if __name__ == "__main__":
    main()
