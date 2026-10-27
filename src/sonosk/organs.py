"""Anatomical phantoms of the breast, prostate, uterus and facial skin.

Each function builds a label map on a 1-, 2- or 3-D grid, gives every
region the properties of sonosk.tissues, and returns an OrganPhantom with
the medium, the names of the target regions (lesions to image or treat)
and a short description. The depth axis (last grid axis) points away from
the probe, which sits at z = 0; x is lateral and, in 3-D, y is elevation.

Layer thicknesses default to published mean values (cited in each
docstring). Interfaces between tissues undulate with smooth random
profiles, so every seed gives a different but plausible anatomy. All
sizes are in metres.

Lesions are described by small classes (Cyst, Fibroadenoma, Carcinoma,
Tumor) that can be placed in any phantom.
"""

from dataclasses import dataclass, field

import numpy as np

from . import phantom as ph
from . import tissues
from .medium import Medium


# --------------------------------------------------------------------------
# lesions


@dataclass
class Lesion:
    """Base class: a region painted with one tissue."""

    center: tuple
    tissue: str = field(default="breast carcinoma", kw_only=True)
    label: str = field(default=None, kw_only=True)

    def mask(self, grid, rng=None):
        raise NotImplementedError

    @property
    def name(self):
        return self.label or self.tissue


@dataclass
class Cyst(Lesion):
    """Simple cyst: an ellipsoid with semi-axes (a, a, b), anechoic."""

    a: float = 3e-3
    b: float = 2.5e-3
    tissue: str = field(default="breast cyst", kw_only=True)

    def mask(self, grid, rng=None):
        return ph.ellipsoid(grid, self.center, (self.a, self.a, self.b))


@dataclass
class Fibroadenoma(Lesion):
    """Oval, circumscribed, wider than tall (parallel orientation), with a
    gently lobulated margin."""

    a: float = 6e-3
    b: float = 3.5e-3
    angle: float = 0.0
    tissue: str = field(default="fibroadenoma", kw_only=True)

    def mask(self, grid, rng=None):
        rng = np.random.default_rng(rng)
        r = max(self.a, self.b)
        return ph.irregular(grid, self.center, r, roughness=0.05, correlation=r,
                            rng=rng, aspect=(self.a / r, self.a / r, self.b / r))


@dataclass
class Carcinoma(Lesion):
    """Irregular, spiculated, taller-than-wide mass (the BI-RADS features of
    a suspicious lesion)."""

    radius: float = 4e-3
    roughness: float = 0.25
    spicules: int = 8
    taller: float = 1.25
    tissue: str = field(default="breast carcinoma", kw_only=True)

    def mask(self, grid, rng=None):
        return ph.irregular(grid, self.center, self.radius, roughness=self.roughness,
                            spicules=self.spicules, rng=rng,
                            aspect=(1.0, 1.0, self.taller))


@dataclass
class Tumor(Lesion):
    """Generic mass of any tissue: an irregular sphere of given radius."""

    radius: float = 5e-3
    roughness: float = 0.1
    tissue: str = field(default="prostate carcinoma", kw_only=True)

    def mask(self, grid, rng=None):
        return ph.irregular(grid, self.center, self.radius, roughness=self.roughness, rng=rng)


@dataclass
class Calcification(Lesion):
    """Small bright calcification."""

    radius: float = 0.6e-3
    tissue: str = field(default="calcification", kw_only=True)

    def mask(self, grid, rng=None):
        return ph.sphere(grid, self.center, self.radius)


# --------------------------------------------------------------------------
# phantom container


@dataclass
class OrganPhantom:
    """An anatomical phantom.

    medium : Medium
    targets : names of the target materials (lesions)
    description : what the phantom contains
    probe : suggested probe and position, as text
    """

    medium: Medium
    targets: list = field(default_factory=list)
    description: str = ""
    probe: str = ""

    @property
    def grid(self):
        return self.medium.grid

    @property
    def names(self):
        return [m.name for m in self.medium.materials]

    def sources(self):
        """Citations of the tissue properties used."""
        out = []
        seen = set()
        for m in self.medium.materials:
            key = m.source.split("'")[1] if "'" in m.source else m.name
            for c in tissues.cite(key):
                if c not in seen:
                    seen.add(c)
                    out.append(c)
        return out


class _Builder:
    """Collects regions and their tissues in painting order."""

    def __init__(self, grid):
        self.grid = grid
        self.labels = np.zeros(grid.shape, int)
        self.materials = []
        self.index = {}

    def material(self, label, tissue=None, **override):
        if label not in self.index:
            self.index[label] = len(self.materials)
            self.materials.append(tissues.material(tissue or label, label=label, **override))
        return self.index[label]

    def fill(self, label, tissue=None, **override):
        self.labels[...] = self.material(label, tissue, **override)

    def paint(self, mask, label, tissue=None, **override):
        i = self.material(label, tissue, **override)
        self.labels[np.broadcast_to(mask, self.labels.shape)] = i

    def medium(self):
        used = np.unique(self.labels)
        remap = -np.ones(len(self.materials), int)
        remap[used] = np.arange(used.size)
        mats = [self.materials[i] for i in used]
        return Medium(self.grid, remap[self.labels], mats)


def _z(grid):
    return grid.mesh()[-1]


def _lateral(grid):
    m = grid.mesh()
    x = m[0] if grid.ndim >= 2 else np.zeros(grid.shape)
    y = m[1] if grid.ndim == 3 else np.zeros(grid.shape)
    return x, y


# --------------------------------------------------------------------------
# breast


def breast(grid, lesions=(), skin=1.45e-3, fat=5e-3, gland=15e-3, retro_fat=2e-3,
           muscle=8e-3, density=0.6, ligaments=True, ribs=False, rng=None):
    """Breast seen by a linear array on the skin.

    Layers from the skin down: skin (dermis), subcutaneous fat with Cooper's
    ligaments, fibroglandular tissue mixed with fat lobules, retromammary
    fat, pectoralis muscle and, optionally, ribs.

    Parameters
    ----------
    lesions : list of Lesion
        For example [Cyst((x, 0, z)), Fibroadenoma(...), Carcinoma(...)].
    skin : float
        Skin thickness; default 1.45 mm, the mean measured by dedicated
        breast CT (Shi et al. 2013, Med. Phys. 40:031913; Huang et al.
        2008, Med. Phys. 35:1199).
    fat, gland, retro_fat, muscle : float
        Mean thickness of the layers.
    density : float
        Fraction of the glandular layer occupied by fibroglandular tissue
        (the rest are fat lobules): low for a fatty breast, high for a
        dense one. The default 0.6 is a modelling choice.
    ligaments : bool
        Draw Cooper's ligaments through the subcutaneous fat.
    ribs : bool
        Add rib cross-sections below the muscle (fluid model of bone).
    """
    rng = np.random.default_rng(rng)
    b = _Builder(grid)
    corr = 6e-3

    b.fill("skin", "dermis")
    top_fat = skin
    gland_top = top_fat + fat
    gland_bot = gland_top + gland
    retro_bot = gland_bot + retro_fat
    musc_bot = retro_bot + muscle

    b.paint(ph.wavy_below(grid, top_fat, 0.15e-3, corr, rng), "subcutaneous fat", "fat")
    gland_mask = ph.wavy_below(grid, gland_top, 0.25 * fat, corr, rng)
    b.paint(gland_mask, "fibroglandular tissue")
    if density < 1:
        lob = np.asarray(ph.random_field(grid, 2.0e-3, rng))
        thr = np.quantile(lob, density)
        b.paint(gland_mask & (lob > thr), "fat lobules", "fat")
    b.paint(ph.wavy_below(grid, gland_bot, 0.1 * gland, corr * 1.5, rng), "retromammary fat",
            "fat")
    b.paint(ph.wavy_below(grid, retro_bot, 0.5e-3, corr * 2, rng), "pectoralis", "muscle")
    deeper = ph.wavy_below(grid, musc_bot, 0.5e-3, corr * 2, rng)
    b.paint(deeper, "chest wall", "muscle", echo=-8.0)

    if ligaments and grid.ndim >= 2:
        width = 0.25e-3
        xs = np.arange(grid.origin[0], grid.origin[0] + grid.extent[0], 4e-3)
        xs = xs + rng.uniform(-1e-3, 1e-3, xs.size)
        band = np.zeros(grid.shape, bool)
        for x0 in xs:
            bend = rng.uniform(-3e-3, 3e-3)
            pts = [(x0, 0.0, gland_top + 0.5e-3), (x0 + 0.5 * bend, 0.0, 0.5 * (skin + gland_top)),
                   (x0 + bend, 0.0, skin)]
            band |= ph.curve(grid, pts, width)
        fat_now = b.labels == b.index["subcutaneous fat"]
        b.paint(band & fat_now, "cooper ligament")

    if ribs and grid.ndim >= 2:
        zc = musc_bot + 6e-3
        for xr in np.arange(grid.origin[0] + 5e-3, grid.origin[0] + grid.extent[0], 20e-3):
            b.paint(ph.ellipsoid(grid, (xr, 0.0, zc), (6e-3, 1.0, 4e-3)), "rib", "bone")

    targets = []
    for les in lesions:
        b.paint(les.mask(grid, rng), les.name, les.tissue)
        targets.append(les.name)

    return OrganPhantom(
        b.medium(), targets,
        description=(f"breast: skin {skin * 1e3:.2f} mm, subcutaneous fat {fat * 1e3:.1f} mm, "
                     f"glandular layer {gland * 1e3:.1f} mm (density {density:.2f}), "
                     f"retromammary fat {retro_fat * 1e3:.1f} mm, pectoralis "
                     f"{muscle * 1e3:.1f} mm; lesions: {', '.join(targets) or 'none'}"),
        probe="linear array on the skin at z = 0, 7-15 MHz",
    )


# --------------------------------------------------------------------------
# prostate


def prostate(grid, tumors=(), calcifications=(), rectal_wall=3e-3, fat=1.5e-3,
             size=(45e-3, 35e-3, 30e-3), tz_fraction=0.45, rng=None):
    """Prostate seen from the rectum (transrectal probe at z = 0).

    Transverse plane in 2-D (x lateral, z anterior). The gland is an
    ellipsoid of the given width, length and anteroposterior size; the
    default 45 x 35 x 30 mm gives about 25 mL, close to the mean volume of
    26.9 mL of the patients treated by Pichardo et al. 2008 (Ultrasound
    Med. Biol. 34:1105). The transition zone fills the anterior part
    around the urethra, the peripheral zone the posterior and lateral part
    facing the probe, with a thickness inside the 1.6-15 mm range measured
    by Sellers et al. 2022 (Res. Rep. Urol. 14:265). The anterior
    fibromuscular stroma covers the front. A 3 mm rectal wall separates
    the probe from the gland.

    Parameters
    ----------
    tumors : list of Lesion, for example [Tumor((8e-3, 0, 10e-3), radius=4e-3)]
    calcifications : list of (x, y, z) centres of small calcifications
    tz_fraction : float
        Size of the transition zone relative to the gland (grows with age).
        The default leaves a peripheral zone about 9 mm thick behind it.
    """
    rng = np.random.default_rng(rng)
    b = _Builder(grid)
    w, L, ap = size
    zc = rectal_wall + fat + ap / 2

    b.fill("periprostatic fat", "fat")
    b.paint(ph.wavy_below(grid, 0.0, 0.0, 5e-3, rng) & (_z(grid) < rectal_wall),
            "rectal wall")
    if grid.ndim >= 2:
        x, _ = _lateral(grid)
        # levator ani on both sides
        for s in (-1, 1):
            b.paint(ph.ellipsoid(grid, (s * (w / 2 + 6e-3), 0.0, zc), (3e-3, L, ap * 0.7)),
                    "levator ani", "muscle")
    gland = ph.irregular(grid, (0.0, 0.0, zc), ap / 2, roughness=0.03, correlation=ap,
                         rng=rng, aspect=(w / ap, L / ap, 1.0))
    b.paint(gland, "peripheral zone", "prostate")
    z_tz = zc + 0.06 * ap
    tz = ph.ellipsoid(grid, (0.0, 0.0, z_tz),
                      (tz_fraction * w * 0.7, tz_fraction * L * 0.7, tz_fraction * ap * 0.58))
    b.paint(tz & gland, "transition zone", "prostate transition zone")
    stroma = gland & (_z(grid) > zc + 0.38 * ap)
    b.paint(stroma, "anterior stroma", "muscle", echo=-10.0)
    b.paint(ph.sphere(grid, (0.0, 0.0, z_tz), 1.5e-3), "urethra", "prostate", echo=4.0)

    for cc in calcifications:
        b.paint(ph.sphere(grid, cc, 0.6e-3), "calcification")
    targets = []
    for t in tumors:
        b.paint(t.mask(grid, rng) & gland, t.name, t.tissue)
        targets.append(t.name)
    vol = np.pi / 6 * w * L * ap
    return OrganPhantom(
        b.medium(), targets,
        description=(f"prostate {w * 1e3:.0f} x {L * 1e3:.0f} x {ap * 1e3:.0f} mm "
                     f"({vol * 1e6:.1f} mL) behind a {rectal_wall * 1e3:.1f} mm rectal wall; "
                     f"targets: {', '.join(targets) or 'none'}"),
        probe="transrectal convex or side-fire array at z = 0, 6-9 MHz; HIFU probe in the "
              "rectum",
    )


# --------------------------------------------------------------------------
# uterus


def uterus(grid, fibroids=(), approach="transabdominal", skin=2e-3, fat=15e-3, muscle=10e-3,
           bladder=40e-3, size=(80e-3, 50e-3, 45e-3), tilt=np.deg2rad(20),
           endometrium=8e-3, rng=None):
    """Uterus in the sagittal plane.

    approach="transabdominal": probe on the abdomen; from the skin down:
    skin, subcutaneous fat, rectus muscle, a full bladder (wall 3 mm, the
    mean of Hakenberg et al. 2000, Neurourol. Urodyn. 19:585) and the
    uterus behind it, surrounded by pelvic fat. approach="transvaginal":
    probe in the vaginal fornix, the uterus starts a few millimetres away.

    The default uterus is 80 mm long, 50 mm wide and 45 mm thick, inside the
    range of the mean sizes measured by ultrasound in women of reproductive
    age, 72.8 x 42.8 x 32.4 mm (nulliparous) to 90.8 x 51.7 x 43.0 mm
    (multiparous) (Esmaelzadeh et al. 2004, East. Mediterr. Health J.
    10:437). It is tilted by tilt from the lateral axis (anteverted).

    Parameters
    ----------
    fibroids : list of Lesion, for example [Tumor((5e-3, 0, 85e-3), radius=15e-3,
        tissue="fibroid")]
    endometrium : float
        Thickness of the endometrial stripe.
    """
    rng = np.random.default_rng(rng)
    b = _Builder(grid)
    length, width, thick = size
    if approach == "transabdominal":
        b.fill("skin", "dermis")
        top = skin
        b.paint(ph.wavy_below(grid, top, 0.3e-3, 10e-3, rng), "subcutaneous fat", "fat")
        top += fat
        b.paint(ph.wavy_below(grid, top, 1e-3, 15e-3, rng), "rectus muscle", "muscle")
        top += muscle
        b.paint(ph.wavy_below(grid, top, 0.5e-3, 15e-3, rng), "preperitoneal fat", "fat")
        top += 2e-3
        b.paint(ph.halfspace(grid, top), "bladder wall")
        b.paint(ph.halfspace(grid, top + 3e-3), "urine")
        top += 3e-3 + bladder
        b.paint(ph.halfspace(grid, top), "bladder wall")
        top += 3e-3
        b.paint(ph.halfspace(grid, top), "pelvic fat", "fat")
        zc = top + thick / 2 + 2e-3
    elif approach == "transvaginal":
        b.fill("pelvic fat", "fat")
        zc = 8e-3 + thick / 2
    else:
        raise ValueError("approach must be 'transabdominal' or 'transvaginal'")

    axes_ = (length / 2, width / 2, thick / 2)
    body = ph.ellipsoid(grid, (0.0, 0.0, zc), axes_, angle=-tilt)
    b.paint(body, "myometrium")
    # cervix continues the body on the negative x side
    ccx = -np.cos(tilt) * (length / 2 + 10e-3)
    ccz = zc + np.sin(tilt) * (length / 2 + 10e-3)
    b.paint(ph.ellipsoid(grid, (ccx, 0.0, ccz), (18e-3, 14e-3, 13e-3), angle=-tilt), "cervix")
    endo = ph.ellipsoid(grid, (0.0, 0.0, zc), (0.38 * length, 0.3 * width, endometrium / 2),
                        angle=-tilt)
    b.paint(endo, "endometrium")
    targets = []
    for f in fibroids:
        b.paint(f.mask(grid, rng), f.name, f.tissue)
        targets.append(f.name)
    return OrganPhantom(
        b.medium(), targets,
        description=(f"uterus {length * 1e3:.0f} x {width * 1e3:.0f} x {thick * 1e3:.0f} mm, "
                     f"{approach}; targets: {', '.join(targets) or 'none'}"),
        probe=("convex array on the abdomen at z = 0, 2-5 MHz; HIFU through the abdominal "
               "wall" if approach == "transabdominal" else
               "endocavity convex array at z = 0, 5-9 MHz"),
    )


# --------------------------------------------------------------------------
# face


def face(grid, epidermis=0.1e-3, dermis=1.3e-3, fat=2.6e-3, smas=1.0e-3, muscle=8e-3,
         bone=False, rng=None):
    """Facial skin and soft tissue for aesthetic focused ultrasound.

    Layers from the surface: epidermis, dermis, superficial fat, SMAS,
    facial muscle (and optionally bone). The defaults give a full skin
    thickness of 1.4 mm, inside the 1.31-1.64 mm range measured by
    ultrasound over eight facial sites (Jeong et al. 2023, Skin Res.
    Technol. 29:e13426), and put the SMAS between 4.0 and 5.0 mm, the depth
    reached by the 4.5 mm micro-focused ultrasound transducer (Chaves
    Bellote and Miot 2021, Clin. Cosmet. Investig. Dermatol. 14:1613).
    """
    rng = np.random.default_rng(rng)
    b = _Builder(grid)
    b.fill("epidermis")
    top = epidermis
    b.paint(ph.halfspace(grid, top), "dermis")
    top += dermis
    b.paint(ph.wavy_below(grid, top, 0.08e-3, 3e-3, rng), "superficial fat", "fat")
    top += fat
    b.paint(ph.wavy_below(grid, top, 0.15e-3, 4e-3, rng), "smas")
    top += smas
    b.paint(ph.wavy_below(grid, top, 0.15e-3, 4e-3, rng), "facial muscle", "muscle")
    top += muscle
    if bone:
        b.paint(ph.halfspace(grid, top), "bone")
    return OrganPhantom(
        b.medium(), ["smas"],
        description=(f"face: epidermis {epidermis * 1e3:.2f} mm, dermis {dermis * 1e3:.2f} mm, "
                     f"fat {fat * 1e3:.1f} mm, SMAS {smas * 1e3:.1f} mm at "
                     f"{(epidermis + dermis + fat) * 1e3:.1f} mm depth"),
        probe="micro-focused transducer on the skin at z = 0 (4 MHz / 4.5 mm for the SMAS, "
              "7 MHz / 3.0 mm, 10 MHz / 1.5 mm)",
    )
