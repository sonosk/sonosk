import numpy as np
import pytest

import sonosk as sk
from sonosk import organs, phantom, tissues


def test_every_property_has_a_known_reference():
    for name in tissues.names():
        props = tissues.properties(name)
        assert set(props) == set(tissues.PROPERTIES), name
        for p, (value, ref, note) in props.items():
            assert ref in tissues.REFERENCES, (name, p, ref)
            assert np.isfinite(value) or p == "echo", (name, p)


def test_materials_are_physical():
    for name in tissues.names():
        m = tissues.material(name)
        assert 1300 < m.c < 4000
        assert 800 < m.rho < 2000
        assert m.alpha0 >= 0 and 0.5 <= m.y <= 2.5
        assert m.k > 0 and m.cp > 0 and m.perfusion >= 0


def test_aliases_labels_and_overrides():
    m = tissues.material("subcutaneous fat")
    assert m.name == "subcutaneous fat"
    assert m.c == tissues.material("fat").c
    o = tissues.material("fat", echo=-3.0)
    assert o.echo == -3.0 and "changed: echo" in o.source
    with pytest.raises(KeyError):
        tissues.material("unknown tissue")


def test_selected_values_match_their_papers():
    # values copied from the cited papers, checked once more here
    assert tissues.material("fat").c == 1422.0                  # Li et al. 2009
    assert tissues.material("fibroglandular tissue").c == 1487.0
    assert tissues.material("breast carcinoma").c == 1548.0
    assert tissues.material("breast carcinoma").alpha0 == 1.16  # Nam et al. 2013
    assert tissues.material("fibroadenoma").alpha0 == 0.94
    assert tissues.material("dermis").c == 1595.0               # Moran et al. 1995
    assert tissues.material("myometrium").c == 1613.0           # Keshavarzi et al. 2001


def test_uterine_power_laws_reproduce_the_measured_points():
    # Kiss et al. 2011: 4.15 and 12.6 dB/cm (uterus), 5.1 and 16.0 (leiomyoma),
    # 6.3 and 26.8 (cervix) at 5 and 10 MHz
    for name, a5, a10 in (("myometrium", 4.15, 12.6), ("fibroid", 5.1, 16.0),
                          ("cervix", 6.3, 26.8)):
        m = tissues.material(name)
        assert np.isclose(m.alpha0 * 5 ** m.y, a5, rtol=0.02)
        assert np.isclose(m.alpha0 * 10 ** m.y, a10, rtol=0.02)


def test_prostate_values_of_suomi_and_itis():
    # Suomi et al. 2018: impedances 1.63 and 3.20 MRayl, perfusion of
    # carcinoma 1.7 kg/(m^3 s)
    pr = tissues.material("prostate")
    ca = tissues.material("calcification")
    assert np.isclose(pr.impedance / 1e6, 1.63, atol=0.005)
    assert np.isclose(ca.impedance / 1e6, 3.20, atol=0.005)
    assert np.isclose(tissues.material("prostate carcinoma").perfusion, 1.7, rtol=0.01)
    # IT'IS V5.0 muscle: 7.1088 Np/(m MHz^1.0827)
    assert np.isclose(tissues.material("muscle").alpha0, 7.1088 * 20 / np.log(10) / 100,
                      rtol=0.002)
    assert tissues.pending() == []


def test_perfusion_conversion():
    # 60 mL/(min kg) in tissue of 1000 kg/m^3 is 1 mL/(s L) of blood
    assert np.isclose(tissues.perfusion_si(60.0, 1000.0), 1e-3 * tissues.BLOOD_DENSITY)


def test_breast_phantom_layers_and_lesions():
    g = sk.Grid.centered((201, 251), 0.2e-3, depth_start=0.0)
    ph = organs.breast(g, lesions=[organs.Cyst((-8e-3, 0, 14e-3)),
                                   organs.Carcinoma((8e-3, 0, 15e-3), radius=3e-3)], rng=0)
    med = ph.medium
    assert ph.targets == ["breast cyst", "breast carcinoma"]
    z = g.axis(1)
    col = med.labels[100]
    names = [med.materials[i].name for i in col]
    assert names[0] == "skin"
    assert names[np.argmin(np.abs(z - 3e-3))] in ("subcutaneous fat", "cooper ligament")
    assert np.isclose(med.volume("breast cyst"), np.pi * 3e-3 * 2.5e-3, rtol=0.05)
    assert ph.sources()


def test_prostate_volume_and_zones():
    g = sk.Grid.centered((61, 51, 51), 1e-3, depth_start=0.0)
    ph = organs.prostate(g, rng=1)
    med = ph.medium
    gland = sum(med.volume(n) for n in ("peripheral zone", "transition zone",
                                        "anterior stroma", "urethra"))
    assert 20e-6 < gland < 30e-6
    # peripheral zone lies between the rectum and the transition zone
    x0 = g.index((0, 0, 0))
    col = [med.materials[i].name for i in med.labels[x0[0], x0[1], :]]
    first_pz = col.index("peripheral zone")
    first_tz = col.index("transition zone")
    assert col[0] == "rectal wall" and first_pz < first_tz
    assert (first_tz - first_pz) * 1e-3 > 5e-3


def test_uterus_and_face():
    g = sk.Grid.centered((121, 131), 1e-3, depth_start=0.0)
    ph = organs.uterus(g, fibroids=[organs.Tumor((5e-3, 0, 95e-3), radius=10e-3,
                                                 tissue="fibroid", label="fibroid")], rng=0)
    assert "urine" in ph.names and ph.targets == ["fibroid"]
    gf = sk.Grid.centered((41, 161), 0.05e-3, depth_start=0.0)
    f = organs.face(gf, rng=0)
    z = gf.axis(1)
    col = [f.medium.materials[i].name for i in f.medium.labels[20]]
    smas = [zz for zz, n in zip(z, col) if n == "smas"]
    assert 3.6e-3 < min(smas) < 4.4e-3 and 4.6e-3 < max(smas) < 5.4e-3


def test_irregular_mass_and_fluctuations():
    g = sk.Grid.centered((121, 121), 0.1e-3, depth_start=0.0)
    m = phantom.irregular(g, (0, 0, 6e-3), 3e-3, roughness=0.2, spicules=6, rng=2)
    area = m.sum() * 1e-8
    assert 0.6 * np.pi * 9e-6 < area < 1.6 * np.pi * 9e-6
    med = sk.Medium(g, m.astype(int), [tissues.material("fat"),
                                       tissues.material("breast cyst")])
    f = phantom.fluctuations(med, level=0.02, rng=0)
    inside = m
    assert np.allclose(f.rho[inside], med.rho[inside])      # anechoic: no variation
    assert 0.015 < np.std(f.rho[~inside] / med.rho[~inside]) < 0.025


def test_random_field_statistics():
    g = sk.Grid((256, 256), 0.1e-3)
    f = np.asarray(phantom.random_field(g, 1e-3, rng=0))
    assert abs(f.mean()) < 1e-9 and np.isclose(f.std(), 1.0)
