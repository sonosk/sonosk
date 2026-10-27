"""Tissue property database with a published source for every value.

Each tissue is a set of properties. Every property carries its value, the
key of the reference it comes from (see REFERENCES) and a short note on
how it was measured or derived. Use material() to get a Material for a
phantom, and info() or table() to see where the numbers come from.

Properties and units
--------------------
c          speed of sound [m/s]
rho        density [kg/m^3]
alpha0     attenuation coefficient [dB/(cm MHz^y)]
y          exponent of the attenuation power law
ba         nonlinearity parameter B/A
k          thermal conductivity [W/(m K)]
cp         specific heat capacity [J/(kg K)]
perfusion  blood perfusion [mL/(min kg)], converted to kg/(m^3 s) in material()
echo       mean backscatter level relative to breast fat [dB]

Echo levels are the least certain numbers in the table. Where measured
backscatter coefficients exist they are cited; elsewhere the level is a
modelling choice that reproduces the usual B-mode appearance of the tissue
(hyperechoic, isoechoic, hypoechoic), and the note says so.

Density, thermal properties and perfusion come from the IT'IS tissue
property database V5.0 where no tissue-specific measurement was found;
the spread of those values is in RANGES, provenance() says how each value
was obtained, and sample() draws materials within the spread for Monte
Carlo runs. to_csv() writes the whole table.
The few values that rest on a modelling choice carry the key "model".
pending() lists placeholders still to be checked; there are none in this
version.
"""

import numpy as np

from . import water as _water
from .medium import Material

BLOOD_DENSITY = 1050.0   # kg/m^3, used to convert perfusion units

REFERENCES = {
    "li2009": "Li C., Duric N., Littrup P., Huang L. In vivo breast sound-speed imaging with "
              "ultrasound tomography. Ultrasound Med. Biol. 35(10):1615-1628, 2009. "
              "doi:10.1016/j.ultrasmedbio.2009.05.011",
    "nasief2015": "Nasief H.G., Rosado-Mendez I.M., Zagzebski J.A., Hall T.J. Acoustic "
                  "properties of breast fat. J. Ultrasound Med. 34(11):2007-2016, 2015. "
                  "doi:10.7863/ultra.14.07039",
    "nam2013": "Nam K., Zagzebski J.A., Hall T.J. Quantitative assessment of in vivo breast "
               "masses using ultrasound attenuation and backscatter. Ultrason. Imaging "
               "35(2):146-161, 2013. doi:10.1177/0161734613480281",
    "katzhanani2014": "Katz-Hanani I., Rothstein T., Gaitini D., Gallimidi Z., Azhari H. "
                      "Age-related ultrasonic properties of breast tissue in vivo. Ultrasound "
                      "Med. Biol. 40(9):2265-2271, 2014. doi:10.1016/j.ultrasmedbio.2014.03.034",
    "culjat2010": "Culjat M.O., Goldenberg D., Tewari P., Singh R.S. A review of tissue "
                  "substitutes for ultrasound imaging. Ultrasound Med. Biol. 36(6):861-873, "
                  "2010. doi:10.1016/j.ultrasmedbio.2010.02.012 (Table 1, values from ICRU "
                  "Report 61, 1998)",
    "moran1995": "Moran C.M., Bush N.L., Bamber J.C. Ultrasonic propagation properties of "
                 "excised human skin. Ultrasound Med. Biol. 21(9):1177-1190, 1995. "
                 "doi:10.1016/0301-5629(95)00049-6",
    "keshavarzi2001": "Keshavarzi A., Vaezy S., Kaczkowski P.J., et al. Attenuation "
                      "coefficient and sound speed in human myometrium and uterine fibroid "
                      "tumors. J. Ultrasound Med. 20(5):473-480, 2001. "
                      "doi:10.7863/jum.2001.20.5.473",
    "kiss2011": "Kiss M.Z., Varghese T., Kliewer M.A. Ex vivo ultrasound attenuation "
                "coefficient for human cervical and uterine tissue from 5 to 10 MHz. "
                "Ultrasonics 51(4):467-471, 2011. doi:10.1016/j.ultras.2010.11.012",
    "suomi2018": "Suomi V., Treeby B., Jaros J., et al. Transurethral ultrasound therapy of "
                 "the prostate in the presence of calcifications: a simulation study. Med. "
                 "Phys. 45(11):4793-4805, 2018. doi:10.1002/mp.13183",
    "law1985": "Law W.K., Frizzell L.A., Dunn F. Determination of the nonlinearity parameter "
               "B/A of biological media. Ultrasound Med. Biol. 11(2):307-318, 1985. "
               "doi:10.1016/0301-5629(85)90130-9",
    "valvano1985": "Valvano J.W., Cochran J.R., Diller K.R. Thermal conductivity and "
                   "diffusivity of biomaterials measured with self-heated thermistors. Int. J. "
                   "Thermophys. 6(3):301-311, 1985; values as tabulated by Rossmann C., "
                   "Haemmerich D. Crit. Rev. Biomed. Eng. 42(6):467-492, 2014. "
                   "doi:10.1615/critrevbiomedeng.2015012486",
    "vaupel2022": "Vaupel P., Piazena H. Strong correlation between specific heat capacity "
                  "and water content in human tissues suggests preferred heat deposition in "
                  "malignant tumors upon electromagnetic irradiation. Int. J. Hyperthermia "
                  "39(1):987-997, 2022. doi:10.1080/02656736.2022.2067596",
    "marczak1997": "Marczak W. Water as a standard in the measurements of speed of sound in "
                   "liquids. J. Acoust. Soc. Am. 102(5):2776-2779, 1997. doi:10.1121/1.420332",
    "kell1975": "Kell G.S. Density, thermal expansivity, and compressibility of liquid water "
                "from 0 to 150 C. J. Chem. Eng. Data 20(1):97-105, 1975. "
                "doi:10.1021/je60064a005",
    "birads": "Echo level chosen to reproduce the usual B-mode appearance of the tissue "
              "(ACR BI-RADS ultrasound lexicon; for breast masses see also Nam et al. 2013). "
              "A modelling choice, not a measurement.",
    "itis2025": "IT'IS Foundation. Tissue Properties Database V5.0, 21 August 2025. "
                "doi:10.13099/VIP21000-05-0 (average values; attenuation converted from "
                "Np/(m MHz^y) with 1 Np = 8.686 dB)",
    "pinkerton1949": "Pinkerton J.M.M. The absorption of ultrasonic waves in liquids and its "
                     "relation to molecular constitution. Proc. Phys. Soc. B 62(2):129-141, "
                     "1949. doi:10.1088/0370-1301/62/2/307",
    "model": "Modelling choice where no measurement fits; the note gives the reason.",
}

# Each property: (value, reference key, note)
_P = dict


def _w(prop, temp=37.0):
    if prop == "c":
        return (round(float(_water.sound_speed(temp)), 1), "marczak1997",
                "water at 37 C")
    return (round(float(_water.density(temp)), 1), "kell1975", "water at 37 C")


TISSUES = {
    # ------------------------------------------------------------------ shared
    "fat": _P(
        group="shared",
        c=(1422.0, "li2009", "breast fat in vivo, ultrasound tomography, 61 women; used for "
                              "all adipose layers"),
        rho=(911.0, "itis2025", "fat, 23 samples"),
        alpha0=(0.73, "nasief2015", "breast fat in vivo, slope at 7 MHz, 24 women"),
        y=(1.0, "nasief2015", "linear frequency dependence assumed for the slope"),
        ba=(11.0, "law1985", "fat: B/A around 11"),
        k=(0.211, "itis2025", "fat"),
        cp=(2348.0, "itis2025", "fat"),
        perfusion=(32.7, "itis2025", "fat"),
        echo=(0.0, "nasief2015", "reference level; breast fat is the BI-RADS standard of "
                                 "comparison"),
    ),
    "muscle": _P(
        group="shared",
        c=(1588.4, "itis2025", "skeletal muscle, mean of 24 values"),
        rho=(1090.4, "itis2025", "skeletal muscle, 14 samples"),
        alpha0=(0.617, "itis2025", "7.11 Np/(m MHz^y)"),
        y=(1.08, "itis2025", ""),
        ba=(7.0, "law1985", "muscle: B/A close to 7"),
        k=(0.495, "itis2025", ""),
        cp=(3421.0, "itis2025", ""),
        perfusion=(36.7, "itis2025", "resting muscle"),
        echo=(-6.0, "birads", "muscle is hypoechoic to fat"),
    ),
    "water": _P(
        group="shared",
        c=_w("c"),
        rho=_w("rho"),
        alpha0=(0.0022, "itis2025", "0.0253 Np/(m MHz) at 1 MHz"),
        y=(2.0, "pinkerton1949", "viscous absorption grows with f^2; IT'IS lists a linear "
                                 "law, which understates the loss above 1 MHz"),
        ba=(4.96, "itis2025", "room temperature"),
        k=(0.605, "itis2025", ""),
        cp=(4178.0, "itis2025", ""),
        perfusion=(0.0, "itis2025", ""),
        echo=(-np.inf, "birads", "anechoic"),
    ),
    "blood": _P(
        group="shared",
        c=(1578.2, "itis2025", ""),
        rho=(1049.8, "itis2025", "584 samples"),
        alpha0=(0.206, "itis2025", "2.37 Np/(m MHz^y)"),
        y=(1.05, "itis2025", ""),
        ba=(6.11, "itis2025", ""),
        k=(0.517, "itis2025", ""),
        cp=(3617.0, "itis2025", ""),
        perfusion=(0.0, "model", "flowing blood in a vessel; heat carried by flow is not a "
                                 "Pennes term"),
        echo=(-30.0, "birads", "nearly anechoic at clinical frequencies"),
    ),
    # ------------------------------------------------------------------ skin
    "epidermis": _P(
        group="skin",
        c=(1645.0, "moran1995", "excised skin, 20-30 MHz, 24 C"),
        rho=(1109.0, "itis2025", "skin"),
        alpha0=(0.44, "moran1995", "fit alpha1 f^n over 20-30 MHz"),
        y=(1.55, "moran1995", "fit over 20-30 MHz"),
        ba=(7.0, "law1985", "soft tissue value"),
        k=(0.372, "itis2025", "skin"),
        cp=(3391.0, "itis2025", "skin"),
        perfusion=(0.0, "model", "the epidermis has no blood vessels"),
        echo=(15.0, "birads", "bright entry echo"),
    ),
    "dermis": _P(
        group="skin",
        c=(1595.0, "moran1995", "excised skin, 20-30 MHz, 24 C"),
        rho=(1109.0, "itis2025", "skin"),
        alpha0=(0.264, "moran1995", "fit alpha1 f^n over 20-30 MHz"),
        y=(1.69, "moran1995", "fit over 20-30 MHz"),
        ba=(7.0, "law1985", "soft tissue value"),
        k=(0.372, "itis2025", "skin"),
        cp=(3391.0, "itis2025", "skin"),
        perfusion=(106.4, "itis2025", "skin"),
        echo=(10.0, "birads", "dermis is hyperechoic to subcutaneous fat"),
    ),
    "smas": _P(
        group="skin",
        like="dermis",
        echo=(8.0, "birads", "fibrous layer, echogenic band in the subcutis"),
        note="superficial musculoaponeurotic system, given the acoustic and thermal "
             "properties of dermis (dense connective tissue); no direct measurement",
    ),
    # ------------------------------------------------------------------ breast
    "fibroglandular tissue": _P(
        group="breast",
        c=(1487.0, "li2009", "breast parenchyma in vivo, ultrasound tomography"),
        rho=(1020.0, "culjat2010", "breast"),
        alpha0=(0.75, "culjat2010", "breast; whole breasts in vivo range from 0.96 (over 60 "
                                    "years) to 1.27 (under 30) dB/(cm MHz), Katz-Hanani et "
                                    "al. 2014"),
        y=(1.0, "culjat2010", "slope in dB/(cm MHz)"),
        ba=(7.0, "law1985", "soft tissue value"),
        k=(0.335, "itis2025", "breast gland"),
        cp=(2960.0, "itis2025", "breast gland"),
        perfusion=(150.0, "itis2025", "breast gland"),
        echo=(10.0, "birads", "fibroglandular tissue is hyperechoic to fat"),
    ),
    "cooper ligament": _P(
        group="breast",
        like="dermis",
        echo=(14.0, "birads", "thin echogenic septa"),
        note="connective tissue band, given the properties of dermis",
    ),
    "breast cyst": _P(
        group="breast",
        like="water",
        echo=(-np.inf, "birads", "simple cyst: anechoic"),
        note="simple cyst fluid modelled as water at 37 C",
    ),
    "fibroadenoma": _P(
        group="breast",
        c=(1513.0, "li2009", "benign breast lesions in vivo, mean of the cohort"),
        rho=(1040.5, "itis2025", "breast gland"),
        alpha0=(0.94, "nam2013", "14 fibroadenomas in vivo, reference phantom method"),
        y=(1.0, "nam2013", "slope in dB/(cm MHz)"),
        ba=(7.0, "law1985", "soft tissue value"),
        k=(0.335, "itis2025", "breast gland"),
        cp=(2960.0, "itis2025", "breast gland"),
        perfusion=(150.0, "itis2025", "breast gland"),
        echo=(-3.0, "birads", "oval circumscribed mass, mildly hypoechoic to fat"),
    ),
    "breast carcinoma": _P(
        group="breast",
        c=(1548.0, "li2009", "malignant breast lesions in vivo, mean of the cohort"),
        rho=(1040.5, "itis2025", "breast gland"),
        alpha0=(1.16, "nam2013", "10 carcinomas in vivo, reference phantom method"),
        y=(1.0, "nam2013", "slope in dB/(cm MHz)"),
        ba=(7.0, "law1985", "soft tissue value"),
        k=(0.415, "valvano1985", "human breast adenocarcinoma, 37 C"),
        cp=(3750.0, "vaupel2022", "middle of the 3.6-3.9 J/(g K) range of malignant tumours"),
        perfusion=(150.0, "itis2025", "breast gland"),
        echo=(-6.0, "birads", "hypoechoic mass; most carcinomas have a lower backscatter "
                              "than fibroadenomas (Nam et al. 2013)"),
    ),
    # ---------------------------------------------------------------- prostate
    "prostate": _P(
        group="prostate",
        c=(1561.0, "suomi2018", "from Parker et al. 1993; with rho gives 1.63 MRayl"),
        rho=(1045.0, "suomi2018", "same value in IT'IS"),
        alpha0=(0.78, "suomi2018", "value used for prostate tissue in their simulations"),
        y=(1.1, "suomi2018", "exponent used with alpha0"),
        ba=(7.5, "suomi2018", ""),
        k=(0.51, "suomi2018", "IT'IS gives 0.512"),
        cp=(3400.0, "suomi2018", "freshly excised prostate (Patch et al. 2011)"),
        perfusion=(394.1, "itis2025", "normal prostate"),
        echo=(-6.0, "birads", "peripheral zone: homogeneous, isoechoic, darker than "
                              "periprostatic fat"),
    ),
    "prostate transition zone": _P(
        group="prostate",
        like="prostate",
        echo=(-9.0, "birads", "transition zone: hypoechoic and heterogeneous with age"),
    ),
    "prostate carcinoma": _P(
        group="prostate",
        like="prostate",
        perfusion=(93.0, "suomi2018", "1.7 kg/(m^3 s), measured in patients with locally "
                                      "advanced carcinoma (van Vulpen et al. 2002)"),
        echo=(-14.0, "birads", "classically hypoechoic in the peripheral zone"),
        note="acoustic and thermal properties of normal prostate, with the lower perfusion "
             "measured in carcinoma; no measured in vivo acoustic values",
    ),
    "calcification": _P(
        group="prostate",
        c=(2184.0, "suomi2018", "renal stones (Singh and Agarwal 1989); with rho gives "
                                "3.20 MRayl"),
        rho=(1466.0, "suomi2018", "renal stones"),
        alpha0=(2.64, "suomi2018", "value used for prostatic calcifications"),
        y=(1.1, "suomi2018", "exponent used with alpha0"),
        ba=(7.5, "suomi2018", ""),
        k=(3.85, "suomi2018", "calcium carbonate"),
        cp=(858.0, "suomi2018", "calcium carbonate"),
        perfusion=(0.0, "suomi2018", ""),
        echo=(25.0, "birads", "bright echo with acoustic shadow"),
    ),
    "rectal wall": _P(
        group="prostate",
        like="muscle",
        echo=(-4.0, "birads", "layered muscular wall"),
        note="given the properties of muscle",
    ),
    "bladder wall": _P(
        group="pelvis",
        like="muscle",
        echo=(2.0, "birads", "thin echogenic wall"),
        note="given the properties of muscle",
    ),
    "urine": _P(
        group="pelvis",
        like="water",
        echo=(-np.inf, "birads", "anechoic"),
        note="modelled as water at 37 C",
    ),
    # ------------------------------------------------------------------ uterus
    "myometrium": _P(
        group="uterus",
        c=(1613.0, "keshavarzi2001", "fresh ex vivo, 1-3 MHz: 1611-1616 m/s for myometrium "
                                     "and fibroid"),
        rho=(1104.5, "itis2025", "uterus"),
        alpha0=(0.316, "kiss2011", "power law through the 5 and 10 MHz values of the "
                                   "abstract (4.15 and 12.6 dB/cm)"),
        y=(1.60, "kiss2011", "same fit"),
        ba=(7.0, "law1985", "soft tissue value"),
        k=(0.527, "itis2025", "uterus"),
        cp=(3676.0, "itis2025", "uterus"),
        perfusion=(457.5, "itis2025", "uterus"),
        echo=(-4.0, "birads", "homogeneous, intermediate echo level"),
    ),
    "fibroid": _P(
        group="uterus",
        c=(1613.0, "keshavarzi2001", "fresh ex vivo, 1-3 MHz"),
        rho=(1104.5, "itis2025", "uterus"),
        alpha0=(0.357, "kiss2011", "leiomyoma: power law through 5.1 and 16.0 dB/cm at 5 and "
                                   "10 MHz"),
        y=(1.65, "kiss2011", "same fit"),
        ba=(7.0, "law1985", "soft tissue value"),
        k=(0.527, "itis2025", "uterus"),
        cp=(3676.0, "itis2025", "uterus"),
        perfusion=(100.0, "model", "fibroids chosen for HIFU are usually hypovascular, so a "
                                   "value well below that of myometrium; perfusion varies "
                                   "widely between fibroids"),
        echo=(-8.0, "birads", "usually hypoechoic whorled mass"),
    ),
    "cervix": _P(
        group="uterus",
        c=(1629.0, "itis2025", "cervix"),
        rho=(1104.5, "itis2025", "cervix"),
        alpha0=(0.218, "kiss2011", "power law through 6.3 and 26.8 dB/cm at 5 and 10 MHz"),
        y=(2.09, "kiss2011", "same fit"),
        ba=(7.0, "law1985", "soft tissue value"),
        k=(0.527, "itis2025", "cervix"),
        cp=(3676.0, "itis2025", "cervix"),
        perfusion=(700.0, "itis2025", "cervix"),
        echo=(-2.0, "birads", "intermediate echo level"),
    ),
    "endometrium": _P(
        group="uterus",
        like="myometrium",
        echo=(6.0, "birads", "echogenic stripe (secretory phase)"),
        note="acoustic and thermal properties of myometrium",
    ),
    # ------------------------------------------------------------------ bone
    "bone": _P(
        group="shared",
        c=(3515.0, "itis2025", "cortical bone, compressional wave"),
        rho=(1908.0, "itis2025", "cortical bone, 27 samples"),
        alpha0=(4.74, "itis2025", "cortical bone, 54.6 Np/(m MHz)"),
        y=(1.0, "itis2025", ""),
        ba=(7.0, "model", "no measurement; enters only through the nonlinear term"),
        k=(0.32, "itis2025", "cortical bone"),
        cp=(1313.0, "itis2025", "cortical bone"),
        perfusion=(10.0, "itis2025", "cortical bone"),
        echo=(20.0, "birads", "bright surface with shadow"),
        note="fluid model: shear waves are not included",
    ),
}

PROPERTIES = ("c", "rho", "alpha0", "y", "ba", "k", "cp", "perfusion", "echo")
UNITS = {"c": "m/s", "rho": "kg/m^3", "alpha0": "dB/(cm MHz^y)", "y": "", "ba": "",
         "k": "W/(m K)", "cp": "J/(kg K)", "perfusion": "mL/(min kg)", "echo": "dB"}

ALIASES = {
    "subcutaneous fat": "fat",
    "breast fat": "fat",
    "adipose": "fat",
    "retromammary fat": "fat",
    "periprostatic fat": "fat",
    "abdominal fat": "fat",
    "pectoralis": "muscle",
    "pectoralis muscle": "muscle",
    "rectus muscle": "muscle",
    "facial muscle": "muscle",
    "skin": "dermis",
    "glandular tissue": "fibroglandular tissue",
    "parenchyma": "fibroglandular tissue",
    "cyst": "breast cyst",
    "carcinoma": "breast carcinoma",
    "peripheral zone": "prostate",
    "prostate peripheral zone": "prostate",
    "transition zone": "prostate transition zone",
    "leiomyoma": "fibroid",
    "uterus": "myometrium",
}


# Spread of the IT'IS V5.0 values used above: (standard deviation, minimum,
# maximum, number of samples or studies). A standard deviation of 0 with
# n = 1 means a single source.
RANGES = {
    ("fat", "rho"): (51.42, 812, 961, 23),
    ("fat", "k"): (0.02113, 0.18, 0.2416, 8),
    ("fat", "cp"): (371.8, 1806, 2973, 6),
    ("fat", "perfusion"): (12.86, 19.832, 63.248, 12),
    ("muscle", "c"): (21.56, 1545, 1631, 24),
    ("muscle", "rho"): (55.14, 1041, 1178, 14),
    ("muscle", "k"): (0.04211, 0.4225, 0.56, 8),
    ("muscle", "cp"): (460.2, 2624, 3799, 5),
    ("muscle", "perfusion"): (13.1, 18.5367, 94.2467, 58),
    ("water", "ba"): (0, 4.96, 4.96, 1),
    ("water", "k"): (0.006364, 0.6, 0.609, 2),
    ("water", "cp"): (0, 4178, 4178, 1),
    ("water", "perfusion"): (0, 0, 0, 1),
    ("blood", "c"): (11.27, 1559.2, 1590, 9),
    ("blood", "rho"): (19.17, 1025, 1060, 584),
    ("blood", "ba"): (0.1315, 6, 6.3, 4),
    ("blood", "k"): (0.02546, 0.488, 0.561, 7),
    ("blood", "cp"): (301.4, 3300, 3900, 3),
    ("epidermis", "rho"): (37.53, 1100, 1125, 10),
    ("epidermis", "k"): (0.05905, 0.322, 0.498, 8),
    ("epidermis", "cp"): (233.3, 3150, 3662, 4),
    ("dermis", "rho"): (37.53, 1100, 1125, 10),
    ("dermis", "k"): (0.05905, 0.322, 0.498, 8),
    ("dermis", "cp"): (233.3, 3150, 3662, 4),
    ("dermis", "perfusion"): (36.73, 49, 174.606, 12),
    ("fibroglandular tissue", "k"): (0.01768, 0.322, 0.347, 2),
    ("fibroglandular tissue", "cp"): (0, 2960, 2960, 1),
    ("fibroglandular tissue", "perfusion"): (0, 150, 150, 1),
    ("fibroadenoma", "rho"): (49.11, 990, 1092, 8),
    ("fibroadenoma", "k"): (0.01768, 0.322, 0.347, 2),
    ("fibroadenoma", "cp"): (0, 2960, 2960, 1),
    ("fibroadenoma", "perfusion"): (0, 150, 150, 1),
    ("breast carcinoma", "rho"): (49.11, 990, 1092, 8),
    ("breast carcinoma", "perfusion"): (0, 150, 150, 1),
    ("prostate", "perfusion"): (0, 394.12, 394.12, 1),
    ("myometrium", "rho"): (78.71, 1052, 1157, 6),
    ("myometrium", "k"): (0.02192, 0.511, 0.542, 2),
    ("myometrium", "cp"): (0, 3676, 3676, 1),
    ("myometrium", "perfusion"): (407.5, 170, 1055, 4),
    ("fibroid", "rho"): (78.71, 1052, 1157, 6),
    ("fibroid", "k"): (0.02192, 0.511, 0.542, 2),
    ("fibroid", "cp"): (0, 3676, 3676, 1),
    ("cervix", "c"): (5.657, 1625, 1633, 2),
    ("cervix", "rho"): (78.71, 1052, 1157, 6),
    ("cervix", "k"): (0.02192, 0.511, 0.542, 2),
    ("cervix", "cp"): (0, 3676, 3676, 1),
    ("cervix", "perfusion"): (0, 700, 700, 1),
    ("bone", "c"): (420.3, 2660, 4200, 21),
    ("bone", "rho"): (166, 1800, 2100, 27),
    ("bone", "k"): (0.03464, 0.3, 0.36, 3),
    ("bone", "cp"): (294.7, 826, 1650, 6),
    ("bone", "perfusion"): (0, 10, 10, 1),
}

# How each value was obtained, by reference key. Values whose note mentions
# a fit are "fitted"; inherited values are "inherited".
_KIND = {
    "itis2025": "database average",
    "suomi2018": "used in a simulation study",
    "culjat2010": "review compilation",
    "birads": "modelling choice",
    "model": "modelling choice",
}


def _resolve(name):
    key = name.lower().strip()
    key = ALIASES.get(key, key)
    if key not in TISSUES:
        raise KeyError(f"unknown tissue '{name}'; see sonosk.tissues.names()")
    return key


def names(group=None):
    """Names of the tissues in the database, optionally of one group."""
    return [n for n, t in TISSUES.items() if group is None or t["group"] == group]


def groups():
    return sorted({t["group"] for t in TISSUES.values()})


def properties(name):
    """All properties of a tissue as {property: (value, reference, note)},
    with inherited values resolved."""
    key = _resolve(name)
    t = TISSUES[key]
    out = {}
    if "like" in t:
        base = properties(t["like"])
        for p, v in base.items():
            out[p] = (v[0], v[1], (v[2] + "; " if v[2] else "") + f"value of {t['like']}")
    for p in PROPERTIES:
        if p in t:
            out[p] = t[p]
    return out


def perfusion_si(ml_min_kg, rho):
    """Perfusion in mL/(min kg) to blood mass flow per volume [kg/(m^3 s)]."""
    return ml_min_kg * 1e-6 / 60.0 * BLOOD_DENSITY * rho


def material(name, label=None, **override):
    """Material with the properties of a tissue.

    label is the name of the material in the phantom (default the name
    given). Keyword arguments override single properties, for example
    material("fat", echo=-2.0); overrides are recorded in the source text.
    """
    key = _resolve(name)
    label = label or name.lower().strip()
    props = properties(key)
    vals = {p: float(v[0]) for p, v in props.items()}
    vals.update({p: float(v) for p, v in override.items()})
    refs = sorted({v[1] for p, v in props.items() if p not in override})
    src = "sonosk.tissues '" + key + "': " + ", ".join(refs)
    if override:
        src += "; changed: " + ", ".join(sorted(override))
    return Material(
        name=label,
        c=vals["c"],
        rho=vals["rho"],
        alpha0=vals["alpha0"],
        y=vals["y"],
        ba=vals["ba"],
        k=vals["k"],
        cp=vals["cp"],
        perfusion=perfusion_si(vals["perfusion"], vals["rho"]),
        echo=vals["echo"],
        source=src,
    )


def info(name):
    """Rows (property, value, unit, reference key, note) for one tissue."""
    return [(p, v[0], UNITS[p], v[1], v[2]) for p, v in properties(name).items()]


def table(group=None):
    """Rows (tissue, property, value, unit, reference, note) for the whole
    database or one group."""
    rows = []
    for n in names(group):
        for r in info(n):
            rows.append((n,) + r)
    return rows


def pending():
    """(tissue, property) pairs whose value is still a placeholder."""
    out = []
    for n, t in TISSUES.items():
        for p in PROPERTIES:
            if p in t and t[p][1] == "pending":
                out.append((n, p))
    return out


def provenance(name):
    """{property: kind} for one tissue, where kind is one of "measured",
    "fitted", "database average", "used in a simulation study", "review
    compilation", "generic soft-tissue value", "inherited", "reference
    level" or "modelling choice"."""
    key = _resolve(name)
    out = {}
    for p, (v, ref, note) in properties(key).items():
        if "like" in TISSUES[key] and p not in TISSUES[key]:
            out[p] = "inherited"
        elif ref in _KIND:
            out[p] = _KIND[ref]
        elif p == "echo":
            out[p] = "reference level"
        elif "soft tissue value" in note:
            out[p] = "generic soft-tissue value"
        elif "assumed" in note:
            out[p] = "modelling choice"
        elif "fit" in note:
            out[p] = "fitted"
        else:
            out[p] = "measured"
    return out


def uncertainty(name, prop):
    """(standard deviation, minimum, maximum, n) of a value, or None when no
    spread is known. Inherited values take the spread of their source."""
    key = _resolve(name)
    while True:
        if (key, prop) in RANGES:
            return RANGES[(key, prop)]
        if "like" in TISSUES[key] and prop not in TISSUES[key]:
            key = TISSUES[key]["like"]
            continue
        return None


def sample(name, rng=None, label=None):
    """Material with properties drawn at random within their known spread.

    Each value with a known standard deviation is drawn from a normal
    distribution clipped to the published minimum and maximum; the other
    values stay fixed. For Monte Carlo runs over tissue uncertainty.
    """
    rng = np.random.default_rng(rng)
    key = _resolve(name)
    over = {}
    for p in PROPERTIES:
        u = uncertainty(key, p)
        if u is None or u[0] <= 0:
            continue
        sd, lo, hi, _ = u
        v = properties(key)[p][0]
        over[p] = float(np.clip(rng.normal(v, sd), lo, hi))
    return material(key, label=label, **over)


def to_csv(path, group=None):
    """Write the database (or one group) as a CSV file with value, unit,
    reference, provenance, spread and note of every property."""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tissue", "property", "value", "unit", "reference", "provenance", "sd",
                    "min", "max", "n", "note"])
        for n in names(group):
            kinds = provenance(n)
            for p, v, unit, ref, note in info(n):
                u = uncertainty(n, p) or ("", "", "", "")
                w.writerow([n, p, v, unit, ref, kinds[p], *u, note])


def reference(key):
    """Full citation of a reference key."""
    return REFERENCES[key]


def cite(name):
    """Citations used by one tissue."""
    keys = sorted({v[1] for v in properties(name).values()})
    return [REFERENCES[k] for k in keys]
