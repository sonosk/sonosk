# Examples

Each figure in this guide comes from a script in the `examples` folder of
the repository. Run a script to see the figure and the printed numbers:

```
pip install "sonosk[plot]"
python examples/bmode_phantom.py
```

Material values in the examples are generic numbers chosen to show the
method. They are not measured tissue data.

## B-mode image of a test phantom

Background speckle, an anechoic cyst, a -10 dB and a +6 dB lesion and a row
of wire targets, imaged by a 128-element linear array at 5 MHz with nine
compounded plane waves.

```{image} figures/bmode_phantom.png
:alt: Phantom label map, B-mode image and quality measures
```

```{literalinclude} ../examples/bmode_phantom.py
:language: python
```

## HIFU ablation of a tumour

A 1 MHz bowl heats a tumour of 4 mm radius for 8 s. The single exposure
leaves most of the tumour untreated and a small lesion in front of and
behind it, which is what a treatment plan with several points has to fix.

```{image} figures/hifu_ablation.png
:alt: Peak temperature with lesion and tumour outlines, temperature at the focus, treatment report
```

```{literalinclude} ../examples/hifu_ablation.py
:language: python
```

## Field of a focused bowl

```{image} figures/bowl_field.png
:alt: Field of a focused bowl and its on-axis amplitude against O'Neil's solution
```

```{literalinclude} ../examples/bowl_field.py
:language: python
```

## Linear array beams

```{image} figures/array_beams.png
:alt: Focused, steered and apodized beams of a linear array
```

```{literalinclude} ../examples/array_beams.py
:language: python
```

## Line-focus source

```{image} figures/line_focus.png
:alt: Field of a cylindrically focused source in two planes
```

```{literalinclude} ../examples/line_focus.py
:language: python
```

## Element design with the Mason model

```{image} figures/transducer_stack.png
:alt: Electrical impedance and pulse-echo response with and without a matching layer
```

```{literalinclude} ../examples/transducer_stack.py
:language: python
```

## Mechanical index

```{image} figures/mechanical_index.png
:alt: On-axis pressure in water and derated, with the depth used for the MI
```

```{literalinclude} ../examples/mechanical_index.py
:language: python
```

## Water properties

```{image} figures/water_properties.png
:alt: Sound speed and density of water against temperature
```

```{literalinclude} ../examples/water_properties.py
:language: python
```
