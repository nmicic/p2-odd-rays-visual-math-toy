<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2025 Nenad Micic <nenad@micic.be>
-->

# Power-of-Two Square Odd Rays

**A visual math-toy exploration** (`p2-odd-rays-visual-math-toy`)

> **Warning / Disclaimer**
>
> This repository contains an AI-assisted exploratory visualization project.
> It is educational and experimental, intended for curiosity, illustration, and math-art.
> It is not peer-reviewed, not mathematical research, and not presented as a theorem or verified result.
>
> No practical evaluation, performance assessment, or engineering analysis has been performed.
> Nothing in this repository should be interpreted as providing system behavior, optimization, or functional guarantees.
>
> Visual patterns produced by this code are artifacts of the construction, not conclusions or claims.
>
> This project is archived for transparency and exploration only.
>
> Portions of this repository include AI-assisted code generation.
> All content was reviewed and curated by the author.

---

## Overview

This project visualizes an object constructed by mapping natural numbers to 2D
coordinates using a simple 2-adic decomposition.
The construction uses only integer and bitwise operations.

Every positive integer has a unique decomposition:

```
n = 2^v2(n) × core(n)
```

Where:
- `v2(n)` = number of trailing zeros (defines the shell)
- `core(n)` = odd component (defines the ray)

The mapping satisfies a basic scaling relationship:

```
C(2n) = 2 × C(n)
```

This visualization does not assert any deeper mathematical meaning; it is simply
a structural pattern exhibited by the construction.

---

## Two Equivalent Views

The same structure can be visualized from two perspectives:

| Square-Perimeter View | Ray-Structure View |
|-----------------------|--------------------|
| Shells as concentric squares | Shells as vertical lines |
| Rays emanate from the center | Rays converge toward 1 |
| Angular structure visible | Slope structure visible |

Both views arise naturally from the integer decomposition and preserve
the bijection among all represented integers.

### Square-Perimeter View

Integers 1–256 on concentric square shells with rays highlighted:

![Square perimeter with rays](pngs/map_256_labeled_with_rays.png)

Larger-scale odd-core rays:

![Rays to 4096](pngs/prime_rays_4096.png)

![Rays to 32768](pngs/prime_rays_32768.png)

![Rays to 262144](pngs/prime_rays_262144.png)

### Ray-Structure View

Same structure, different perspective — shells as vertical lines, rays converging to point 1:

![Ray structure horizontal](pngs/ray_structure_512_horizontal.png)

---

## Algorithm 1: Square Perimeter

For integer `n` in shell `k` (where `2^k ≤ n < 2^(k+1)`):

1. Shell radius: `R = 2^k`
2. Parameter: `t = (n - 2^k) / 2^k`
3. Point traced clockwise along the perimeter of the square

A bit-reversal operation provides an integer "angle" (`theta_key`):

```
theta_key(a) = bit_reverse(a, k bits)
```

Top 2 bits determine edge: `00`=TOP, `01`=RIGHT, `10`=BOTTOM, `11`=LEFT

---

## Algorithm 2: Ray Structure

Let `X` denote a horizontal unit and `Y` the vertical midpoint.

- Shell 1 starts at `(X, Y)`
- Higher shells place midpoints closer to 1
- Odd cores define linear slopes

Example slopes:

| Core | Slope |
|------|-------|
| 1 | -1 |
| 3 | +1 |
| 5 | 0 |
| 7 | +4/3 |

For any `n` with odd core `c`:

```
x(n) = X × (2 - 2^(1-k))
y(n) = Y + m_c × x(n)
```

Coordinates involve rational arithmetic only.

---

## Key Properties (Descriptive Only)

- One-to-one mapping among integers
- Scaling: `C(2n) = 2 × C(n)`
- Rational coordinates
- Nested shell structure

These properties simply describe how the visualization behaves.

---

## Python Tools

Main script: `demo.py`

```bash
# Square-perimeter visualization
python3 demo.py small-map --exp 8
python3 demo.py small-map --exp 8 --with-rays

# Odd-core rays
python3 demo.py odd-core-rays --exp 12

# Ray-structure visualization
python3 demo.py ray-structure --exp 9

# Coordinate analysis
python3 demo.py coord-analysis --exp 16
```

---

## Theta Toolkit (Experimental Toy Only)

An accompanying experimental mapping uses:

```
n = 2^v2(n) × core(n)
theta_key(n) = bit_reverse(core(n))
```

This code is included for completeness, not evaluation.
It is not benchmarked, not compared, and not intended for any system-level use.

### What It Is NOT

- **Not for production** — this is a hobby project
- **Not cryptographic** — deterministic, trivially reversible
- **Not a replacement for any existing library**
- **Not validated or tested for correctness**

### Files

| File | Description |
|------|-------------|
| `demo.py` | demo.py used to generate all visuals |


---

## Integer-Native Computation

Core operations:

| Operation | Description |
|-----------|-------------|
| `shell(n)` | `bit_length(n) - 1` |
| `core(n)` | `n >> ctz(n)` |
| `v2(n)` | count trailing zeros |
| `theta_key(a)` | bit reversal |

Floating-point is used only for rendering.

---

## Outputs

- SVG visualizations
- Optional CSV exports
- Coordinate and shell analysis

---

## Archive / Historical Notes

This repository includes a `failed_experiments/` directory containing earlier
exploratory code. These files are kept only for transparency and historical
context and are not intended for use.

The current visualization and toy mapping code resides in the top-level
`demo.py` and `cuda/` directory.

See [`cuda/README.md`](cuda/README.md) for file descriptions.

---

## Future Exploration (Non-committal)

This project does not assess or claim any application or practical relevance.
Readers are free to explore theoretical questions inspired by these structures
for their own curiosity.
No conclusions or implications are made here.

---

## License

Apache-2.0

## Author

Nenad Micic (nenad@micic.be)
Brussels, Belgium — November 2025
