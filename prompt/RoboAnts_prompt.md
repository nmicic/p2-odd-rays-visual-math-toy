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
---

"""
MOTE: To be used as prompt with AI model when working on new program

ROBO-ANT WORLD META-SPEC
========================

Perspective
-----------
We work in the "robo-ant" model:

  * The world is a discrete grid (2D square by default).
  * Every state is labeled by a positive integer n.
  * Ants are born at a common origin ("home") and can only move
    one unit in the four cardinal directions: UP, LEFT, DOWN, RIGHT.
  * Ants do NOT know trigonometry, calculus, or floating-point geometry.
    They only know integers, bitwise operations, and simple comparisons.

Power-of-Two Square Shells
--------------------------
For a fixed exponent `exp`, we consider integers n in [1, 2**exp).

Each n is decomposed into its 2-adic form:
    n = core * 2**v2

  * v2 = v2(n) is the 2-adic valuation: the largest k such that 2**k divides n.
    This is the "shell index": which power-of-two square the ant stands on.
  * core is the odd factor of n (the "odd core").
    This labels the "ray": all numbers with the same odd core lie on the
    same straight-looking grid line across shells.

We define a bijection:
    core  <->  key in [0, 2**(exp-1) - 1]

implemented as:
    odd_core_to_key(core, exp)
    key_to_odd_core(key, exp)

The `key` is a combinatorial perimeter coordinate on the square:
stepping key -> key+1 means walking one step clockwise around the boundary.

The mapping is constructed so that the geometric scaling rule holds:
    C(2*n) = 2 * C(n)
where C(n) is the (implicit) grid embedding. We do NOT need to know C(n)
explicitly; we only rely on shells (v2) and keys.

Algorithmic Rules / Constraints
-------------------------------
All *core* algorithms in this project must obey:

  * NO trigonometry or floating-point math in the core logic.
    - No sin, cos, tan, atan2, floats for placement, ordering, or neighbor search.
    - Trig is allowed ONLY in optional verification / visualization helpers,
      and those MUST be clearly separated (e.g. compute_angle_deg).

  * Shells and rays are defined purely from the integer n via:
        v2(n)      = index of the shell (square level)
        core(n)    = odd core (ray label)
        key(core)  = angular ordering along the square perimeter

  * Any "theta order" or "angular walk" is implemented by:
        - converting n -> (core, v2) -> key
        - stepping key (mod total_keys)
        - reconstructing cores via key_to_odd_core
        - combining with shells via core << shell

  * Operations must be O(1) per step in terms of exp:
        - neighbor lookup along theta
        - shell walks
        - ray walks (fixed core, varying shell)
        - prime-only walks (filtering with is_prime(n))

  * Memory usage should be streaming-friendly:
        - no huge precomputed angle tables
        - any lookup tables must be small and explicable
        - the code should conceptually fit into an ant's tiny memory

Interpretation and Disclaimers
------------------------------
  * Rays, "prime rays", gaps, deserts, and other patterns are treated as
    visual artifacts of this particular embedding, nothing more.
  * We DO NOT claim any new theorems or deep structure about primes or
    number theory.
  * This code is exploratory and educational. All patterns are to be
    interpreted cautiously, if at all.
  
  * EVALUATION STYLE: When assessing results, significance, or novelty:
    - No fanfare, no hype, no grandiose claims
    - Realistic, grounded evaluation only
    - Low context preferred - state facts plainly
    - No sugar-coating, but no unnecessary negativity either
    - If something is just a tautology of the construction, say so
    - If something resembles known structures, name them
    - "Interesting to look at" is NOT "mathematically significant"
    - Safe, non-speculative and no emoji

3D Extension (Optional)
-----------------------
A 3D version can exist with a cube and six directions:
    UP, DOWN, LEFT, RIGHT, FORWARD, BACKWARD
The same philosophical rules apply:
  * integers -> shells (via v2) and discrete "face/perimeter" coordinates
  * only integer / bitwise logic in the core
  * any trigonometry relegated to human-facing visualization helpers.

When writing new functions in this file:
  - Think like a robo-ant: "Can I do this with bit operations and integers only?"
  - Preserve C(2*n) = 2*C(n) whenever you extend / modify the mapping.
  - Keep anything geometric (angles, plots) in a thin wrapper layer.

Framework for Working with AI on Code
-----------------------
https://dev.to/nmicic/compass-steering-wheel-destination-framework-for-working-with-ai-on-code-331
"""
