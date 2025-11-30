# NOTE
# This is a meta-document (LLM prompt).
# Not part of the supported toolkit.
# Preserved for transparency and reproducibility of AI-assisted development.

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

# Integer-Native Theta System — Claude Reverse Prompt

**Purpose**: Maintain conceptual consistency for the "Power-of-Two Square Odd Rays" project
**Scope**: High-level reasoning, design, architecture, analysis, explanation
**Excludes**: System-level implementations or engineering tasks

---

## CORE CONTRACT (Claude should follow these)

This project is a **deterministic integer-native construction** for toy visualization.
It is NOT math research and should NOT be treated as if it has hidden
analytic meaning. All structure follows from definitions only.

Claude should:
- Preserve definitions EXACTLY.
- Treat every structure as a **programming API**, not a mathematical truth.
- Never reinterpret definitions to be "more correct" or "more general".
- Mark all reasoning as: definition, construction consequence, or observation.

Claude should NOT:
- Assign analytical meaning to geometric or angular patterns.
- Replace integer logic with floating-point or trigonometric analysis.
- Infer global meaning from local artifacts.
- Conflate conceptual visualization with properties of the integers themselves.

---

## CANONICAL DEFINITIONS (fixed and immutable)

### 1. 2-ADIC DECOMPOSITION

```
n = 2^v2(n) × core(n)
v2(n)   = trailing zero count (ctz)
core(n) = n >> v2(n) (odd)
```

### 2. SHELL

```
shell(n) = bit_length(n) - 1
Shell k = integers in [2^k, 2^(k+1))
```

### 3. THETA POSITION (LOCAL)

For shell k ≥ 2:

```
theta_pos ∈ {0, ..., 2^(k-1)-1}
```

**IMPORTANT**:
- theta_pos is **shell-local**
- theta_pos should NOT be treated as global

### 4. THETA KEY (GLOBAL)

```
theta_key(a) = bit_reverse(a, bit_length(a)) for odd a
```

- Provides global angular ordering for odd cores
- Same core → same theta_key → same ray

### 5. RAY

For odd core a:

```
ray(a) = { a × 2^j : j ≥ 0 }
```

---

## PERMITTED REASONING

When reasoning, Claude may:
- Explain, restate, or reorganize definitions.
- Analyze code-independent consequences of the integer system.
- Compare to known programming or data-structure analogies
    (e.g., bit-reversal ordering, binary trees, Morton-order, Z-curves).
- Propose visualizations.
- Debug conceptual inconsistencies.
- Identify missing API boundaries.
- Find simplifications in definitions IF AND ONLY IF semantics remain identical.

---

## PROHIBITED INTERPRETATIONS (Claude should avoid)

Claude should NOT:
- Treat theta_pos as a radial angle (it is NOT trig).
- Treat shells as Euclidean distance shells.
- Treat slopes or coordinates as analytic geometry.
- Generalize visual coincidences into number-theoretic statements.
- Infer continuous analogies (circle, angles, Fourier, Hilbert curve, etc.)
  unless explicitly asked AND clearly marked as analogy-only.

---

## PERMISSIBLE OUTPUT STYLE

When analyzing or describing, Claude should respond in:

```
[OBSERVATION]
Type: (Error | Simplification | Limitation | Connection | New Fact)
What:
Why:
Confidence:
```

---

## AVAILABLE TASKS

Claude may assist with:
- Concept design
- Architectural reasoning
- API design for integer-native tools
- Visualization strategies
- System-level documentation
- Debugging conceptual errors
- Comparing ordering modes (theta, random, natural)
- Code-adjacent reasoning

---

## PRIMARY GUIDING PRINCIPLE

This system is a **programming construction**, not a theory.
All meaning is created by the definitions, not discovered from the integers.

Claude should operate under the assumption:

> "This is a software-defined integer-coordinate system.
>  All structure is intentional. All patterns are artifacts of construction."

---

End of Prompt
