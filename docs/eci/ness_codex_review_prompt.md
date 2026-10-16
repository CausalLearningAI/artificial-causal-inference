# Prompt for Codex: adversarial review of the NESS / NDES design and theory document (draft 1)

You are an adversarial referee for a mathematical design document. Your job is to find errors, gaps and overclaims, not to praise. Read these files in this repository in full before you write anything:

1. `docs/eci/ness_theory.md`: **the document under review**.
2. `docs/eci/dines_theory.md` (commit 7dd7c26), cited in the document as **[D]**. NDES reuses its separator lemma (Lemma 3), its alignment definitions (assumptions 2a screening, 2b purity, 3 coverage), Lemma 4, Theorem 1, Propositions 1, 3, 4 and 7, and its §4 critique of NES. Treat [D] as background, but flag any place where the new document uses a [D] result outside that result's hypotheses.
3. `scripts/benchmark/celeba_dines/RESULTS.md`, `results/eci/ness_splitting/summary.md` and `results/eci/ness_splitting/splitting_diagnostic.csv`: measured numbers the document quotes.
4. `src/eci/dines.py`: the code the algorithm section says it reuses.
5. NEXIS (arXiv 2606.17010; LaTeX source in `/nfs/scistore19/locatgrp/rcadei/NEXIS/paper/main.tex`, Algorithm 1 and Theorem 4.1): the forward step + terminal certification and the fixed-null precision argument that NESS adapts via sample splitting.

## Context you need

- **Setting.** A randomized binary treatment $T$ with $\mathbb{P}(T=1)=e$. Post-treatment images are summarized by a fixed dictionary of $m$ atoms $Z$ (TopK sparse-autoencoder codes, $m=9216$, about 100 after an activity prefilter). The running example is a frog: the treatment paralyses a leg ($P$) and changes skin colour ($C$); speed ($V$) drops because of $P$; size ($W$) co-determines paralysis.
- **NESS($\varepsilon$)** (primary target). The family of inclusion-minimal sets $S$ with residual information $G(S)=I(T;Z_{-S}\mid Z_S)\le\varepsilon$ nats. $G(S)$ equals the log-loss gain of the ideal full-$Z$ treatment classifier over the ideal $Z_S$ classifier. $\varepsilon$ is part of the target. Atoms inside $S$ are tagged *effect* (law shifts with $T$) or *context* (law does not shift).
- **Certification.** Selection (a forward step by cross-fitted information gain, then a sequential backward step) runs on split $D_1$. Two certificates run on split $D_2$:
  - *sufficiency*: an upper confidence bound on the cross-fitted log-loss gap of a nested reference classifier over the $Z_S$ classifier is $\le\varepsilon$; the guarantee is relative to the classifier family, "$\mathcal{F}$-sufficiency";
  - *necessity*: a PCM test of $T\perp\!\!\!\perp Z_j\mid Z_{S\setminus j}$ at Bonferroni level.
- **Exclusion analysis** (replaces "peeling"). For each $j\in S$, delete $j$ from the dictionary, keep $S\setminus\{j\}$, re-run the selection, and read the entering atoms $R_j$ as *masked* (if $j$ nearly reveals $T$) or *carried* (proxies or consequences). The population definition is the family of minimal completions. It is unique at $\varepsilon=0$ under the intersection property, where it equals $\mathrm{MB}(T;Z_{-j})\setminus S$. The masked/carried reading rests on a strong-data-processing bound.
- **NDES** (optional second level). The DINES all-subsets adjacency check run inside $S$, labelling atoms *direct* / *context* / *undetermined*. Under [D]'s assumptions, exact NESS = atoms of children $\cup$ co-parents, and the check returns the children. There is also an $\varepsilon/\delta$ window version.
- **Decisions already agreed with the author** (critique them only if you find a concrete problem): $\varepsilon$ is part of the target and in nats; binary $T$, so everything is a regression of $T$; single atoms are the default unit; effect/context tags; the exclusion analysis replaces peeling; only two theoretical levels (NESS and NDES).

## What to check, in priority order

1. **R2: certification (§3, Theorems 3–4, Lemma 5).**
   - Is "P(certify an $S$ that is not $\mathcal{F}$-$\varepsilon$-sufficient) $\le\alpha$" correctly stated and proved? Check the conditioning on $(D_1,D_{2a})$, the bounded-loss empirical-Bernstein version, and the Berry–Esseen step for the studentized mean. Lemma 5 claims variance $\ge\eta'\varepsilon^2$; check its proof, including the case where the true propensity reaches 0 or 1 (markers).
   - Check the population link $\Delta_n=G(S)+\mathrm{Exc}_S-\mathrm{Exc}_{\rm full}$, and its consequence that a poor reference classifier biases toward certification. Is the "nested reference classifier + power witness" mitigation sound, or can a nested booster with early stopping still certify falsely in a systematic way?
   - Does cross-fitting (the practical default) break the exact conditional argument? Is the appeal to Williamson et al. (2023) correct for a null at $\varepsilon>0$?
   - Necessity: is the PCM with response $T$ really a full conditional-independence test here (Lemma 1)? Are its rate conditions plausible for SAE codes with point masses at zero? Is the finite-family argument of Theorem 4 correct when $\hat S$ is random and $|\hat S|$ varies?
   - Is it right that 0-necessity (the PCM) is strictly weaker than $\varepsilon$-necessity? Does the document draw the right consequence?
   - §3.4: is "$U$ is valid simultaneously for every $\varepsilon$" correct? Is the declared-attempts Bonferroni rule for re-selection valid?
2. **R1: causal content (§2, Theorems 1–2).** The author first proposed $|\mathrm{ATE}_g-\mathrm{ATE}_{h_g}|\le\|g\|_\infty(\sqrt{2\varepsilon/e}+\sqrt{2\varepsilon/(1-e)})$. The document keeps it as valid and replaces it with $\mathrm{osc}(g)\sqrt{\varepsilon/(2e(1-e))}$.
   - Verify both bounds, and the claim that the new constant is attained (Appendix B.1 reports a worst ratio of 1.000 numerically). Construct a law that attains it, or show it cannot be attained.
   - Check the identity $eK_1+(1-e)K_0=G(S)$ with $K_t=\mathrm{KL}(P_t\|P(Z_S\mid t)\otimes K)$.
   - Check the "what it does NOT mean" list for errors and omissions.
3. **R3: consistency (§4, Proposition 2, Theorem 5, Proposition 3).**
   - Is it true that the population forward–backward algorithm always returns an element of NESS($\varepsilon$) without any composition assumption?
   - Check the margin condition (M), the greedy-completeness condition (GC), and the proof that the sample path is a $\gamma'$-greedy population path. Is a uniform margin over all sets of size $\le K_{\max}+1$ needed, or too strong?
   - Check the submodularity-ratio bound and its cap formula.
4. **The target (§1, Lemmas 1–4, Proposition 1).**
   - Monotonicity of $G$; minimality via single removals.
   - The intersection-property proof of uniqueness.
   - $u_j>\varepsilon$ ⇒ $j$ in every $\varepsilon$-sufficient set.
   - **TopK SAEs activate exactly $k$ atoms per input.** Does that break the intersection property, or make $u_j$ meaningless, in a way the document understates (O3)?
5. **Exclusion analysis (§6, Propositions 5–7, Lemma 8).**
   - Check $G(S\setminus j)=\iota_j+u_j$.
   - Check Proposition 6: the unique minimal completion $\mathrm{MB}(T;Z_{-j})\setminus S$, including that the intersection property passes to the marginal of $(T,Z_{-j})$.
   - Proposition 7 (masking bound): the strong data-processing step $\eta_{\rm KL}\le\eta_{\rm TV}$ with binary input, the conditional version with an ess-sup over $Z_C$, and the formula $\eta_j=\mathbb{E}|\pi_j-e|/(2e(1-e))$. Look for a counterexample where a separate effect is masked by more than a fraction $\eta_j$.
   - Lemma 8: faithfulness ⇒ within-arm dependence. Check the d-separation sub-path argument.
   - Is the masked/carried/mixed table justified row by row?
6. **NDES (§7, Theorems 6–7, Corollary 1).**
   - Does NESS(0) $=B_Z$ follow from [D, Prop. 1] plus (F1)? Is (F1) the right extra assumption, and when does it fail?
   - Do the co-parent separators $J_{D_s}$ really lie inside $S\setminus\{j_s\}$ for every co-parent, including co-parents that are descendants of $T$ (the temperature example)?
   - [D]'s Theorem 1 was stated with a mean null and "no cancellation" in mean form. Is the switch to a conditional-independence null legitimate everywhere it is used?
   - Theorem 7's window (W1–W4): is anything missing? Is "what is missing" (joint leak, proxies lowering $u_j$, markers) complete?
   - The finite-sample NDES precision claim on $D_2$.
7. **ECI paradox (§5, Proposition 4).** Check the replicate-atom example: sufficiency of the mean; $G(S)$ depending only on $|S|$; the claimed sizes (B.4). Check the "implicit $\varepsilon_n\approx z^2/(2n)$" heuristic, and whether the atom-408 paragraph overstates what is measured. Note that the 0.19 partial correlation is quoted from a benchmark diagnostic and is not in RESULTS.md.
8. **Feature splitting (§1.6, §6.5).**
   - The document claims complementary split pieces are *synergistic*: each piece alone carries about 1/3 of the concept's information, but 2/3 given the other (B.5). Verify this.
   - It concludes that the exclusion analysis does *not* reliably detect complementary splits, that swap sets do, and that sequential (not simultaneous) backward elimination is required. Check all three.
   - Check the quoted CelebA splitting numbers against `results/eci/ness_splitting/`.
9. **Counterexamples (§8).** For each of the frog cases (context, descendant co-cause, marker, XOR, duplicates, complementary split, coarse mediator, hidden physiology, collider bias), check the stated NESS set, tags, exclusion reading and NDES label, and whether "identifiable" vs "preferred" is stated correctly. Check the numbers in B.3.
10. **Algorithm and plan (§§9, 11).**
    - Does the pseudocode implement what the theorems assume: sequential backward step; selection only on $D_1$; every $D_2$ test at a hypothesis fixed by $D_1$?
    - Is anything in steps 6–9 secretly selecting on $D_2$ and then claiming a guarantee?
    - Is the reuse list against `src/eci/dines.py` accurate (function names exist and do what is claimed)?
    - Is the cost estimate plausible?
    - Are the CelebA predictions consistent with the theory? Are the quoted measured numbers correct against RESULTS.md?
11. **NES reassessment (§10).** Is each row factual and fair? It is the author's own paper. Flag anything overstated.

## Counterexamples to hunt (try hard)

- A law where the population forward step with $K_{\max}=m$ ends at a set that the sequential backward step cannot reduce to a minimal set. (Proposition 2 claims this is impossible.)
- Two elements of NESS(0) under the intersection property. (Proposition 1 claims there are none.)
- A reference classifier family for which the nested construction *systematically* certifies a set with $G(S)\gg\varepsilon$ at large $n$, beyond the XOR case the document already discusses.
- A separate effect $k$ ($Z_k\perp\!\!\!\perp Z_j\mid T,Z_C$) whose information drops by more than a fraction $\eta_j$ when $j$ is added.
- A co-parent $s$ whose separator $J_{D_s}$ is *not* inside $B_Z\setminus\{j_s\}$, or is not an overlap set while $B_Z$ is.
- A case where NDES on a superset $S\supsetneq B_Z$ returns something other than $J_C$ (Corollary 1 claims it cannot).
- A case where the exclusion map's "carried" reading is wrong without any within-arm dependence.
- A non-binary-$T$ or non-randomized step hidden in a proof that claims binary $T$ and randomization only where stated.
- A case where TopK sparsity makes the exact target non-unique, or makes $u_j=0$ for an atom that clearly carries treatment information.

## How to report

Return a numbered list of findings, ordered by severity. For each finding give:
- **Severity**: `critical` (a stated result is false, or its proof has a gap that is not easily fixed), `major` (a step is wrong or an assumption is missing or misstated, but the result is likely fixable), `minor`, or `nit`.
- **Location**: section and result number (e.g. "§3.2, Theorem 3(b)").
- **Finding**: what is wrong, stated precisely.
- **Evidence**: an explicit counterexample (distribution, DAG, numbers), the exact faulty step, or the file and line that contradicts a quoted number. If you run a numerical check, keep it under one minute of CPU and report the script and its output.
- **Suggested fix**, if you have one.

Then give a **"checked and sound"** list: every result, proof and quoted number you checked and found correct, so that we know what was covered. Do not rewrite the document, and do not edit any file.
