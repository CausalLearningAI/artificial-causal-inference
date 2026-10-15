# Prompt for Codex: adversarial review of the DINES theory report (draft 2)

You are reviewing a mathematical theory report as an adversarial referee. Your job is to find errors, not to praise. Read the whole report first:

`docs/eci/dines_theory.md` (in this repository). Draft 1 is commit 07f7c11, and Appendix A maps its result numbers to the new ones.

## Context you need

- **DINES** (Direct–Indirect Neural Effect Search) is a proposed successor to **NES** (Neural Effect Search; Mencattini, Cadei, Locatello, arXiv 2510.14073). The setting is a randomized binary treatment $T$, post-treatment images $X$, and a fixed dictionary of $m$ atoms $Z=\psi(X)$ (sparse-autoencoder codes).
- **Objective:** the atoms the treatment changes, excluding those that change only as a consequence of other changed atoms. Running example: a frog whose leg is paralysed ($P$) and whose colour changes ($C$); speed ($V$) drops only because of $P$. The target is $\{P,C\}$.
- **Target (Definition 1):** $\mathrm{Adj}_Z(T)$ = atoms $j$ such that the mean null $H_0(j\mid A):\mathbb{E}[Z^j\mid T,Z^A]=\mathbb{E}[Z^j\mid Z^A]$ is false for every conditioning set $A$ **with overlap** ($0<\pi_A<1$ a.s., $\pi_A=\mathbb{P}(T=1\mid Z^A)$). Draft 1 used every subset; the overlap restriction is new.
- **Algorithm:** a forward IAMB-style phase, then a terminal backward rule that keeps an atom only if it is significant at level $\alpha/m$ given every non-skipped subset of the forward output. It has two options:
  - a margin $\delta$ on the partial-correlation scale $\rho_{j|A}=\mathbb{E}[(T-\pi_A)(Z^j-\mu_{j,A})]/\sqrt{\mathbb{E}(T-\pi_A)^2\,\mathbb{E}(Z^j-\mu_{j,A})^2}$, with null $|\rho|\le\delta$;
  - an overlap rule: skip sets where more than a fraction $f$ of the units have $\hat\pi\notin[\varepsilon,1-\varepsilon]$. Treatment-revealing atoms are set aside in the forward phase, and an atom is reported as "undetermined" when its only non-rejections are at skipped sets.
- **Selection guarantee:** adapted from **NEXIS** (Cadei et al., arXiv 2606.17010; source in `/nfs/scistore19/locatgrp/rcadei/NEXIS/paper/`). Precision bounds each false atom by its test at one *fixed* true null inside the forward output, which avoids post-selection inference. The dictionary here is post-treatment, which creates explaining-away (collider) dependence.
- **Assumptions (§0 and the table in §2.6):** randomization; alignment in NEXIS's screening form plus purity on the treatment side; no hidden common cause (coverage of $\mathrm{MB}(T)$); no exact cancellation; valid and consistent tests. For arbitrary dictionaries without alignment, "assumption L" (local separation: a fixed separator pool plus forward completeness) is kept explicitly.

## What to check (in priority order)

1. **Theorem 3 (realistic alignment, margin $\delta$) and Theorem 2 (generic guarantee).** Is "leak $\le\delta<$ direct effect" at fixed separators, plus forward completeness at margin $\delta$, really enough for $\liminf\mathbb{P}(\hat R_n=J_C)\ge1-\alpha$, with probability $\to1$ for strict leaks? Specific points:
   - Check the boundary validity of the margin test $p=\Phi((\delta-|\hat\rho|)/\widehat{se})+\Phi((-\delta-|\hat\rho|)/\widehat{se})$, and the claim that the GCM numerator's bias is a product of nuisance errors under alternatives.
   - Hunt for precision counterexamples involving the set-aside list $K$, the skipped sets, and the event (O) that the estimated overlap classification is correct.
   - Is (T3) with $K$ stated in the form the proof needs?
2. **Lemma 3 (separator lemma, ancestral form).** (a) $T\perp_d Y\setminus D\mid D$ with $D=\mathrm{MB}(T)\cap\mathrm{An}(Y)$; (b) $T\perp_d s\mid(\mathrm{MB}(T)\cap\mathrm{An}(s))\setminus\{s\}$ for a non-child $s$. Both are for a root $T$, via moralization. Try hard to break them: other causes that descend from $T$, other causes that are ancestors of other children, sets $Y$ that contain an other cause and a descendant of its child, and non-root $T$.
3. **Alignment in screening form.** Lemma 4 (contraction from factors to atoms; CI and mean versions; cases (a), full blanket with no purity, and (b), with purity) and Lemma 5 (mean screening ⟺ the principal atoms determine $\mathbb{E}[Z^j\mid L_D]$). Check:
   - Is the joint screening condition 2a correctly derived from NEXIS's Eq. 6 plus its composition clause?
   - Is purity (2b) really needed only where the report says? Example (d) is the claimed necessity witness.
   - Shared measurement noise across atoms: does any step silently assume independent noise?
4. **Theorem 1 (identification on overlap sets).** Does $\mathrm{Adj}=J_C$ follow when $B_Z\in\mathcal{O}$? Is it true that the separators in items 2–4 are fixed and lie in $\mathcal{O}$ whenever the report says so? In the frog with colour revealing $T$: does $P$ stay, is $V$ excluded via $\{P\}$, and are atoms changed only through colour kept?
5. **Propositions 3 and 4 (non-identification).** Proposition 3: does the quantile construction give a valid model with independent noises in which $V$ is a child of $T$? Proposition 4: is the claimed symmetry between $P$ and colour-only atoms correct (the substitution in both directions, acyclicity, $T$ a root)? Note that the report flags $P$ as "undetermined" too in the exact-copy case.
6. **Propositions 5–6 (leak size).** Check the exact identity $\rho_{j|A}=\rho_{g|A}\sqrt{\mathbb{E}\mathrm{Var}(g\mid Z^A)/\mathbb{E}\mathrm{Var}(Z^j\mid Z^A)}$ and its condition. Check the linear-Gaussian formula $\theta=\gamma^\top\Omega(\Sigma+\Omega)^{-1}\tau_D$, the diagonal form $\sum_k\gamma_k\tau_k(1-R_k^2)$, the operator-norm bound, and the $\rho$ conversion. Is Conjecture 1 correctly labelled?
7. **Remaining proofs.** Lemma 1 (including item 5), Lemma 2 (folding), Propositions 1, 2, 7 (feature splitting: the clean split keeps both atoms, the redundant split loses all three), 8, 9 and 10 (covariates; a measured common cause needs no atom). Check each step. Are the conditional-independence rules (weak union, contraction, decomposition) used correctly?
8. **Mediation framing and the NEXIS comparison (§0, §2.2).** Is it fair to say that the direct/indirect split needs untestable assumptions (a) and (b) for *any* method? Is the description of NEXIS ($W^{\mathrm{dir}}$ definition, Principal Alignment Eq. 6, the 94–96% and $p<10^{-14}$ evidence) accurate against the NEXIS source?
9. **NES critique (§4.1).** For each numbered hidden assumption, check the equation reference (Eqs. 24, 26, 27–28, 29–30) against the NES paper if available, and the code line references against `/nfs/scistore19/locatgrp/tmencatt/projects/neuralscience/src/neuralscience/nes.py`. Check the counterexample $v_1=(3,0,-1)$, $v_2=(0,1.5,2)$, $\tau^Y=(1,1)$, and the claim that the Eq. 29 term is identically zero in a randomized experiment. Flag anything unfair or overstated.
10. **Labels and numbers.** Is anything labelled Theorem/Proposition/Lemma without a complete proof? Is any measured number (Appendix B, including the population computations in B.4 and the δ window $[0.015,0.090)$) used to support a claim it cannot support?

## How to report

Return a numbered list of findings. For each finding give:
- **Severity**: `critical` (a stated result is false or its proof has a gap that is not easily fixed), `major` (a proof step is wrong or an assumption is missing or misstated, but the result is likely fixable), `minor`, or `nit`.
- **Location**: section and result number.
- **Finding**: what is wrong, stated precisely.
- **Evidence**: a counterexample (explicit DAG, distribution, numbers) or the exact faulty step.
- **Suggested fix**, if you have one.

Order findings by severity. End with a brief "checked and found sound" list, so we know what was covered. Do not rewrite the report.
