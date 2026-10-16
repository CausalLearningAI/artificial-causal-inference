# NESS / NDES — Neural Effect Sufficiency Search and Neural Direct Effect Search: design and theory

*Draft 1, 2026-10-16 (R. Cadei). Successor of NES (Neural Effect Search, "Exploratory Causal Inference", arXiv 2510.14073) and of the DINES draft (`docs/eci/dines_theory.md`, commit 7dd7c26, cited below as **[D]**, e.g. [D, Lemma 3]). Adversarial review prompt: `docs/eci/ness_codex_review_prompt.md`.*

**Labels.** *Theorem*, *Proposition*, *Lemma* and *Corollary* have a complete proof here, or a one-line pointer to a proof in [D]. *Conjecture* = believed, not proved. *Open* = unknown. *Heuristic* = an approximation used for a recommendation, not a result. Numbers marked "measured" come from Appendix B (closed-form computations, each under one minute of CPU) or from `scripts/benchmark/celeba_dines/RESULTS.md`.

---

## 0. Top page (readable alone)

**Objective.** A randomized binary treatment $T$ changes what we see in images or videos, and a fixed dictionary of $m$ atoms $Z$ (for example sparse-autoencoder codes) summarizes each image. We want a small set of atoms that carries all but $\varepsilon$ nats of the information $Z$ holds about $T$, and, as an optional second step, the atoms in that set that the treatment changes directly.

**The two levels.**

| | Target | Assumptions | Output |
|---|---|---|---|
| **NESS($\varepsilon$)** (Neural Effect Sufficiency Search) | an inclusion-minimal set $S\subseteq[m]$ with $I(T;Z_{-S}\mid Z_S)\le\varepsilon$, i.e. a $Z_S$-only treatment classifier loses at most $\varepsilon$ nats of log-loss against the best classifier that sees all of $Z$ | i.i.d. units, dictionary fixed before the analysis, sample splitting. Randomization is needed only for the causal reading (§2). **No alignment, no causal graph.** | $S$; per atom an *effect* / *context* tag; the certified residual information; interchangeable ("swap") atoms; the exclusion map $j\mapsto R_j$ |
| **NDES** (Neural Direct Effect Search; optional) | the atoms of $T$'s children, found inside $S$ | DINES's: latent causal DAG with $T$ a root; alignment on the Markov blanket of $T$ (screening + purity); no hidden co-parent; overlap; no cancellation | each atom of $S$ tagged *direct* / *context* / *undetermined* |

**Guarantees (one sentence each).**
- **NESS.** With probability at least $1-\alpha$, a certified set leaves at most $\varepsilon$ nats of treatment information that the declared classifier family can find in $Z$, and every atom certified *necessary* really carries information given the other selected atoms. Under exact sufficiency the treatment's effect on *any* feature of $Z$ equals its effect on a function of $Z_S$, and under $\varepsilon$-sufficiency the gap is at most $\mathrm{osc}(g)\sqrt{\varepsilon/(2e(1-e))}$, where $e=\mathbb{P}(T=1)$ (Theorems 1–3).
- **NDES.** Under DINES's assumptions, exact NESS equals the atoms of the treatment's children and their co-parents, and the in-set adjacency check returns exactly the children, with false "direct" labels at most $\alpha$ (Theorems 6–7).

**Exclusion analysis** (replaces peeling; informational, no causal assumption).
1. For each selected atom $j$, delete $j$ from the dictionary, keep $S\setminus\{j\}$, and re-run the selection. The atoms $R_j$ that enter are what $j$ stood in for.
2. If $j$ nearly reveals $T$, then $R_j$ may hold effects that $j$ **masked**: excluding colour brings paralysis back. If $j$ does not, every atom whose information $j$ removed is provably dependent on $j$ within each arm, so it is **carried** by $j$ (a proxy or consequence): excluding paralysis brings speed in (Proposition 7).
3. Exclusion never conditions on $j$, so it avoids the overlap failure that conditioning on a marker causes. It detects duplicated atoms. It does **not** reliably detect complementary splits; swap sets do (§6.5).

**Algorithm (six lines).**
1. Prefilter atoms using $Z$ only; split the units into $D_1$ (selection) and $D_2$ (certification).
2. On $D_1$, forward step: add the atom with the largest cross-fitted gain in treatment information (nats), until a held-out upper bound on the residual information is $\le\varepsilon$, or a cap is reached.
3. On $D_1$, backward step: remove atoms one at a time while the set stays $\varepsilon$-sufficient.
4. On $D_2$: certify sufficiency (upper confidence bound on the log-loss gap $\le\varepsilon$), and test the necessity of each atom with a conditional-independence test (PCM) at level $\alpha/|S|$.
5. Tag effect/context, compute swap sets and the exclusion map.
6. Optionally run NDES: the DINES all-subsets check inside $S$.

**Proved vs open.**
- *Proved:*
  - the binary-$T$ reduction and the log-loss identity;
  - minimality via single removals;
  - uniqueness of the exact target under the intersection property;
  - R1, with a constant tighter than first proposed (Theorems 1–2);
  - certification validity, relative to the classifier family (Theorems 3–4);
  - consistency of the selection under a margin and greedy-completeness condition, with a submodularity-ratio bound on the cap (Theorem 5, Proposition 3);
  - the ECI paradox as a fixed-target statement, with a worked example (Proposition 4);
  - the exclusion map, its exact form, and the masking bound (Propositions 5–7, Lemma 8);
  - NDES identification, exact and in an $\varepsilon/\delta$ window (Theorems 6–7).
- *Open:*
  - bounding the joint leak $G(B_Z)$ from per-atom alignment quality (Conjecture 1);
  - the effect of TopK sparsity on the intersection property (O3);
  - learner-free certification, which is impossible in general (O5);
  - group discovery from $Z$ (O6);
  - choosing $\varepsilon$ automatically (O7).

---

## 1. Setup and definitions

### 1.1 Notation

- **Units.** $n$ i.i.d. units (for video, i.i.d. clusters; §9.4). Binary treatment $T\in\{0,1\}$, $\mathbb{P}(T=1)=e\in(0,1)$.
- **Dictionary.** $Z=\psi(X)\in\mathbb{R}^m$, with $\psi$ fixed before the analysis (not fitted on the analysed units; [D, §3.6]). $[m]=\{1,\dots,m\}$, $Z_A=(Z_k)_{k\in A}$, $Z_{-A}=Z_{[m]\setminus A}$, $Z_\emptyset$ is a constant. (Notation differs from [D], which writes $Z^A$.)
- **Propensity given atoms:** $\pi_A(z_A):=\mathbb{P}(T=1\mid Z_A=z_A)$, $\pi:=\pi_{[m]}$, $\pi_\emptyset=e$.
- **Log-loss** of a probability $p$ for outcome $t$: $\ell(t,p)=-t\log p-(1-t)\log(1-p)$, in nats. Risk $\mathcal{R}(f)=\mathbb{E}\,\ell(T,f(Z))$.
- **Residual information** of $S$: $G(S):=I(T;Z_{-S}\mid Z_S)$.
- **Unique information** of atom $j$: $u_j:=I(T;Z_j\mid Z_{-j})=G([m]\setminus\{j\})$.
- **Overlap family** [D, §1]: $\mathcal{O}:=\{A:0<\pi_A<1\text{ a.s.}\}$; it is closed under subsets.
- $h(p)=-p\log p-(1-p)\log(1-p)$; $H(T)=h(e)\le\log2\approx0.693$ nats.

**Running example: the frog.** The treatment paralyses a hind leg ($P$) and changes skin colour ($C$). Speed ($V$) drops because of $P$. Size ($W$, pre-treatment, independent of $T$) co-determines how badly the leg is paralysed. Atoms $Z_P,Z_C,Z_V,Z_W$ read these factors. Variants add a temperature factor $\Theta$ (colour changes temperature, temperature affects paralysis), a lethargy factor, or a hidden physiology factor.

### 1.2 Two basic lemmas

**Lemma 1 (binary treatment: sufficiency is a regression of $T$).** For $A\subseteq B\subseteq[m]$, the following are equivalent: (i) $T\perp\!\!\!\perp Z_{B\setminus A}\mid Z_A$; (ii) $\pi_B=\pi_A$ a.s.; (iii) $I(T;Z_{B\setminus A}\mid Z_A)=0$. Moreover
$$I(T;Z_{B\setminus A}\mid Z_A)=\sum_{t}\mathbb{P}(T=t)\,\mathbb{E}\big[\mathrm{KL}\big(P(Z_{B\setminus A}\mid Z_A,T=t)\,\big\|\,P(Z_{B\setminus A}\mid Z_A)\big)\,\big|\,T=t\big],$$
so *any* difference between the arms in the conditional law of $Z_{B\setminus A}$ given $Z_A$ (mean, spread, firing rate, shape) makes it positive.

*Proof.* (i)⇒(ii) is immediate. (ii)⇒(i): given $Z_B$, $T$ is Bernoulli with parameter $\pi_B=\pi_A$, a function of $Z_A$, so $P(T\mid Z_B)=P(T\mid Z_A)$. (i)⇔(iii) is the zero property of conditional mutual information. The display is the standard symmetric form $I(T;Y\mid X)=\mathbb{E}\,\mathrm{KL}(P(Y\mid X,T)\|P(Y\mid X))$. $\square$

This supersedes the mean null of [D, §1]. For binary $T$, the conditional-mean null *for the response $T$* is the full conditional-independence null, so a test that regresses $T$ (the PCM of §3.3) sees every distributional change. A test that regresses the atom (the GCM in [D]) sees only mean shifts.

**Lemma 2 (log-loss identity, chain rule, monotonicity).**
1. $G(S)=\mathcal{R}(\pi_S)-\mathcal{R}(\pi)$: the residual information is the log-loss gain of the ideal classifier given all of $Z$ over the ideal classifier given $Z_S$.
2. For $A\subseteq B$: $G(A)=I(T;Z_{B\setminus A}\mid Z_A)+G(B)$. Hence $G$ is non-increasing, $G([m])=0$, $G(\emptyset)=I(T;Z)\le H(T)$, and $G(S)\le H(T\mid Z_S)$.
3. For any $f$ that is a function of $Z_A$: $\mathcal{R}(f)-\mathcal{R}(\pi_A)=\mathbb{E}\,\mathrm{KL}(\mathrm{Bern}(\pi_A)\|\mathrm{Bern}(f))\ge0$ (log-loss is a proper scoring rule).

*Proof.* (1) $\mathcal{R}(\pi_S)-\mathcal{R}(\pi)=\mathbb{E}\log\{P(T\mid Z)/P(T\mid Z_S)\}=I(T;Z_{-S}\mid Z_S)$. (2) Chain rule of mutual information; $G(S)=H(T\mid Z_S)-H(T\mid Z)$. (3) Condition on $Z_A$ and expand. $\square$

### 1.3 The target

**Definition 1 (NESS).** For $\varepsilon\ge0$, a set $S$ is **$\varepsilon$-sufficient** if $G(S)\le\varepsilon$. $\mathrm{NESS}(\varepsilon)$ is the family of inclusion-minimal $\varepsilon$-sufficient sets. It is non-empty, because $[m]$ is $0$-sufficient. **$\varepsilon$ is part of the target**: different $\varepsilon$ define different targets.

We do *not* target $\arg\min|S|$:
- it is combinatorially hard (best-subset selection);
- it is not unique;
- it is not what a forward–backward search can certify.

**Lemma 3 (minimality is checked one atom at a time).** An $\varepsilon$-sufficient $S$ is inclusion-minimal iff $G(S\setminus\{j\})>\varepsilon$ for every $j\in S$. The *necessity margin* of $j$ is $G(S\setminus\{j\})-G(S)=I(T;Z_j\mid Z_{S\setminus\{j\}})$, so in a minimal set every atom has $I(T;Z_j\mid Z_{S\setminus\{j\}})>\varepsilon-G(S)\ge0$.

*Proof.* Every proper subset of $S$ lies in some $S\setminus\{j\}$, and $G$ is non-increasing (Lemma 2). $\square$

**Lemma 4 (the core: atoms that every $\varepsilon$-sufficient set must contain).** If $j\notin S$, then $G(S)\ge u_j$. Hence every atom with $u_j>\varepsilon$ belongs to every $\varepsilon$-sufficient set, and in particular to every element of $\mathrm{NESS}(\varepsilon)$.

*Proof.* By the chain rule, $G(S)=I(T;Z_{-S}\mid Z_S)=I(T;Z_{-(S\cup\{j\})}\mid Z_S)+I(T;Z_j\mid Z_{-j})\ge u_j$. $\square$

$\{j:u_j>0\}$ is Kohavi and John's set of *strongly relevant* features. Lemma 4 is its $\varepsilon$ version.

**Proposition 1 (the exact target is unique under the intersection property).** Suppose the law of $(T,Z)$ has the intersection property ($X\perp\!\!\!\perp Y\mid(U,W)$ and $X\perp\!\!\!\perp W\mid(U,Y)$ imply $X\perp\!\!\!\perp(Y,W)\mid U$; for example, a strictly positive joint density). Then $\mathrm{NESS}(0)$ has exactly one element, the Markov boundary
$$\mathrm{MB}^\star=\{j:u_j>0\}.$$
Without the intersection property, $\mathrm{NESS}(0)$ can have several elements, but each contains $\{j:u_j>0\}$.

*Proof.* Let $S_1,S_2$ be $0$-sufficient. Write $A=S_1\cap S_2$, $B=S_1\setminus S_2$, $C=S_2\setminus S_1$, $D$ = the rest. Then $T\perp\!\!\!\perp(C,D)\mid(A,B)$ and $T\perp\!\!\!\perp(B,D)\mid(A,C)$.
- Weak union gives $T\perp\!\!\!\perp C\mid(A,B,D)$ and $T\perp\!\!\!\perp B\mid(A,C,D)$. Intersection gives $T\perp\!\!\!\perp(B,C)\mid(A,D)$.
- Weak union on the first statement gives $T\perp\!\!\!\perp D\mid(A,B,C)$. Intersection again gives $T\perp\!\!\!\perp(B,C,D)\mid A$.

So $S_1\cap S_2$ is $0$-sufficient, and two distinct minimal sets cannot exist. For the characterization:
- If $u_j>0$, $j$ is in every sufficient set (Lemma 4).
- If $j\in\mathrm{MB}^\star$ had $u_j=0$, intersection applied to $T\perp\!\!\!\perp Z_j\mid Z_{-j}$ and $T\perp\!\!\!\perp Z_{-\mathrm{MB}^\star}\mid Z_{\mathrm{MB}^\star}$ would make $\mathrm{MB}^\star\setminus\{j\}$ sufficient, which contradicts minimality.

The last sentence of the proposition is Lemma 4. $\square$

**Non-uniqueness for $\varepsilon>0$ is generic, not a pathology.**
- Let $Z_1,Z_2$ be conditionally independent given $T$ with equal information. For any $\varepsilon\in[G(\{1\}),I(T;Z_1,Z_2))$, both $\{1\}$ and $\{2\}$ are minimal.
- Minimal sets can also differ in size. In the replicate example of §5, every set of a fixed size $s(\varepsilon)$ is minimal.
- Exact duplicates break the intersection property, so they give non-unique sets even at $\varepsilon=0$.

**How to report interchangeable atoms.** For $j\in S$, the **swap set** is $\mathrm{Sw}_\varepsilon(j;S):=\{k\notin S:\ G((S\setminus\{j\})\cup\{k\})\le\varepsilon\}$: the atoms that could replace $j$ without losing $\varepsilon$-sufficiency. It is a population quantity, and it is certified on $D_2$ like $S$ itself (§3). Report each selected atom with its swap set, and treat atoms in the same swap set as one finding.

### 1.4 Tags inside $S$: effect and context

**Definition 2.** For $j\in S$:
- $j$ is an **effect** atom if its law shifts with $T$, i.e. $I(T;Z_j)>0$;
- $j$ is a **context** atom if $I(T;Z_j)=0$. Its law is the same in both arms, but it is needed to interpret the others.

*Frog.*
- $P$ and $C$ are effects.
- Size $W$ is context: the treatment does not change size, but a small frog's paralysis says less about the treatment than a large frog's. In the explicit table of [D, Prop. 1], $I(T;W\mid P)=0.029$ nats while $I(T;W)=0$ (measured, B.3).
- Context atoms are exactly the *modifiers* of the treatment's effect on the effect atoms, which links NESS to NEXIS's effect-modifier targets.

**Estimating the tags.**
- *Effect.* Because $T$ is randomized, a permutation test of $T\perp\!\!\!\perp Z_j$ is exact at any $n$. Use a statistic that sees any change, such as the held-out log-loss of a classifier of $T$ from $Z_j$ alone, or an energy distance.
- *Context.* This must be *shown*, not inferred from a non-significant test. Certify $I(T;Z_j)\le\varepsilon_c$ with an upper confidence bound (§3.2 machinery, with known baseline $h(e)$).
- *Weak.* Neither of the above. Report it as such.

**A co-cause that is also a descendant of $T$.** Temperature variant: $T\to C\to\Theta\to P\leftarrow T$.
- $\Theta$ is a co-parent of $P$ and a descendant of $T$, so its law shifts. NESS tags it **effect**, and that is correct: the treatment does change temperature, through colour.
- It is not a direct effect, and its role in $S$ is to interpret $P$. Only NDES can say this: its separator $\{Z_C\}$ lies inside $S$ ([D, Lemma 3(b)]; Theorem 6 below), so NDES tags it **context**.
- So the two levels answer different questions. NESS asks "does the atom's law move?" (total effect). NDES asks "is the treatment one of its direct causes?".

### 1.5 Units of $\varepsilon$

- **Absolute.** $\varepsilon$ is in nats and $\varepsilon\le H(T)\le0.693$.
- **Correlation equivalent (heuristic).** For small Gaussian-like effects, $I(T;Z_j\mid Z_A)\approx\rho^2_{j|A}/2$, with $\rho$ the partial correlation of [D, §1]. This holds because $\mathrm{KL}(\mathrm{Bern}(p')\|\mathrm{Bern}(p))\approx(p'-p)^2/(2p(1-p))$. Measured for a Gaussian shift: the ratio $I/(\rho^2/2)$ is 1.000, 1.005, 1.03 and 1.11 at $\rho=0.05$, $0.10$, $0.24$ and $0.45$ (B.2).
  - DINES's margin $\delta=0.05$ ≈ 0.00125 nats.
  - The CelebA atom 408's partial correlation 0.19 ≈ 0.018 nats.
- **Relative.** Also report $\kappa(S)=G(S)/I(T;Z)$, the fraction of treatment information left out. A fixed $\varepsilon$ in nats means different things at different effect sizes (§12, O7).

### 1.6 The discovery unit, and feature splitting

**Default unit: single atoms.**
- Every result in §§1–5 concerns index sets. It holds verbatim when the "atoms" are groups of atoms fixed in advance from $Z$ only (replace $j$ by a group $g$ and $Z_j$ by $Z_g$).
- So the NESS guarantee is valid under feature splitting. What splitting costs is interpretability, and, under $\varepsilon$, coverage.

**What the CelebA splitting diagnostic measured** (job 862381, `results/eci/ness_splitting/`; 9,216 TopK atoms, 19,962 test images):
- *No near-duplicates.* For 0 of 40 attributes does an atom have firing-set Jaccard $>0.5$ with the attribute's best atom.
- *Complementary coverage is common.* For 22 of 40 attributes, an OR of two atoms gains more than 0.05 F1 over the best single atom. The median gain is 0.068 and the maximum 0.230 (Wearing_Lipstick). The pairs rarely co-fire: median co-fire Jaccard 0.026, but Eyeglasses' pair 6051+2218 has 0.118.
- *Several atoms are usually needed.* For 34 of 40 attributes, three or more atoms are needed to reach 95% of the 5-atom held-out AUROC.
- *Single-atom attributes.* Only Wearing_Hat (atom 38; best pair gain 0.001) and Blond_Hair (5154) qualify. Eyeglasses gains +0.082 pair F1 (partner atom 2218); Smiling gains +0.120 and is diffuse.

**What complementary splitting does.** Model it as $Z_1=UF$, $Z_2=(1-U)F$, where $F$ is the concept, $U\perp\!\!\!\perp(T,F)$ is a coin, and the two pieces never co-fire. Frog version: the treatment paralyses one hind leg, left or right at random, and the dictionary has a "left leg drags" atom and a "right leg drags" atom. With $\mathbb{P}(F=1\mid T)=\tfrac12\pm\tfrac\tau2$, $\tau=0.5$ and $q=\mathbb{P}(U=1)=\tfrac12$ (measured, B.5):
- the concept carries $I(T;F)=I(T;Z_1,Z_2)=0.131$ nats;
- each piece alone carries $I(T;Z_1)=0.043$, about one third;
- each piece given the other carries $I(T;Z_1\mid Z_2)=0.088$, about two thirds.

**The pieces are synergistic, not redundant.** A piece alone confuses "concept absent" with "concept present in the other piece". Consequences:
1. **The guarantee is intact.** Each piece has positive unique information, so exact NESS keeps both.
2. **The forward step is diluted.** A split concept enters the forward step late: its first piece's gain is a third of the concept's information. Mixed atoms such as 408 can enter before it.
3. **Under-coverage under $\varepsilon$.** Take $\tau=0.3$: the concept carries 0.046 and each piece given the other 0.031. With $\varepsilon=0.04$, the *sequential* backward step drops one piece ($G=0.031\le\varepsilon$) and keeps the other ($G$ would be $0.046>\varepsilon$). The set stays $\varepsilon$-sufficient (Proposition 2), but it shows only half the concept.
4. **A sufficiency failure under simultaneous removal.** A rule that drops all individually removable atoms at once (DINES's terminal style) drops *both* pieces, and the set is then not $\varepsilon$-sufficient. That failure would be caught at certification. NESS therefore uses a **sequential** backward step only.
5. **Exclusion is a weak detector of complementary pieces.** Excluding the kept piece leaves residual information $I(T;Z_1\mid Z_{S\setminus j})\approx0.015$ (the other piece alone), which is below $\varepsilon$. So nothing enters. The **swap set** detects it instead (the other piece can replace the kept one, $G=0.031\le\varepsilon$), and so does the **residual-carrier ranking** (the other piece has the largest $I(T;Z_k\mid Z_S)$ among atoms outside $S$). See §6.5.

**What to do (proposals).**
- (a) Sequential backward elimination only (proved to keep sufficiency).
- (b) After selection, report a **concept group** per selected atom: the atom, its swap set, and its top residual carriers, together with the group's information $I(T;Z_{\rm group}\mid Z_{S\setminus j})$.
- (c) Optionally certify the group-completed set $S^+=S\cup\text{groups}$ as well, at $\alpha/2$ each. The groups must be built from $D_1$ or from $Z$ alone so that $S^+$ is fixed before $D_2$ is touched.
- (d) Group atoms *before* the analysis, and run NESS with groups as units.
  - Co-activation clustering cannot build the groups, because split pieces rarely co-fire.
  - Two $Z$-only options are **SAE decoder-direction similarity** and the **hierarchy of a Matryoshka SAE**.
  - **Both are untested proposals** (O6).

---

## 2. Causal content of sufficiency (R1)

Write $P_t:=P(Z\in\cdot\mid T=t)$, $K(\cdot\mid z_S):=P(Z_{-S}\in\cdot\mid Z_S=z_S)$ (pooled over arms), $h_g(z_S):=\mathbb{E}[g(Z)\mid Z_S=z_S]$ (pooled), $\mathrm{ATE}_g:=\mathbb{E}[g(Z)\mid do(T=1)]-\mathbb{E}[g(Z)\mid do(T=0)]$, and $\mathrm{osc}(g)=\sup g-\inf g\le2\|g\|_\infty$.

**Theorem 1 (exact sufficiency).** Let $T$ be randomized and $G(S)=0$. Then, for $t\in\{0,1\}$,
$$P(Z\mid do(T=t))=P(Z_S\mid do(T=t))\cdot K(Z_{-S}\mid Z_S),\qquad K\text{ the same for both arms},$$
and $\mathrm{ATE}_g=\mathrm{ATE}_{h_g}$ for every integrable $g$.

*Proof.* Randomization gives $P(Z\mid do(t))=P_t$. Sufficiency (Lemma 1) gives $P(Z_{-S}\mid Z_S,T=t)=K(\cdot\mid Z_S)$, $P(Z_S\mid T=t)$-a.s. Then $\mathbb{E}[g\mid T=t]=\mathbb{E}[\mathbb{E}[g\mid Z_S,T=t]\mid T=t]=\mathbb{E}[h_g(Z_S)\mid T=t]$. $\square$

**Theorem 2 (ε-sufficiency; R1 corrected).** Let $T$ be randomized, $G(S)\le\varepsilon$, and $\mathrm{osc}(g)<\infty$. Then
$$|\mathrm{ATE}_g-\mathrm{ATE}_{h_g}|\ \le\ \mathrm{osc}(g)\sqrt{\frac{\varepsilon}{2e(1-e)}}\ \le\ \|g\|_\infty\sqrt{\frac{2\varepsilon}{e(1-e)}} .$$
At $e=\tfrac12$ this is $\mathrm{osc}(g)\sqrt{2\varepsilon}\le2\sqrt2\,\|g\|_\infty\sqrt\varepsilon$.

*Proof.*
- Let $\tilde P_t:=P(Z_S\mid T=t)\otimes K$, and $K_t:=\mathrm{KL}(P_t\|\tilde P_t)=\mathbb{E}[\mathrm{KL}(P(Z_{-S}\mid Z_S,t)\|K(\cdot\mid Z_S))\mid T=t]$, since the $Z_S$-marginals agree.
- By Lemma 1, $eK_1+(1-e)K_0=G(S)\le\varepsilon$.
- $\mathbb{E}_{\tilde P_t}g=\mathbb{E}[h_g(Z_S)\mid T=t]$. So $\mathrm{ATE}_g-\mathrm{ATE}_{h_g}=\sum_t(\pm1)(\mathbb{E}_{P_t}g-\mathbb{E}_{\tilde P_t}g)$.
- Each term is at most $\mathrm{osc}(g)\,\mathrm{TV}(P_t,\tilde P_t)\le\mathrm{osc}(g)\sqrt{K_t/2}$ (Pinsker).
- By Cauchy–Schwarz, $\sqrt{K_1}+\sqrt{K_0}=e^{-1/2}\sqrt{eK_1}+(1-e)^{-1/2}\sqrt{(1-e)K_0}\le\sqrt{(1/e+1/(1-e))\,\varepsilon}=\sqrt{\varepsilon/(e(1-e))}$. $\square$

**About the bound first proposed.** The proposed bound $\|g\|_\infty(\sqrt{2\varepsilon/e}+\sqrt{2\varepsilon/(1-e)})$, i.e. $4\|g\|_\infty\sqrt\varepsilon$ at $e=\tfrac12$, is **correct but loose**.
- Its derivation is right: the decomposition of $I$ by arm, Pinsker, then Jensen.
- It bounds each arm separately by $K_t\le\varepsilon/\mathbb{P}(T=t)$ instead of using the constraint $eK_1+(1-e)K_0\le\varepsilon$ jointly. That loses a factor $\sqrt e+\sqrt{1-e}$, which is $\sqrt2$ at $e=\tfrac12$.
- Jensen is not needed, because Pinsker applies to the joint KL directly.
- Using $\mathrm{osc}(g)$ (equivalently, centring $g$) gains another factor of 2.
- Measured on 20,000 random discrete laws (B.1): the worst case of the true supremum over the corrected bound is **1.000**, so the corrected constant is attained; against the proposed bound the worst ratio is 0.83.

**What Theorems 1–2 say.** $Z_S$ is an ($\varepsilon$-deficient) *sufficient statistic* for the two-point family $\{P_0,P_1\}$: the likelihood ratio $dP_1/dP_0=\frac{1-e}{e}\frac{\pi(Z)}{1-\pi(Z)}$ is (nearly) a function of $Z_S$. Every treatment effect on any bounded feature of the representation is, up to the bound, an effect on a function of $Z_S$, pushed forward through a kernel that does not depend on the arm.

**What they do NOT say.**
1. **Not mediation.** $K$ is a conditional law, not a mechanism, and $Z_S$ need not cause $Z_{-S}$.
   - Marker frog: colour reveals $T$, so $S=\{Z_C\}$ is sufficient and $K(P\mid C=\text{treated colour})$ says "paralysed".
   - Yet dyeing a frog does not paralyse it.
   - Context atoms are in $S$ although the treatment does not change them at all.
2. **Not "intervening on $Z_S$ reproduces the effect".** Atoms are not manipulable. Even for the latent factors behind $Z_S$, $do(L_S=\ell)$ does not induce $K$, because $K$ mixes downstream mechanisms with explaining-away (backward) dependence.
3. **Not "atoms outside $S$ are unaffected".** Speed shifts; its shift is a function of the shift of $Z_S$.
4. **Not "$S$ is the set of directly changed factors".** $S$ may contain consequences (a coarse mediator, §8) and context atoms. Directness is NDES's question (§7).
5. **Relative to the reference representation.** The bound covers functions of $Z$ (or of whatever reference representation the full classifier sees; §3.2), and only bounded ones: for heavy-tailed atoms, apply it to truncated features.

---

## 3. Certification with sample splitting (R2)

### 3.1 Protocol

Split the units, or whole clusters, into $D_1$ and $D_2$, stratified by $T$. **Selection (forward + backward) uses only $D_1$**, so $\hat S=\hat S(D_1)$ is fixed when $D_2$ is analysed. On $D_2$ we run two certificates at levels $\alpha_s$ (sufficiency) and $\alpha_n$ (necessity), with $\alpha_s+\alpha_n=\alpha$.

This is NEXIS's precision principle (forward step plus terminal certification; precision through tests at *fixed* nulls; arXiv 2606.17010, Algorithm 1 and Theorem 4.1), implemented by splitting. Splitting makes every hypothesis tested on $D_2$ a fixed hypothesis for every selection rule, at the price of using about half the data for each task.

### 3.2 Sufficiency

**Estimator.**
- Split $D_2$ into a fitting part $D_{2a}$ and an evaluation part $D_{2b}$. (Cross-fitting over $K$ folds is the practical default; see the end of this subsection.)
- On $D_{2a}$, fit the $Z_S$ classifier $\hat f_S$ and the reference classifier $\hat f_{\rm full}$.
  - The reference classifier sees $Z$, or a richer reference representation such as all 9,216 atoms or the embedding the SAE was trained on.
  - It is **nested**: it boosts from $\mathrm{logit}\hat f_S$ as an offset (or stacks on it), so it cannot be worse than $\hat f_S$ by much.
- Clip both to $[\eta,1-\eta]$.
- On $D_{2b}$, let $d_i=\ell(T_i,\hat f_S(Z_{S,i}))-\ell(T_i,\hat f_{\rm full}(Z_i))$, with mean $\hat\Delta$, standard deviation $\hat s$, and upper confidence bound $U=\hat\Delta+z_{1-\alpha_s}\hat s/\sqrt{n_b}$.
- **Certify sufficiency iff $U\le\varepsilon$.**

**Estimand.** $\Delta_n(S):=\mathbb{E}[d\mid D_1,D_{2a}]=\mathcal{R}(\hat f_S)-\mathcal{R}(\hat f_{\rm full})$: the population log-loss gain of the fitted reference classifier over the fitted $Z_S$ classifier.

**Definition 3 ($\mathcal{F}$-sufficiency).** $S$ is **$\mathcal{F}$-$\varepsilon$-sufficient** (at sample size $n$) if $\Delta_n(S)\le\varepsilon$. In words: *the classifier family $\mathcal{F}$, trained on $n_a$ units, does not beat the $Z_S$ classifier by more than $\varepsilon$ nats on new units.* This is the honest content of the certificate. It is a statement about what $\mathcal{F}$ can find, not about all functions of $Z$.

**Lemma 5 (the loss difference is not degenerate away from zero).** Let $d=\log\{f(Z)_T/g(Z)_T\}$ for two classifiers $f,g$ (with $p_1=p$, $p_0=1-p$), and suppose $\mathbb{E}d\ge\varepsilon>0$. Then $\mathrm{Var}(d)>0$. If in addition $\mathbb{P}(T=t\mid Z)\ge\eta'$ a.s. for both $t$, then $\mathrm{Var}(d)\ge\eta'\varepsilon^2$.

*Proof.* For each $z$, $d(z,1)>0$ and $d(z,0)>0$ cannot both hold, since that would give $f_1+f_0>g_1+g_0=1$. So given $Z$, $d\le0$ with probability at least $\min_t\mathbb{P}(T=t\mid Z)$. Then $\mathrm{Var}(d)\ge\mathbb{E}[(d-\mathbb{E}d)^2;d\le0]\ge\varepsilon^2\,\mathbb{P}(d\le0)\ge\eta'\varepsilon^2$. A constant $d$ would need a constant $c$ with $e^c(g_1+g_0)=f_1+f_0$, i.e. $c=0<\varepsilon$. $\square$

This is why the null "gain $\ge\varepsilon$" can use a normal approximation while a null "gain $=0$" cannot (Williamson et al., 2023, show that variable-importance estimators are degenerate at zero importance).

**Theorem 3 (sufficiency certificate).** Fix $\varepsilon>0$, clipping $\eta$, and the learners.
- (a) *Finite-sample, $\mathcal{F}$-relative.* If $U$ is an empirical-Bernstein upper bound for variables in $[-2\log(1/\eta),2\log(1/\eta)]$ (Maurer & Pontil, 2009), then $\mathbb{P}(\text{certify and }\Delta_n(\hat S)>\varepsilon)\le\alpha_s$ exactly.
- (b) *Asymptotic, $\mathcal{F}$-relative.* With the normal bound and $\mathbb{P}(T=t\mid Z)\ge\eta'>0$: $\limsup_n\mathbb{P}(\text{certify and }\Delta_n(\hat S)>\varepsilon)\le\alpha_s$.
- (c) *Population link.* $\Delta_n(S)=G(S)+\mathrm{Exc}_S-\mathrm{Exc}_{\rm full}$, with excess risks $\mathrm{Exc}_S=\mathcal{R}(\hat f_S)-\mathcal{R}(\pi_S)\ge0$ and $\mathrm{Exc}_{\rm full}=\mathcal{R}(\hat f_{\rm full})-\mathcal{R}(\pi)\ge0$. Hence
  $$\mathbb{P}\big(\text{certify and }G(\hat S)>\varepsilon+\mathrm{Exc}_{\rm full}\big)\le\alpha_s\ (+o(1)\text{ in (b)}).$$
  If the reference learner is KL-consistent ($\mathrm{Exc}_{\rm full}\to0$ in probability; this needs $\pi\in[\eta,1-\eta]$ or $\eta=\eta_n\to0$), then for every $\kappa>0$, $\limsup_n\mathbb{P}(\text{certify and }G(\hat S)>\varepsilon+\kappa)\le\alpha_s$.

*Proof.*
- Conditionally on $(D_1,D_{2a})$, the $d_i$, $i\in D_{2b}$, are i.i.d., bounded, and have mean $\Delta_n(\hat S)$.
- (a) The empirical-Bernstein inequality holds conditionally at every value of $\Delta_n$; integrate over $(D_1,D_{2a})$.
- (b) On $\{\Delta_n>\varepsilon\}$, Lemma 5 gives conditional variance $\ge\eta'\varepsilon^2$, and the $d_i$ are bounded by $B=2\log(1/\eta)$. A Berry–Esseen bound for the studentized mean (Bentkus & Götze, 1996) bounds the conditional size by $\alpha_s+C B/(\varepsilon\sqrt{\eta' n_b})$, uniformly over the fitted classifiers. Integrate.
- (c) Lemma 2(1, 3): $\mathcal{R}(\hat f_S)-\mathcal{R}(\hat f_{\rm full})=[\mathcal{R}(\pi_S)-\mathcal{R}(\pi)]+\mathrm{Exc}_S-\mathrm{Exc}_{\rm full}$. Since $\mathrm{Exc}_S\ge0$, $G(S)\le\Delta_n(S)+\mathrm{Exc}_{\rm full}$. $\square$

**Read this plainly.**
- The certificate is **only as strong as the reference classifier**. A weak, misspecified or overfitted reference classifier biases the estimated gap *downward*, toward certifying. A linear reference classifier cannot see an XOR (§8). A reference classifier with 100+ features at $n_a\approx2000$ has parametric excess risk of order $p/(2n_a)\approx0.025$ nats, which is larger than a typical $\varepsilon$.
- A poor $Z_S$ classifier errs the other way: it is conservative.
- Three mitigations, all required in the implementation:
  1. the **nested** reference classifier: boost from the $Z_S$ offset with early stopping, so "nothing found" gives $\Delta\approx0$ rather than a negative, noise-driven gap;
  2. a **flexible** reference learner that can model interactions (gradient boosting);
  3. a **power witness**: report $\min_{j\in S}\mathrm{LCB}[\Delta(S\setminus\{j\})]$. If the same reference learner shows that removing a selected atom costs more than $\varepsilon$, it demonstrably detects information at that level. If it cannot, the certificate is uninformative and is reported as such.

**Cross-fitting** over $K$ folds of $D_2$ uses all of $D_2$. The estimand is then an average over folds of fold-specific $\Delta$'s. Inference is standard away from the zero boundary (Williamson et al., 2023); the single inner split above is the version with an exact conditional argument.

### 3.3 Necessity

For each $j\in\hat S$, test $H_{0,j}:\ T\perp\!\!\!\perp Z_j\mid Z_{\hat S\setminus\{j\}}$. By Lemma 1 this is $\mathbb{E}[T\mid Z_{\hat S}]=\mathbb{E}[T\mid Z_{\hat S\setminus\{j\}}]$, which is exactly the null of the **projected covariance measure** (PCM; Lundborg et al., 2024) with response $T$, tested variable $Z_j$, and conditioning set $Z_{\hat S\setminus\{j\}}$.
- The PCM is valid at the boundary under rate conditions on its nuisance regressions, and it has power against every alternative in which $\mathbb{E}[T\mid Z_{\hat S}]$ depends on $Z_j$, which by Lemma 1 is every alternative.
- A log-loss difference is *not* used here, because at the boundary ($I=0$) it is degenerate (Lemma 5; Williamson et al., 2023).
- The GCM with response $Z_j$ ([D, §3.1]) would see only mean shifts of $Z_j$.

**Theorem 4 (necessity certificate).** Suppose that, for every fixed $S$ with $|S|\le K_{\max}$ and every $j\in S$ for which $H_{0,j}(S)$ holds, the test is asymptotically valid: $\limsup_n\mathbb{P}(p_j(S)\le t)\le t$ for $t\in(0,1)$. Test each $j\in\hat S$ at level $\alpha_n/|\hat S|$ on $D_2$. Then
$$\limsup_n\mathbb{P}\big(\exists j\in\hat S:\ j\text{ certified necessary and }T\perp\!\!\!\perp Z_j\mid Z_{\hat S\setminus\{j\}}\big)\le\alpha_n .$$

*Proof.*
- Condition on $D_1$. Then $\hat S$ is a fixed set $S$ in the finite family $\mathcal{S}_{K_{\max}}$ of sets of size at most $K_{\max}$, and $|S|$ is fixed.
- The error probability is $\sum_{S}\mathbb{P}(\hat S=S)\,a_n(S)$, where $a_n(S)\le\sum_{j\in S:H_{0,j}}\mathbb{P}(p_j(S)\le\alpha_n/|S|)\le\alpha_n+r_n(S)$, with $r_n(S)\to0$ for each $S$.
- Hence the error probability is at most $\alpha_n+\max_{S\in\mathcal{S}_{K_{\max}}}r_n(S)\to\alpha_n$, because the family is finite. $\square$

**Plainly.** "Certified necessary" means *the atom carries information about $T$ that the other selected atoms do not.* It is **weaker than $\varepsilon$-necessity** ($G(S\setminus\{j\})>\varepsilon$, which minimality requires).
- By Lemma 3, $\varepsilon$-minimality implies 0-necessity, so the PCM certificate checks a necessary condition.
- At large $n$, an atom with tiny but non-zero conditional information passes it.
- Optional, stronger: certify $\varepsilon$-necessity with a lower confidence bound on $\Delta(S\setminus\{j\})>\varepsilon$. The boundary is at $\varepsilon>0$, so the bound is non-degenerate. It is $\mathcal{F}$-relative in the *anti-conservative* direction: a poor $Z_{S\setminus\{j\}}$ classifier inflates it. That classifier is low-dimensional, so the risk is smaller.

### 3.4 When certification fails

- **Sufficiency fails** ($U>\varepsilon$). Report "**not certified at $\varepsilon$**" together with $U$. The statement "$\hat S$ leaves at most $U$ nats that $\mathcal{F}$ can find" is still valid at level $\alpha_s$, simultaneously for every $\varepsilon$, because it is a single confidence bound. So the residual-information metric is always reportable.
- **Do not** enlarge $\hat S$ using $D_2$ and re-certify on $D_2$. That selects on the certification data, and the certificate becomes invalid.
- *Allowed:* re-run selection on $D_1$ (a larger cap, or a smaller internal threshold) and certify again on $D_2$. Split $\alpha_s$ by Bonferroni over a number of attempts **declared in advance**: each attempt's set depends only on $D_1$, so the union bound holds. Or collect new data.
- **Necessity fails** for some $j$. Keep $j$ in the reported set, flagged "not certified necessary". Removing it would change the set whose sufficiency was certified. If dropping uncertified atoms is wanted, declare it in advance and certify the reduced set too, at a split level.

---

## 4. Consistency of the selection (R3)

**Population algorithm $\mathcal{A}(\varepsilon,K_{\max})$.**
- *Forward:* $S\leftarrow\emptyset$; while $G(S)>\varepsilon$ and $|S|<K_{\max}$, add some $j^\star\in\arg\max_j I(T;Z_j\mid Z_S)$.
- *Backward (sequential):* while some $j\in S$ has $G(S\setminus\{j\})\le\varepsilon$, remove the one with the smallest $I(T;Z_j\mid Z_{S\setminus\{j\}})$.

**Proposition 2 (the population algorithm needs no composition).** If the forward step ends with $G(S)\le\varepsilon$ (always the case with $K_{\max}=m$), the output is in $\mathrm{NESS}(\varepsilon)$.

*Proof.* Each backward removal keeps $\varepsilon$-sufficiency, and the loop ends after at most $|S|$ removals. At the end no single removal keeps sufficiency, so the set is minimal by Lemma 3. $\square$

The forward step stops on *sufficiency*, not on "no single atom is informative" (IAMB's and NES's rule). So an XOR pair does not make it stop early. It makes it add uninformative atoms until it stumbles on the pair, and the backward step then removes the junk. Composition-type conditions matter only for **reaching sufficiency within the cap**, which matters in finite samples and for cost.

**Sample version.** On $D_1$, use cross-fitted estimates $\hat I(j\mid S)$ of $I(T;Z_j\mid Z_S)$ and $\hat G(S)$ (or its held-out upper bound) of $G(S)$ (§9).

**Theorem 5 (consistency of the selection).** Let $\mathcal{S}_{K+1}$ be the sets of size at most $K_{\max}+1$. Assume:
- **(M) Margin.** $|G(S)-\varepsilon|\ge\gamma>0$ for every $S\in\mathcal{S}_{K+1}$. In words: no set of up to $K_{\max}+1$ atoms has residual information within $\gamma$ of $\varepsilon$. This contains "necessary atoms carry more than $\varepsilon$ plus a gap, unnecessary ones less than $\varepsilon$ minus a gap".
- **(E) Consistent estimates.** For each fixed $S\in\mathcal{S}_{K+1}$ and $j$, $\hat G(S)\to G(S)$ and $\hat I(j\mid S)\to I(T;Z_j\mid Z_S)$ in probability. The family is finite, so this is uniform.
- **(GC) Greedy completeness at the cap.** For some $\gamma'>0$, every *$\gamma'$-greedy* population path reaches a set with $G\le\varepsilon$ within $K_{\max}$ steps. A $\gamma'$-greedy path is $\emptyset=S_0\subset S_1\subset\cdots$, where each step adds a $j$ with $I(T;Z_j\mid Z_{S_k})\ge\max_iI(T;Z_i\mid Z_{S_k})-\gamma'$.

Then $\mathbb{P}(\hat S_n\in\mathrm{NESS}(\varepsilon))\to1$. If $\mathrm{NESS}(\varepsilon)$ has a single element $S^\star$ (e.g. Theorem 7), then $\mathbb{P}(\hat S_n=S^\star)\to1$.

*Proof.*
- Let $E_n$ be the event that every estimation error over the finite family is below $\min(\gamma,\gamma'/2)$. Then $\mathbb{P}(E_n)\to1$.
- On $E_n$, every sufficiency decision for a set in $\mathcal{S}_{K+1}$ is correct, by (M).
- Each forward choice has true gain within $\gamma'$ of the maximum. So the sample path is a $\gamma'$-greedy population path, and by (GC) it reaches a sufficient set within the cap.
- Backward decisions concern sets $S\setminus\{j\}\in\mathcal{S}_{K+1}$, so they are correct, and the argument of Proposition 2 applies. $\square$

**Proposition 3 (a quantitative composition condition gives the cap).** Let $S^\circ$ be an $\varepsilon^\circ$-sufficient set with $\varepsilon^\circ<\varepsilon$ and $|S^\circ|=s$. Define the *submodularity ratio* (Das & Kempe, 2011)
$$\gamma_{\rm sub}:=\min_{|S|\le K_{\max}}\frac{\sum_{j\in S^\circ\setminus S}I(T;Z_j\mid Z_S)}{I(T;Z_{S^\circ\setminus S}\mid Z_S)}\qquad(0/0:=1).$$
If $\gamma_{\rm sub}>0$, every $\gamma'$-greedy path satisfies
$$G(S_k)-\varepsilon^\circ\le(1-\gamma_{\rm sub}/s)^k\,I(T;Z)+\gamma's/\gamma_{\rm sub}.$$
So (GC) holds when $\varepsilon-\varepsilon^\circ>\gamma's/\gamma_{\rm sub}$ and $K_{\max}\ge(s/\gamma_{\rm sub})\log\big(I(T;Z)/(\varepsilon-\varepsilon^\circ-\gamma's/\gamma_{\rm sub})\big)$.

*Proof.*
- By the chain rule, $I(T;Z_{S^\circ\setminus S}\mid Z_S)=G(S)-G(S\cup S^\circ)\ge G(S)-\varepsilon^\circ$.
- So $\max_jI(T;Z_j\mid Z_S)\ge\frac1s\sum_{j\in S^\circ\setminus S}I(\cdot)\ge\frac{\gamma_{\rm sub}}{s}(G(S)-\varepsilon^\circ)$.
- A $\gamma'$-greedy step reduces $G$ by $I(T;Z_{j}\mid Z_S)\ge\frac{\gamma_{\rm sub}}{s}(G(S)-\varepsilon^\circ)-\gamma'$.
- Iterate, and use $G(\emptyset)\le I(T;Z)$ and $(1-x)^k\le e^{-xk}$. $\square$

An XOR pair has $\gamma_{\rm sub}=0$: no guarantee. The stall breaker of §9 handles it, and certification catches what it misses.

---

## 5. The ECI paradox (R4)

**Proposition 4 (fixed target; the exact target can be the whole cluster).**
- (a) $\mathrm{NESS}(\varepsilon)$ is a functional of the law of $(T,Z)$; it does not depend on $n$. Under the conditions of Theorem 5, with probability tending to 1, $|\hat S_n|\le\max\{|S|:S\in\mathrm{NESS}(\varepsilon)\}$, a fixed number. **At fixed $\varepsilon$ the output does not grow with $n$.**
- (b) *Replicate atoms.* Let $F=\beta T+\mathcal{N}(0,1)$ and $Z_k=F+\sigma\eta_k$, $k=1,\dots,m$, with i.i.d. $\eta_k\sim\mathcal{N}(0,1)$.
  - Then $u_k>0$ for every $k$, the law has a positive density, and $\mathrm{NESS}(0)=\{[m]\}$: **all $m$ atoms**.
  - For $\varepsilon>0$, $\mathrm{NESS}(\varepsilon)$ is the family of *all* subsets of size $s(\varepsilon)=\min\{s:I(T;\bar Z_m)-I(T;\bar Z_s)\le\varepsilon\}$, where $\bar Z_s$ is the mean of $s$ atoms.
  - Measured at $\beta=1$, $\sigma=2$, $m=50$ (B.4): $I(T;Z)=0.104$ nats, $u_k=1.4\times10^{-4}$ nats, and $s(\varepsilon)=50,44,29,20,11$ at $\varepsilon=0,0.001,0.005,0.01,0.02$.

*Proof.* (a) Definition 1 involves only the law. The bound follows from $\mathbb{P}(\hat S_n\in\mathrm{NESS}(\varepsilon))\to1$. (b) By exchangeability and Gaussianity, $\bar Z_m$ is sufficient for $F$, and $T\to F\to Z$. So $T\perp\!\!\!\perp Z\mid\bar Z_m$, and $G(S)=I(T;\bar Z_m)-I(T;\bar Z_S)$, which depends only on $|S|$ and is strictly decreasing in it. Hence every set of size $\ge s(\varepsilon)$ is sufficient and no smaller set is, and the inclusion-minimal sets are exactly those of size $s(\varepsilon)$. At $\varepsilon=0$, $s=m$. A positive density gives the intersection property, so Proposition 1 applies. $\square$

**The paradox, stated precisely.** An imperfectly aligned dictionary spreads each factor over many atoms, each with a little unique information. The exact target (the Markov boundary) then contains all of them. A consistent procedure aimed at it — IAMB, NEXIS-style tests at $\delta=0$, NES, DINES at $\delta=0$ — returns more of them as $n$ grows. *Heuristic:* a test at level $\alpha/m$ detects conditional information above roughly $z^2_{\alpha/m}/(2n)$ nats (§1.5), so a test-based method behaves like NESS with an **implicit $\varepsilon_n\approx z^2/(2n)\to0$**. With $m=10^4$, $\alpha=0.05$ and $n=5000$, $\varepsilon_n\approx0.002$ nats. Fixing $\varepsilon$ fixes the target.

**What fixing $\varepsilon$ does not do.** It does not make a mixed atom disappear. On CelebA:
- the mixed atom 408 (F1 0.48 for Wearing_Hat in the saved NES runs [D, §4.1]; best attribute Big_Nose, F1 0.36) remains conditionally dependent on $T$ given the selected atoms, with partial correlation 0.19. This number was reported by the benchmark diagnostics and is not yet recorded in RESULTS.md;
- by §1.5 that is roughly 0.018 nats;
- DINES at $\delta=0$ keeps it in 33/80 runs at $N=5000$ (RESULTS.md, original design);
- so 408 belongs to $\mathrm{NESS}(\varepsilon)$ for any $\varepsilon$ below roughly 0.018 nats, **at every $n$**.

Precision against "one atom per attribute" is a property of the dictionary relative to $\varepsilon$. NESS makes the target stable and honest (408 *does* carry treatment information that the hat atom misses). It does not make the target equal to the attribute list.

---

## 6. The exclusion analysis

### 6.1 Population definition

Fix $S\in\mathrm{NESS}(\varepsilon)$ and $j\in S$. Use the dictionary without $j$ as the reference:
$$G^{(-j)}(A):=I(T;Z_{[m]\setminus\{j\}}\mid Z_A),\qquad A\subseteq[m]\setminus\{j\}.$$

**Definition 4.**
- The **exclusion family** $\mathcal{R}_j(\varepsilon)$ is the set of inclusion-minimal $R\subseteq[m]\setminus S$ with $G^{(-j)}((S\setminus\{j\})\cup R)\le\varepsilon$. It is non-empty, because $R=[m]\setminus S$ works.
- The algorithm reports one member $R_j$.
- Two algorithm-free scalars: the **replaceable information** $\iota_j:=G^{(-j)}(S\setminus\{j\})=I(T;Z_{-S}\mid Z_{S\setminus\{j\}})$, and the unique (**irreplaceable**) information $u_j=I(T;Z_j\mid Z_{-j})$.
- One algorithm-free profile, the **retention ratio**: $r_{jk}:=I(T;Z_k\mid Z_S)\,/\,I(T;Z_k\mid Z_{S\setminus\{j\}})$ for $k\notin S$.

**Proposition 5 (decomposition).** $G(S\setminus\{j\})=\iota_j+u_j$. So the information $j$ carries beyond the rest of $S$ splits into a part that other atoms of the dictionary also carry ($\iota_j$, what the exclusion run recovers) and a part only $j$ carries ($u_j$). Moreover $R_j=\emptyset$ is a completion iff $\iota_j\le\varepsilon$.

*Proof.* Chain rule: $I(T;Z_j,Z_{-S}\mid Z_{S\setminus\{j\}})=I(T;Z_{-S}\mid Z_{S\setminus\{j\}})+I(T;Z_j\mid Z_{-j})$. $\square$

Report $\iota_j/G(S\setminus\{j\})$, the *replaceable fraction*. An atom with $\iota_j\approx0$ is **irreplaceable**: the dictionary has no other carrier of its treatment information.

**Proposition 6 (exact case: the exclusion set is unique and algorithm-free).** If the law of $(T,Z)$ has the intersection property, then
$$\mathcal{R}_j(0)=\{\mathrm{MB}(T;Z_{-j})\setminus S\},\qquad \mathrm{MB}(T;Z_{-j})=\{k\ne j:\ I(T;Z_k\mid Z_{[m]\setminus\{j,k\}})>0\}.$$
If moreover $S=\mathrm{MB}^\star$ (i.e. $\varepsilon=0$), every $k\in R_j$ satisfies $T\perp\!\!\!\perp Z_k\mid Z_{-k}$ and $T\not\perp\!\!\!\perp Z_k\mid Z_{-\{j,k\}}$. In words: **$j$ screens $k$ from the treatment, given everything else.**

*Proof.*
- The intersection property passes to the marginal law of $(T,Z_{-j})$.
- By Proposition 1 applied to that law, $\mathrm{MB}_{-j}:=\mathrm{MB}(T;Z_{-j})$ is its unique Markov boundary.
- Every completion $R$ contains $\mathrm{MB}_{-j}\setminus S$. If $k\in\mathrm{MB}_{-j}\setminus S$ were not in $(S\setminus\{j\})\cup R$, weak union would give $I(T;Z_k\mid Z_{[m]\setminus\{j,k\}})=0$.
- $(S\setminus\{j\})\cup(\mathrm{MB}_{-j}\setminus S)\supseteq\mathrm{MB}_{-j}$ is a blanket within $Z_{-j}$, by weak union.
- So $\mathrm{MB}_{-j}\setminus S$ is the unique minimal completion.
- The second claim holds because $k\notin S=\mathrm{MB}^\star=\{u>0\}$. $\square$

For $\varepsilon>0$, $\mathcal{R}_j(\varepsilon)$ can have several members, like $\mathrm{NESS}(\varepsilon)$ itself. Report the member found, its swap sets, and the retention ratios $r_{jk}$ of the top entering atoms. Those ratios do not depend on the algorithm.

### 6.2 Masked or carried

**Proposition 7 (masking bound).** Let $j\in S$, $C:=S\setminus\{j\}$, $k\notin S$. Write
$$\eta_j:=\operatorname*{ess\,sup}_c\ \mathrm{TV}\big(P(Z_j\mid T=1,Z_C=c),\,P(Z_j\mid T=0,Z_C=c)\big)\in[0,1]$$
for the **overlap failure** of $j$ given $C$. At $C=\emptyset$, $\eta_j=\mathbb{E}|\pi_{\{j\}}-e|/(2e(1-e))$ and $1-\eta_j$ is the overlap coefficient of the two arm laws of $Z_j$. If $Z_k\perp\!\!\!\perp Z_j\mid(T,Z_C)$ (*$k$ and $j$ are separate readings of the treatment*), then
$$I(T;Z_k\mid Z_S)\ \ge\ (1-\eta_j)\,I(T;Z_k\mid Z_C),\qquad\text{i.e. } r_{jk}\ge1-\eta_j .$$

*Proof.*
- The identity $I(T;Z_k\mid Z_j,Z_C)=I(T;Z_k\mid Z_C)-I(Z_k;Z_j\mid Z_C)+I(Z_k;Z_j\mid T,Z_C)$ holds, and its last term is 0 under the assumption.
- Given $Z_C=c$, $Z_k-T-Z_j$ is a Markov chain. The strong data-processing inequality gives $I(Z_k;Z_j\mid c)\le\eta_{\rm KL}(c)\,I(Z_k;T\mid c)$, where $\eta_{\rm KL}(c)$ is the contraction coefficient of the channel $T\mapsto Z_j$ given $c$.
- For any channel $\eta_{\rm KL}\le\eta_{\rm TV}$ (Dobrushin's coefficient; Cohen, Kemperman & Zbăganu, 1998; Polyanskiy & Wu, 2017). With a binary input, $\eta_{\rm TV}(c)$ is the TV distance between the two rows.
- Average over $c$, then bound $\eta_{\rm KL}(c)\le\eta_j$.
- The formula at $C=\emptyset$ follows from $p_1-p_0=p\,(\pi-e)/(e(1-e))$. $\square$

Measured on 50,000 random discrete laws (B.6), the bound holds with worst-case ratio 1.000, so it is tight.

**Corollary (what the data can say about each entering atom).**
- If $r_{jk}<1-\eta_j$, then $Z_k\not\perp\!\!\!\perp Z_j\mid(T,Z_C)$. Atom $k$ shares information with $j$ *within each arm*, beyond the treatment: **$k$ is carried by $j$.** This needs no causal assumption.
- If $j$ has good overlap ($\eta_j$ small), every atom whose information $j$ substantially removed is carried.
- If $j$ nearly reveals $T$ ($\eta_j\approx1$), the bound is vacuous, and $k$ may be a separate effect **masked** by $j$.
- A direct check is a conditional-independence test of $Z_k\perp\!\!\!\perp Z_j\mid(T,Z_C)$.
- How much overlap failure is needed depends on $\varepsilon$. For a separate effect carrying $\iota$ nats given $C$ to fall below $\varepsilon$ when $j$ is added, $\eta_j\ge1-\varepsilon/\iota$ is needed.

| overlap failure $\eta_j$ | $k$ and $j$ dependent within arms? | reading of $k\in R_j$ | causal reading (needs the latent model) |
|---|---|---|---|
| small | yes | **carried**: proxy or consequence of $j$ | $k$ reads a descendant or ancestor of $j$'s factor, or another effect of $j$ when $j$ is a co-parent (Lemma 8) |
| small | no | entry only by a threshold effect: $k$ loses at most a fraction $\eta_j$ (Prop. 7) | — |
| large | no | **masked**: a separate effect hidden by $j$ | a separate child of $T$ |
| large | yes | **mixed**: masked and/or carried; not separable | not identified ([D, Prop. 4]) |

**Lemma 8 (with exact atoms and faithfulness, screening means a causal link).** Assume A1 of [D] with atoms equal to factors, and a law faithful to $\mathcal{G}_L$. Let $Q:=[m]\setminus\{j,k\}$. If $T\perp\!\!\!\perp Z_k\mid Z_Q,Z_j$ but $T\not\perp\!\!\!\perp Z_k\mid Z_Q$, then $j$ is a non-collider on a path between $T$ and $k$ that is d-connecting given $Q$, and $Z_j\not\perp\!\!\!\perp Z_k\mid(T,Z_Q)$. In particular, **at $\varepsilon=0$, masking without within-arm dependence is impossible under faithfulness**. It needs a faithfulness failure, such as a deterministic marker, or $\varepsilon>0$.

*Proof.*
- Since $T\not\perp\!\!\!\perp Z_k\mid Z_Q$, the global Markov property gives a path $p$ between $T$ and $k$ that d-connects given $Q$. Since $T\perp\!\!\!\perp Z_k\mid Z_Q,Z_j$, faithfulness gives $T\perp_d k\mid Q\cup\{j\}$, so every such path is blocked given $Q\cup\{j\}$.
- Adding $j$ blocks a path only if $j$ is a non-collider on it (adding a vertex can only open colliders).
- Take the sub-path of $p$ from $j$ to $k$. It does not contain $T$. Its colliders have descendants in $Q$, and its non-colliders are outside $Q\cup\{T\}$. So it d-connects $j$ and $k$ given $Q\cup\{T\}$.
- Faithfulness turns this d-connection into dependence. $\square$

The non-collider position of $j$ gives three shapes:
- $\to j\to$: $k$ is downstream of $j$ (a consequence; *excluding paralysis brings speed in*);
- $\leftarrow j\to$ with $j$ a co-parent: $k$ is another effect of $j$ (*excluding size brings in other size-readers, e.g. speed if speed depends on size*);
- $\leftarrow j\leftarrow$: $k$ is upstream of $j$.

### 6.3 What can and cannot be claimed

- **Can (no causal assumption).**
  - "$R_j$ carries treatment information that $j$ makes redundant."
  - The scalars $\iota_j$, $u_j$, $\eta_j$ and $r_{jk}$.
  - Within-arm dependence of each $k\in R_j$ with $j$.
  - The table's informational column.
  - The **indirect-effect output**: for an atom $j$ with good overlap, $R_j$ is the list of atoms whose treatment shift is carried by $j$.
- **Cannot without assumptions.**
  - Whether a carried atom is a consequence, a proxy, or shares a cause with $j$ (Lemma 8 needs exact atoms and faithfulness).
  - Whether a masked atom is changed "by $T$" or "by $j$" when $j$ is a perfect marker. This is not identified ([D, Prop. 4]).
  - In the mixed row, nothing separates the two readings.

### 6.4 Why exclusion avoids the overlap failure

Conditioning on a near-marker leaves almost no units of one arm at each value of the marker ($\pi_A\in\{0,1\}$). Every null given it then holds vacuously ([D, Lemma 1.2]), and no test can say anything. Exclusion never conditions on $j$: the conditioning sets are subsets of $(S\setminus\{j\})\cup R$, so they keep overlap as long as no *other* marker is present.
- With several markers (colour and an exact "sheen" copy), excluding one leaves the other. Exclude the whole **marker group**: all atoms with $\eta_j$ above a threshold declared in advance.

### 6.5 Exclusion and feature splitting

| kind of split | exclusion detects it? | what does |
|---|---|---|
| near-duplicate ($Z_{j'}\approx Z_j$) | **yes**: $R_j\ni j'$, with near-maximal within-arm dependence; also $j'\in\mathrm{Sw}(j)$ | exclusion, swap set |
| redundant split (whole + parts) | yes: the parts or the whole enter | exclusion |
| **complementary split** (pieces rarely co-fire; common on CelebA, §1.6) | **usually not**: if the other piece is outside $S$, its information *alone* is below $\varepsilon$ ($I(T;Z_1)=0.015<\varepsilon=0.04$ in §1.6) | **swap set** (the other piece replaces the kept one) and **residual-carrier ranking** ($\arg\max_{k\notin S}I(T;Z_k\mid Z_S)$) |

Running the exclusion analysis at a smaller internal threshold ($\varepsilon_x=\varepsilon/2$ by default; O11) widens what enters, at the cost of more noise.

---

## 7. NDES: the direct level

**Definition 5 (NDES on a set $S$).** Let $\mathcal{O}_S:=\{A\subseteq S:A\in\mathcal{O}\}$. For $j\in S$:
- **direct** if $T\not\perp\!\!\!\perp Z_j\mid Z_A$ for every $A\in\mathcal{O}_S$ with $A\subseteq S\setminus\{j\}$;
- **context** if $T\perp\!\!\!\perp Z_j\mid Z_A$ for some such $A$;
- **undetermined** otherwise, i.e. when the only sets explaining $j$ away lack overlap. There conditional independence holds vacuously.

The $\delta$ version replaces "$T\perp\!\!\!\perp Z_j\mid Z_A$" by "$I(T;Z_j\mid Z_A)\le\delta$" (or $|\rho|\le\delta$ as in [D]). This is the DINES terminal rule ([D, §2.5]) run **inside $S$** with the conditional-independence null of Lemma 1. Notation for the latent model ($\mathrm{Ch}$, $\mathrm{Sp}$, $\mathrm{MB}$, $J_C$, $B_Z$, $D_s$, assumptions A1, 2a, 2b, 3) is that of [D, §§1, 2.1].

**Theorem 6 (NDES, exact).** Assume A1, assumptions 2 (2a screening, 2b purity) and 3 (no hidden co-parent / coverage) of [D], $B_Z\in\mathcal{O}$, and:
- **(F1) total-conditioning dependence on the blanket:** $u_j>0$ for every $j\in B_Z$;
- **(F2) no cancellation at the children (conditional-independence form):** $T\not\perp\!\!\!\perp Z_{j_c}\mid Z_A$ for every child $c$ and every $A\in\mathcal{O}$ with $j_c\notin A$.

Then:
- (i) $\mathrm{NESS}(0)=\{B_Z\}$: the exact NESS set is the atoms of children $\cup$ co-parents;
- (ii) NDES on $S=B_Z$ labels exactly $J_C$ as direct and $B_Z\setminus J_C$ as context, and nothing as undetermined.

*Proof.*
- (i) By [D, Prop. 1] (which uses Lemma 4(a) there, with 2a and 3), $T\perp\!\!\!\perp Z_{-B_Z}\mid Z_{B_Z}$, so $B_Z$ is 0-sufficient. By (F1) and Lemma 4, every 0-sufficient set contains $B_Z$. So $B_Z$ is the only minimal one.
- (ii) Children: since $B_Z\in\mathcal{O}$ and $\mathcal{O}$ is closed under subsets, every $A\subseteq B_Z\setminus\{j_c\}$ is in $\mathcal{O}_S$, and (F2) applies.
- Co-parent atoms $j_s$, $s\in\mathrm{Sp}(T)\setminus\mathrm{Ch}(T)$: [D, Thm. 1 item 3] (separator lemma [D, Lemma 3(b)] plus [D, Lemma 4(b)] with 2a, 2b) gives $T\perp\!\!\!\perp Z_{j_s}\mid Z_{J_{D_s}}$, with $D_s=(\mathrm{MB}(T)\cap\mathrm{An}(s))\setminus\{s\}$. Hence **$J_{D_s}\subseteq B_Z\setminus\{j_s\}=S\setminus\{j_s\}$: the separator lies inside $S$**, and it is in $\mathcal{O}$.
- The undetermined clause is empty because every subset of $S$ has overlap. $\square$

**Corollary 1 (NDES is robust to a set that is too large).** If $S\supseteq B_Z$ and $S\in\mathcal{O}$, NDES on $S$ still returns exactly $J_C$. An extra atom $j\notin B_Z$ has the separator $B_Z\subseteq S\setminus\{j\}$ ([D, Thm. 1 item 2]).

So leaks and too small an $\varepsilon$ hurt NESS's size, but not NDES's answer. A set that is **too small** hurts NDES's precision: if a co-parent is missing, an atom whose separator needs it can be labelled direct.

**Finite-sample NDES.** Run on $D_2$, given $\hat S$ from $D_1$.
- A non-child $j$ is falsely labelled direct only if its test rejects at its *fixed* separator $A_j\subseteq\hat S\setminus\{j\}$.
- With per-atom level $\alpha'/|\hat S|$ (one fixed null per atom, as in [D, Thm. 2(iii)]), $\mathbb{P}(\text{any false direct}\mid\hat S\supseteq B_Z)\le\alpha'+o(1)$.
- No validity at data-chosen sets is needed, because $\hat S$ is fixed by $D_1$.

**Theorem 7 (NESS and NDES in an $\varepsilon/\delta$ window; no exact alignment).** Assume A1. For each $k\in\mathrm{MB}(T)$ pick a distinct atom $j_k$ (its best atom; no alignment assumed). Define $B_Z$, $J_C$ and $J_{D_s}$ as before. Suppose:
- **(W1)** the **joint leak** is small: $G(B_Z)\le\varepsilon$;
- **(W2)** every blanket atom is $\varepsilon$-strongly relevant: $u_j>\varepsilon$ for $j\in B_Z$.

Then $\mathrm{NESS}(\varepsilon)=\{B_Z\}$. If moreover
- **(W3)** $B_Z\in\mathcal{O}$, and every co-parent atom has a small leak at its separator: $I(T;Z_{j_s}\mid Z_{J_{D_s}})\le\delta$;
- **(W4)** children are above the margin: $I(T;Z_{j_c}\mid Z_A)>\delta$ for every $A\subseteq B_Z\setminus\{j_c\}$;

then NDES$_\delta$ on $B_Z$ returns $J_C$. Combined with Theorem 5 and the finite-sample paragraph, $\liminf_n\mathbb{P}(\text{NDES output}=J_C)\ge1-\alpha'$.

*Proof.* By (W2) and Lemma 4, every $\varepsilon$-sufficient set contains $B_Z$. By (W1), $B_Z$ is $\varepsilon$-sufficient, so it is the unique minimal one. The NDES claims restate (W3) and (W4) against Definition 5. $\square$

**What is missing for a full $\varepsilon/\delta$ theory.**
1. **No bound on the joint leak $G(B_Z)$ from per-atom alignment quality.** [D, Props. 5–6] bound one atom's leak on the $\rho$ scale. $G(B_Z)$ is the *joint* information of all $m-|B_Z|$ non-blanket atoms, and many small leaks can add up (Conjecture 1).
2. **(W2) can fail because of proxies.** A good proxy of $P$ (such as speed) lowers $u_{j_P}$. With imperfect atoms the window $G(B_Z)\le\varepsilon<\min_{B_Z}u_j$ may be empty. Then NESS is still well defined, but it is not $B_Z$.
3. **Markers.** With a marker, (F1) fails for the atoms it masks, and $B_Z\notin\mathcal{O}$. NESS returns the marker, the exclusion analysis supplies the masked atoms, and their NDES status is "undetermined" ([D, Prop. 4]).

---

## 8. Counterexamples on the frog

All use $T\sim\mathrm{Bern}(1/2)$. "Atoms = factors" unless stated.

| case | NESS($\varepsilon$) | NESS tags | exclusion | NDES | identifiable from $(T,Z)$? | what a scientist prefers to report |
|---|---|---|---|---|---|---|
| **1. context (size)** | $\{P,C,W\}$ if $\varepsilon<I(T;W\mid P,C)$ | $P,C$ effect; $W$ context | $R_W$ = other size-readers (e.g. $V$ if speed depends on size) | $P,C$ direct; $W$ context (separator $\emptyset$) | yes | "size modifies the paralysis effect"; not an effect of $T$ |
| **1'. co-cause that descends from $T$ (temperature $\Theta$)** | $\{P,C,\Theta\}$ | $\Theta$ **effect** | — | $\Theta$ **context** (separator $\{C\}\subseteq S$) | yes, under NDES's assumptions | "temperature changes (via colour) and matters for paralysis" |
| **2. marker (colour ≈ $T$)** | $\{C\}$ | $C$ effect | $R_C=\{P,W\}$, $\eta_C\approx1$, $P\perp\!\!\!\perp C\mid T$: **masked** | $C$ direct; $P$ undetermined (if added) | which effect is "by $T$" vs "by colour" is **not** identified ([D, Prop. 4]) | colour and paralysis both as effects |
| **3. XOR (handedness × dragged side)** | $\{X_1,X_2\}$ (exact target fine) | $X_1$ context, $X_2$ context (both marginally independent of $T$) | — | — | yes | "the treatment paralyses the dominant leg" |
| **4a. duplicates ($Z_P=Z_{P'}$)** | $\{P,C,W\}$ or $\{P',C,W\}$ | — | $R_P=\{P'\}$, dependence ≈ max: **duplicate** | either copy direct | which copy: no (and irrelevant) | one concept, two atoms |
| **4b. complementary split (left/right leg drags)** | both pieces if $\varepsilon$ is small; **one piece** if $\varepsilon\in$ (piece-given-other, concept) | both effect | weak; **swap set** finds the other piece | both direct ([D, Prop. 7] clean split) | yes | "hind-leg paralysis" as one group |
| **5. coarse mediator ($Z_P=\mathbb{1}[P>0]$)** | $\{Z_P,C,W,V\}$: speed enters | $V$ effect | $R_{Z_P}$ may be empty; $R_V$ = other severity proxies | **$V$ direct (wrong)** | mediation not identified ([D, Prop. 3]) | "speed drops because of paralysis" — needs a better atom |
| **6. hidden physiology (lethargy $L$, $T\to L\to V$, no atom)** | $\{P,C,W,V\}$ | $V$ effect | $R_V=\emptyset$ ($V$ irreplaceable) | $V$ direct, **correct relative to the dictionary** (fold $L$ into $V$, [D, Lemma 2]) | yes, relative to the dictionary | "something besides paralysis slows the frog" |
| **7. collider bias (hidden size, $T\to P\leftarrow W\to V$, no size atom)** | $\{P,C,V\}$ | $V$ effect | $R_P\ni V$ | **$V$ direct (wrong)** | no: same law as case 6 ([D, Prop. 3]) | cannot be told from case 6 |

**1. Context atom.** Take the explicit table of [D, Prop. 1] (large frog w.p. ½; paralysis probability 0.9 for small treated, 0.5 for large treated, 0.05 for controls). Measured (B.3): $I(T;P)=0.257$, $I(T;P,W)=0.286$, $I(T;W\mid P)=0.029$ nats.
- So $\mathrm{NESS}(\varepsilon)=\{P,W\}$ for $\varepsilon<0.029$, and $\{P\}$ above it.
- Size enters only through explaining away: given paralysis, size tells how surprising the paralysis is.
- **Design consequence:** context atoms are legitimate members of the target, and the tag (§1.4) is what keeps them from being read as effects.
- NES's S design is this structure (Eyeglasses as the co-parent of Hat): NES selecting the spouse is *correct* for NESS and incorrect for DINES (§11).

**2. Marker.** Colour equal to $T$ with probability 0.99: $H(T\mid C)=0.056$ nats (B.3), so $G(\{C\})\le0.056$.
- For $\varepsilon\ge0.056$, $\{C\}$ is sufficient by construction. With an exact copy, $\{C\}$ is the unique element of $\mathrm{NESS}(\varepsilon)$ for every $\varepsilon\ge0$.
- Paralysis disappears from the target, which is correct as information: given colour there is nothing left to learn about $T$.
- The exclusion analysis recovers it. $R_C=\mathrm{MB}(T;Z_{-C})\setminus S=\{P,W\}$ (Proposition 6). Paralysis is within-arm independent of colour, so it is read as **masked**; $W$ enters as context.
- **Design consequence:** exclusion replaces conditioning on the marker, which would have no overlap.
- This is the mice "bag" neuron case ([D, §2.9]).

**3. XOR.** Handedness $X_1\sim\mathrm{Bern}(1/2)$ is pre-treatment. A treated frog drags its dominant leg with probability 0.9; a control drags a random leg. So the dragged side $X_2$ is $\mathrm{Bern}(1/2)$ in both arms.
- $I(T;X_1)=I(T;X_2)=0$, while $I(T;X_1,X_2)=h(0.7)-\tfrac12(h(0.9)+h(0.5))=0.102$ nats.
- The forward step sees zero gain for each atom ($\gamma_{\rm sub}=0$). Proposition 2 still guarantees that a population forward step without a cap eventually finds a sufficient set, but with a cap it stops short.
- The $D_2$ certificate then fails, **provided the reference classifier can represent interactions**. A linear-logistic reference classifier would certify falsely: this is the $\mathcal{F}$-relativity of Theorem 3 in its clearest form.
- **Design consequences:** (i) the reference learner must model interactions (gradient boosting); (ii) a **stall breaker** in the forward step: when the best gain is below $\gamma_{\rm stall}$ but $\hat G(S)>\varepsilon$, try the best *pair* among the atoms with the highest importance in the reference model.
- Both $X_1$ and $X_2$ are tagged **context**: neither one's law shifts. This is a limit of single-atom tags, not an error. The pair's joint law shifts.

**4. Duplicates and splitting.** See §1.6 and §6.5.
- Exact duplicates break the intersection property and give two minimal sets. Report them as a swap set.
- Complementary pieces are synergistic.
- **Design consequences:** sequential backward elimination only, group reporting, and swap sets.
- NDES inside $S$ also avoids DINES's *redundant-split* recall loss ([D, Prop. 7]): there the whole and its parts explain each other away. A minimal set never contains both the whole and all of its parts, because one of them would be removable.

**5. Coarse mediator.** "Leg drags yes/no" is too coarse: speed reflects the degree of paralysis, so $V$ carries treatment information beyond $Z_P$ and enters NESS. That is correct for NESS: $V$ legitimately carries residual information. NDES labels $V$ direct, which is wrong causally. No method can tell from $(T,Z)$ ([D, Prop. 3]).

**6 vs 7. Hidden physiology vs collider bias.**
- In case 6, lethargy is an unrecorded mediator. Folding it into $V$ makes $V$ a child, so "$V$ direct" is true *relative to the dictionary*.
- In case 7, size is an unrecorded co-parent. Conditioning on paralysis opens $T\to P\leftarrow W\to V$, so $V$ looks direct at every conditioning set inside $S$.
- The two laws can be identical ([D, Prop. 3]).
- **Why NESS is immune and NDES is not.** NESS makes no claim about any edge. Its claim, $T\perp\!\!\!\perp Z_{-S}\mid Z_S$, is true or false of the observed law whatever created the dependence, and collider-induced information is real information about $T$. NDES reads conditional dependences causally, and a missing co-parent makes collider-induced dependence indistinguishable from an edge.

---

## 9. Algorithm

### 9.1 Pseudocode

```
NESS(T, Z; ε, α_s, α_n, K_max, frac_D1 = 0.5, L = 10, γ_stall, ε_x = ε/2, ε_c, clusters = None)
  # Learners: small classifiers = ridge-logistic on value / firing-indicator / pairwise columns (dines._flex_columns);
  #           reference classifier = gradient boosting with the Z_S classifier's logit as offset (nested), clipped to [η, 1-η].
  # Gain(S → S∪B) = cross-fitted log-loss(f_S) - log-loss(f_{S∪B}) on held-out folds  (an estimate of I(T; Z_B | Z_S), nats)

  0. Prefilter (Z only): candidates M = atoms firing in ≥ 1% of units (m ≈ 100 of 9216). Reference R = all atoms
     (or the embedding the SAE was trained on).
  1. Split units (whole clusters) into D1, D2, stratified by T.

  ── on D1 (K-fold cross-fitting) ──
  2. Forward:  S ← ∅
       loop
         U1 ← held-out UCB of Gain(S → R) at level α_s;  if U1 ≤ ε: break
         if |S| = K_max: flag 'capped'; break
         C ← top-L of M\S by linear GCM |stat| given S (dines.linear_gcm)  ∪  top-L by reference-model importance
         g_j ← Gain(S → S∪{j}) for j ∈ C
         if max g_j < γ_stall:                                  # stall breaker (XOR)
             add the best pair {j,k} ⊆ top-L-by-importance by Gain(S → S∪{j,k}); continue
         S ← S ∪ {argmax g_j}
  3. Backward (sequential): repeat
         b_j ← Ĝ(S\{j}) for j ∈ S;  j* ← argmin b_j
         if b_j* ≤ ε: S ← S \ {j*} else break

  ── on D2 (S fixed) ──
  4. Sufficiency: U ← UCB of Δ(S) (nested reference vs Z_S classifier); certified iff U ≤ ε.
     Always report U, κ = Ĝ(S)/Î(T;Z), and the power witness min_j LCB Δ(S\{j}).
  5. Necessity: PCM( T ~ Z_j | Z_{S\{j}} ) for j ∈ S at level α_n/|S|.
  6. Tags: effect if the permutation test of T vs Z_j rejects at α/|S|; context if UCB of I(T;Z_j) ≤ ε_c; else weak.
  7. Swap sets: for j ∈ S and k among the top-L residual carriers: certify Ĝ(S\{j}∪{k}) ≤ ε (Bonferroni over pairs).

  ── informational (selection on D1, all reported numbers estimated on D2) ──
  8. Exclusion: for j ∈ S (and once for the marker group {j : η̂_j ≥ η_0}):
         rerun 2-3 on Z_{-j} from S\{j} at threshold ε_x  →  R_j
         report ι_j, u_j, η_j, and for k ∈ R_j: r_jk, the within-arm dependence test Z_k ⊥ Z_j | T, Z_{S\{j}},
         reading ∈ {carried, masked, mixed, duplicate}
  9. Optional NDES on D2: DINES terminal phase inside S (dines._terminal / _subset_order with the overlap rule),
     with the PCM null of step 5, per-atom level α'/|S|  →  direct / context / undetermined.
```

### 9.2 Choice of the forward score

- The forward score is the cross-fitted held-out log-loss gain. It estimates $I(T;Z_j\mid Z_S)$ in nats, the same units as $\varepsilon$, so it directly measures progress toward the target (Lemma 2).
- It is **scale-free**, unlike NES's raw magnitude ordering [D, §4.1 item 3].
- It is tied to the target, unlike DINES's $p$-value ordering, which mixes effect size with $n$ and noise.
- The order does not affect validity: certification is on $D_2$. It affects only the size reached within the cap (Proposition 3).
- The linear-GCM shortlist is a speed device. Consistency (Theorem 5) needs the shortlist to contain a $\gamma'$-greedy choice; adding the reference model's top-importance atoms makes that likely.

### 9.3 Cost ($m\approx100$ after the prefilter; 9,216 before)

- **Forward:** at most $K_{\max}\approx20$ rounds × $L\approx10$ candidates (or 20 with the importance list) × 5 folds. That is about 1,000–2,000 small ridge-logistic fits, at 10–50 ms each for $n_1=2500$: about a minute. The vectorized linear GCM over 100 atoms per round is negligible.
- **Reference model:** one gradient-boosting fit per fold per set checked (nested offset), at seconds each, so minutes per trial. A reference on all 9,216 atoms is still feasible at $n\le5000$ (sparse input).
- **Backward:** at most $|S|^2$ small fits.
- **Exclusion:** $|S|$ reruns of the forward step, about 10× the forward cost.
- **NDES:** $|S|\,2^{|S|-1}$ tests (5,120 at $|S|=10$), each with small nuisance models. Use the existing subset cap of `dines.py`.
- **Per trial:** minutes on one CPU core. The CelebA benchmark (§11) runs as a Slurm CPU array, never on a login node.

### 9.4 Finite-sample feasibility of the split (heuristic)

- For small gaps, $\mathrm{Var}(d)\approx2G$, so the standard error of $\hat\Delta$ near the boundary is about $\sqrt{2\varepsilon/n_2}$.
- To certify a set with true gap $\varepsilon/2$ at one-sided 95%, we need $\varepsilon/2+1.645\sqrt{\varepsilon/n_2}\le\varepsilon$, i.e. **$\varepsilon\gtrsim11/n_2$ nats**.
  - $N=5000$ ($n_2=2500$): $\varepsilon\gtrsim0.004$;
  - $N=1000$: $\varepsilon\gtrsim0.02$;
  - $N=250$: $\varepsilon\gtrsim0.09$. At this size certification at useful $\varepsilon$ is infeasible: report the bound $U$ only.
- The learning noise of the reference model adds to this; the nested form keeps it small.
- Necessity needs each atom's conditional information well above $z^2_{\alpha/|S|}/(2n_2)$.
- Clustered data: split, fold and bootstrap by cluster, and use a cluster-level Bernstein or $t_{G-1}$ bound ([D, §3.4]).

### 9.5 Reuse of `src/eci/dines.py`

- **Reused as is:**
  - `active_neurons` (prefilter, from `nes.py`);
  - `_make_folds` (cluster-aware folds);
  - `linear_gcm` (forward shortlist);
  - `_flex_columns` and `_fit_propensity` (the $Z_S$ classifiers: ridge-logistic or HGB);
  - `_forward_skipped` (overlap measurement, used for $\eta_j$ and NDES);
  - `_subset_order` and `_terminal` (NDES inside $S$);
  - the cluster-robust SE helpers.
- **New:**
  - the log-loss gain estimator with held-out upper and lower bounds (normal and empirical-Bernstein);
  - the nested reference classifier;
  - the PCM with response $T$;
  - the sufficiency-driven forward step and the sequential backward step;
  - the stall breaker;
  - swap sets;
  - the exclusion driver;
  - the tags.
- **Changed:** NDES uses the PCM conditional-independence null instead of the GCM mean null.

---

## 10. NES reassessment

NES (Mencattini, Cadei, Locatello, arXiv 2510.14073) made the central observation that ECI rests on: marginal tests on an entangled dictionary flag more and more atoms as power grows. The table maps each limitation documented in [D, §4] to how NESS/NDES handles it. Equation numbers refer to the arXiv version.

| NES element | Limitation (details in [D, §4]) | NESS / NDES |
|---|---|---|
| Additive decomposition (Eq. 24) | excludes mediation; the target "coarsest abstraction" is not defined by a testable statement | the target is a conditional-independence statement about $(T,Z)$ (Def. 1). No model of how atoms combine. Any distributional change counts (Lemma 1) |
| Principal alignment / Principal–Max (Eq. 26) | raw-magnitude ordering depends on scale; "the argmax stays principal" is false ([D] counterexample; atom 408 picked before 6051) | NESS needs **no alignment**. Ordering by information in nats (scale-free), and the guarantee does not depend on the order (backward step + $D_2$ certification). Alignment enters only NDES (Theorems 6–7) |
| Recursive stratification and cancellation (Eqs. 27–29) | binarized strata leak about 36% of a mediated effect; termination needs exact cancellation; the transport term is zero in a randomized trial | no strata: continuous conditioning through classifiers. Termination by **measured** residual information $G(S)\le\varepsilon$, not exact cancellation |
| Post-treatment collider associations | stratifying on selected post-treatment atoms compares different frogs in the two arms | NESS is immune: collider information is real information about $T$, and context atoms are part of the target and tagged. NDES inherits [D]'s assumptions and fails as documented (§8, case 7) |
| Adaptive multiple testing, "FWER → 0" | tests at data-chosen strata; the claimed rate is false at fixed $\alpha$ | selection and inference on separate splits. Every $D_2$ test is at a fixed hypothesis. Guarantees $\le\alpha$, exact in finite samples with Bernstein bounds (Theorems 3–4) |
| The ECI paradox | precision falls with $N$ (0.80 → 0.49 at $\tau=0.7$) | a fixed-$\varepsilon$ target does not grow with $n$ (Prop. 4). Test-based methods have an implicit $\varepsilon_n\to0$ |
| Spouse kept (S design: up to 0.8 at $N=5000$) | unlabelled | correct under NESS; tagged **context**; NDES separates it |
| Treatment-revealing atom (mice "bag" neuron) | round 2 stalls ("no overlap") | NESS returns the marker; the exclusion analysis recovers what it masked |

---

## 11. Focused CelebA experiment plan (not run)

**Data and harness.** The existing benchmark: `scripts/benchmark/celeba_dines/`, TopK SAE with 9,216 atoms, designs in `gen_designs.py`, seeds 420–429, $\tau\in\{0,\dots,0.8\}$, $N\in\{1000,2500,5000\}$ plus $N=250$ to show infeasibility. Slurm CPU arrays only. Fixed before any metric: $\varepsilon\in\{0.005,0.01,0.02\}$ nats, $\alpha=0.05$ ($\alpha_s=\alpha_n=0.025$), $K_{\max}=20$, a 50/50 split, and NDES at $\delta\in\{0,0.00125\}$ nats ($\rho\approx0.05$).

**Step 0 (truth known exactly): label dictionary.**
- Use the 40 CelebA attribute labels as atoms, with and without added noise, as a perfectly aligned dictionary. The Markov blanket, children and co-parents are then known exactly.
- This checks the code against Theorems 5–7 before any SAE run.

**Existing designs re-scored at both levels.** Truth at the NESS level = best-F1 atoms of the blanket attributes; truth at the NDES level = atoms of the children.

| design | structure | NESS should | NDES should | measured so far (RESULTS.md) |
|---|---|---|---|---|
| original / I | $T\to$ Hat, $T\to$ Eyeglasses $\leftarrow$ Smiling | 38, 6051, plus Smiling carriers (3278, 1592, ...) as **context** | 38, 6051 direct | DINES δ=0.05 precision 0.87 at $N=5000$, $\tau=0.7$; published NES 0.49 |
| M | $T\to$ Hat $\to$ Eyeglasses | 38, plus leaks (6051 carries what 38 misses) | 38 direct; 6051 context, or direct if it leaks above $\delta$ | DINES selects 6051 in 60% of runs at $N=5000$, $\tau=0.7$ ($\delta=0.05$): "the mediator design still leaks" |
| S | $T\to$ Hat $\leftarrow$ Eyeglasses, Eyeglasses $\to$ Blond $\leftarrow$ Hat | 38 **and 6051 (spouse = context)** | 38 direct; 6051 context | DINES spouse 0/80 (correct for DINES); NES spouse 0.8 at $N=5000$, $\tau=0.7$ (**correct for NESS**) |
| H | as S, spouse = Smiling (no aligned atom) | 38 plus several Smiling carriers as context | 38 direct; Blond (5154) may be falsely direct (separator missing) | DINES 10/10 exact at $\tau=0.7$, $\delta=0.05$ |

**New designs.**
1. **M without the hat atom.** Delete atom 38 from the dictionary. NESS should return 6051 together with hat proxies (8805, 1852). This is also the exclusion analysis of M.
   - Prediction: $R_{38}\ni6051$, read as **carried**. Eyeglasses depends on hat within each arm by design, and 38 has good overlap for $\tau\le0.5$.
   - At $\tau=0.8$ the hat label's arm TV is 0.8 (near-marker), so expect "mixed".
2. **Marker.** $p(\text{Hat}\mid T)=0.97/0.03$, plus an independent effect $T\to$ Eyeglasses with $\tau=0.3$. Check bucket sizes first: there are about 838 hat images.
   - Prediction: NESS $=\{38\}$ or $\{38,6051\}$ depending on $\varepsilon$.
   - Excluding 38 brings 6051 back, read as **masked**: within-arm independence holds by design.
3. **XOR.** Eyeglasses $\sim\mathrm{Bern}(1/2)$ independent of $T$; Hat equals Eyeglasses with probability $(1+\tau)/2$ if treated and $1/2$ if control. Both are marginally independent of $T$.
   - Predictions: (a) without the stall breaker, certification fails with a boosting reference and **falsely succeeds with a linear one** (this demonstrates $\mathcal{F}$-relativity); (b) with the stall breaker, $\{38,6051\}$ is recovered.
4. **Duplicates and splits, artificial.** Append an exact copy of 38, a noisy copy, and a random-mask complementary split of 38 ($UZ_{38}$, $(1-U)Z_{38}$).
   - Predictions: one copy is selected and the other appears in its swap set and its $R$ (duplicate); the complementary pieces show the dilution and under-coverage of §1.6, and the swap set detects them; DINES loses the redundant split ([D, Prop. 7]).
5. **Natural split** (from the splitting diagnostic). A single direct effect $T\to$ Eyeglasses (pair-split, 6051+2218, pair F1 gain +0.082) vs $T\to$ Hat (single atom 38, gain 0.001).
   - Predictions: Hat gives $\{38\}$. Eyeglasses gives $\{6051,2218\}$ at small $\varepsilon$, and $\{6051\}$ with 2218 as top residual carrier or in the swap set at larger $\varepsilon$.
   - Measure: group information $I(T;Z_{6051},Z_{2218})$, swap sets, $R_{6051}$, and whether decoder-cosine grouping (proposal) pairs them.
6. **Certified residual information as a metric.** For every method's output set, compute the $D_2$ upper bound $U$ on $G(S)$ against the same nested reference, and plot it against $|S|$.
   - NESS's forward path traces the information–size frontier.
   - This metric needs no ground truth, so it carries over to real data.

**Baselines.** Published NES (`sneurons_*.json`), repo NES, marginal Bonferroni tests, DINES as is ($\delta=0$ and $0.05$), and IAMB forward only.

**Metrics.** Precision and recall against the NESS-level and NDES-level truths; exact recovery; certification rate; certified residual information $U$; $|S|$; tag accuracy (effect/context); exclusion readings (carried/masked) against the design; runtime.

**Real data, once the new representations are ready.**
- **Frogs** (Sweeney *Xenopus*): genotype contrasts at stages 44–48 and Juv. The running example made literal. Expect colour-like markers.
- **Mice** (phase contrast): the "bag" neuron is a marker. NESS should return it, and the exclusion analysis should show what it masks.
- **Ants** (v2 / ISTAnt): grooming should be a carried or masked atom of the treatment contrast.

On real data the certified residual information and the exclusion map are the primary outputs; there is no ground truth for precision.

---

## 12. Open questions and conjectures

- **Conjecture 1 (joint leak).** For pure atoms with independent noise, $G(B_Z)$ is bounded by a constant times $\sum_{j\notin B_Z}(\text{per-atom leak of }j)$, in the sense of [D, Props. 5–6] converted to information. In general the leaks can be dependent; not proved.
- **O1 (nesting).** Is every element of $\mathrm{NESS}(\varepsilon_2)$ contained in some element of $\mathrm{NESS}(\varepsilon_1)$ for $\varepsilon_1<\varepsilon_2$? Not claimed; the forward path is nested by construction.
- **O2 (uniformity, large $m$).** Theorems 4–5 are pointwise over a finite but very large family ($\sum_{k\le K_{\max}}\binom{m}{k}$ sets).
- **O3 (TopK sparsity and the intersection property).** A TopK SAE activates exactly $k$ atoms per input, so whether $Z_j$ fires is partly determined by how many other atoms fire. Total-conditioning quantities $u_j$ can then be distorted, and the intersection property can fail (non-unique exact targets). Effect on $\varepsilon>0$ targets: unknown.
- **O4 (dictionary fitted on the same data).** As [D, O8].
- **O5 (learner-free certification).** Impossible in general: no conditional-independence test with continuous conditioning has uniform power (Shah & Peters, 2020), so the $\mathcal{F}$-relativity of Theorem 3 cannot be removed, only made explicit (power witness).
- **O6 (groups from $Z$).** Decoder-direction similarity and Matryoshka hierarchies as group builders are untested proposals; so is the vector-valued PCM for groups.
- **O7 (choosing $\varepsilon$).** Absolute vs relative ($\kappa$); a principled default tied to $n_2$ (§9.4) or to the effect size.
- **O8 (k-way synergies).** The pair stall breaker handles pairs only.
- **O9 (exclusion threshold).** A statistical guarantee for $R_j$ at $\varepsilon_x<\varepsilon$; currently informational.
- **O10 (NDES with markers).** Combining the exclusion map with NDES's undetermined label into one output.
- **O11 (clustered data).** Bernstein and normal bounds for cluster-level sums; straightforward but not written.

---

## 13. References

- Bentkus, V., Götze, F. (1996). The Berry–Esseen bound for Student's statistic. *Ann. Probab.* 24(1).
- Cadei, R., et al. (2026). From Tokens to Policy: Causal and Interpretable Heterogeneous Treatment Effects Identification (NEXIS). arXiv 2606.17010.
- Cohen, J. E., Kemperman, J. H. B., Zbăganu, G. (1998). *Comparisons of Stochastic Matrices with Applications in Information Theory, Statistics, Economics and Population Sciences.* Birkhäuser.
- Das, A., Kempe, D. (2011). Submodular meets spectral: greedy algorithms for subset selection, sparse approximation and dictionary selection. *ICML*.
- Kohavi, R., John, G. H. (1997). Wrappers for feature subset selection. *Artif. Intell.* 97.
- Lundborg, A. R., Kim, I., Shah, R. D., Samworth, R. J. (2024). The projected covariance measure for assumption-lean variable significance testing. *Ann. Statist.*
- Maurer, A., Pontil, M. (2009). Empirical Bernstein bounds and sample variance penalization. *COLT*.
- Mencattini, T., Cadei, R., Locatello, F. (2025). Exploratory Causal Inference (NES). arXiv 2510.14073.
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems.* Morgan Kaufmann.
- Peters, J. (2015). On the intersection property of conditional independence and its application to causal discovery. *J. Causal Inference* 3(1).
- Polyanskiy, Y., Wu, Y. (2017). Strong data-processing inequalities for channels and Bayesian networks. In *Convexity and Concentration*, Springer.
- Shah, R. D., Peters, J. (2020). The hardness of conditional independence testing and the generalised covariance measure. *Ann. Statist.* 48(3).
- Tsamardinos, I., Aliferis, C. F., Statnikov, A. (2003). Algorithms for large scale Markov blanket discovery. *FLAIRS*.
- Williamson, B. D., Gilbert, P. B., Simon, N. R., Carone, M. (2023). A general framework for inference on algorithm-agnostic variable importance. *JASA* 118.
- All causal-model references (Lauritzen et al. 1990; Robins & Greenland 1992; Pearl 2001; Imai et al. 2010; Mooij et al. 2020) as in [D, §7].

---

## Appendix A. Index of results

| Label | Statement | Status |
|---|---|---|
| Lemma 1 | binary $T$: sufficiency ⟺ $\pi_B=\pi_A$; detects any distributional change | proved |
| Lemma 2 | $G(S)$ = log-loss gain; chain rule; monotone; proper scoring | proved |
| Lemma 3 | minimality via single removals | proved |
| Lemma 4 | $u_j>\varepsilon$ ⇒ $j$ in every $\varepsilon$-sufficient set | proved |
| Proposition 1 | uniqueness of the exact target under intersection; $=\{u_j>0\}$ | proved |
| Theorem 1 | exact sufficiency: $P(Z\mid do(t))=P(Z_S\mid do(t))K$ | proved |
| Theorem 2 | R1 bound $\mathrm{osc}(g)\sqrt{\varepsilon/(2e(1-e))}$ (corrects the proposed constant) | proved; measured tight |
| Lemma 5 | the loss difference has variance $\ge\eta'\varepsilon^2$ when the gap is $\ge\varepsilon$ | proved |
| Theorem 3 | sufficiency certificate ($\mathcal{F}$-relative; finite-sample and asymptotic; population link) | proved |
| Theorem 4 | necessity certificate (PCM, Bonferroni, finite family) | proved |
| Proposition 2 | the population forward–backward algorithm returns an element of NESS($\varepsilon$) | proved |
| Theorem 5 | consistency of the selection under margin + greedy completeness | proved |
| Proposition 3 | submodularity ratio ⇒ greedy completeness, explicit cap | proved |
| Proposition 4 | ECI paradox: fixed target; replicate example | proved; numbers measured |
| Proposition 5 | $G(S\setminus j)=\iota_j+u_j$ | proved |
| Proposition 6 | exact exclusion set $=\mathrm{MB}(T;Z_{-j})\setminus S$ | proved |
| Proposition 7 | masking bound $r_{jk}\ge1-\eta_j$ | proved; measured tight |
| Lemma 8 | under faithfulness, screening ⇒ within-arm dependence | proved |
| Theorem 6 | NDES exact: NESS(0) $=B_Z$; check returns $J_C$; separators inside $S$ | proved (uses [D] Lemma 3, Thm. 1, Prop. 1) |
| Corollary 1 | NDES robust to supersets of $B_Z$ | proved |
| Theorem 7 | $\varepsilon/\delta$ window | proved (conditional on W1–W4) |
| Conjecture 1 | joint leak bound | conjecture |
| §1.5, §5 paradox paragraph, §9.4 | $I\approx\rho^2/2$; implicit $\varepsilon_n$; feasible $\varepsilon\gtrsim11/n_2$ | heuristic |
| §8 | counterexamples | proved by construction or computation |

## Appendix B. Measurements (CPU, each under one minute; scripts in the session scratchpad, not committed)

- **B.1 R1 bound.** 20,000 random discrete laws ($|Z_S|\le3$, $|Z_{-S}|\le4$, $e\in(0.05,0.95)$, Dirichlet concentrations 0.2/1/5). The supremum over $\mathrm{osc}(g)\le1$ is $\tfrac12\sum_z|w(z)|$. Worst ratio to $\mathrm{osc}\sqrt{\varepsilon/(2e(1-e))}$: **1.000**. Worst ratio to the proposed bound (centred $g$): 0.827.
- **B.2 Information vs $\rho^2/2$** (Gaussian shift $\theta$, $e=\tfrac12$, numerical integration):
  - $\theta=0.05$: $I=0.00031$, $\rho^2/2=0.00031$;
  - $\theta=0.1$: $I=0.00125$ (ratio 1.001);
  - $\theta=0.2$: 1.005;
  - $\theta=0.3$: 1.011;
  - $\theta=0.5$: 1.031;
  - $\theta=1$: 1.114.
- **B.3 Frog numbers.**
  - Context table of [D, Prop. 1]: $I(T;P)=0.2569$, $I(T;P,W)=0.2862$, $I(T;W\mid P)=0.0293$ nats.
  - Marker 0.99/0.01: $H(T\mid C)=0.0560$ nats.
  - XOR: $h(0.7)-\frac12(h(0.9)+h(0.5))=0.102$ nats (closed form).
- **B.4 Replicate atoms** ($\beta=1$, $\sigma=2$, $m=50$): $I(T;Z)=0.1040$ nats; $u_k=1.41\times10^{-4}$; minimal size 50/44/29/20/11 at $\varepsilon=0/0.001/0.005/0.01/0.02$.
- **B.5 Complementary split** ($Z_1=UF$, $Z_2=(1-U)F$; $I(T;F)$; $I(T;Z_1)$, $I(T;Z_2)$; $I(T;Z_1\mid Z_2)$, $I(T;Z_2\mid Z_1)$):
  - $\tau=0.5$, $q=0.5$: 0.1308; 0.0432, 0.0432; 0.0876, 0.0876;
  - $\tau=0.3$, $q=0.5$: 0.0457; 0.0152, 0.0152; 0.0305, 0.0305;
  - $\tau=0.5$, $q=0.2$: 0.1308; 0.0145, 0.0863; 0.0445, 0.1163.
- **B.6 Masking bound.** 50,000 random discrete laws with $Z_k\perp\!\!\!\perp Z_j\mid T$. The minimum of $[I(T;Z_k\mid Z_j)/I(T;Z_k)]/(1-\mathrm{TV}_j)$ is **1.0000**, with no violations. A first run reported a violation that was a $\log0$ artefact; it was fixed by using $0\log0=0$.
- **B.7 CelebA splitting diagnostic** (job 862381, `results/eci/ness_splitting/summary.md`, `splitting_diagnostic.csv`): the numbers quoted in §1.6.
