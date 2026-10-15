# DINES — Direct–Indirect Neural Effect Search: theory report

*Draft for review (R. Cadei; adversarial check by Codex). 2026-10-15. Working name; may change.*

**How results are labelled.** *Theorem*, *Proposition*, *Lemma* and *Corollary* come with a complete proof in this document. *Conjecture* = we believe it but have no proof. *Open question* = we do not know. *Hypothesis (empirical)* = a proposed explanation of a measured behaviour that has not been tested. Numbers marked "measured" come from runs listed in §9.4 and Appendix B.

---

## 1. Summary

**Problem.** A binary treatment $T$ is randomized. After treatment we observe a complex measurement $X$ (an image, a video frame) and summarize it with a fixed dictionary of $m$ atoms $Z=\psi(X)\in\mathbb{R}^m$ (for example sparse-autoencoder codes on foundation-model features). We want to know *what the treatment changed*, read off as atoms.

**Target.** The main output is the **direct set**
$$\mathrm{Adj}_Z(T)=\{\,j\in[m]:\ \mathbb{E}[Z^j\mid T,Z^A]\neq\mathbb{E}[Z^j\mid Z^A]\ \text{for every } A\subseteq[m]\setminus\{j\}\,\}:$$
the atoms whose mean shift under $T$ cannot be explained away by conditioning on any other atoms. This is a purely statistical object, defined for any dictionary (Tier 1, §3.1). Under a latent-factor model where the treatment changes some factors and the dictionary aligns with the factors around $T$ (Tier 2, §3.2), the direct set is exactly the set of principal atoms of the **children of $T$** (Theorem 6), which are the factors whose causal mechanism the treatment changes (Proposition 8, Tier 3). It is *not* the smallest set of atoms that makes $T$ independent of the rest (the Markov boundary): that set also contains the unaffected co-parents of affected factors ("spouses"). The hat/hair example in §3.4 shows the difference. The second output is the **indirect set**: atoms that $T$ shifts on average but that are not direct, each tagged with the direct atoms that carry its effect (§5).

**Algorithm.** A forward phase adds, one at a time, the atom with the smallest conditional $p$-value given the atoms already selected (not the largest raw effect). A terminal backward phase keeps an atom only if it stays significant at level $\alpha/m$ given *every* subset of the other selected atoms. An optional third stage builds the indirect output.

**Main theorem (Theorem 11, one sentence).** At fixed $m$, if the tests are valid at a fixed true null and consistent against fixed alternatives, and a fixed "separator pool" of atoms exists that the forward phase is forced to reach, then $\liminf_n \mathbb{P}(\hat S_n=\mathrm{Adj}_Z(T))\ge 1-\alpha$. Under Tier 2 the pool is the set of principal atoms of $T$'s Markov blanket, so $\liminf_n\mathbb{P}(\hat S_n=\{\text{principal atoms of the children of }T\})\ge 1-\alpha$ (Corollary 12).

**The separating-set question (settled).** NEXIS's precision proof needs, for every false atom, a *fixed* true null whose conditioning set lies inside the forward output. With a post-treatment dictionary the target set itself is not such a set (conditioning on a child opens the path from $T$ to its spouses). We prove (Lemma 5) that because $T$ is a root, every factor $s$ that is not a child of $T$ is d-separated from $T$ by $\mathrm{MB}(T)\setminus(\{s\}\cup\mathrm{De}(s))$ — the Markov blanket of $T$ minus $s$ and its descendants. This covers the hard case (a spouse that is also a descendant of $T$, e.g. $T\to A\to B\to W\to C\leftarrow T$, separated by $\{A\}$). So the NEXIS structure "one fixed true null per false atom, no post-selection" survives unchanged.

**What changed from NES** (details and evidence in §8): a defined target instead of "the coarsest abstraction"; ordering by $p$-value instead of raw magnitude (NES's "Principal–Max" property is not preserved across rounds even under its own assumptions — counterexample in §8.2 — and in the saved CelebA runs a mixed atom, N_408, is picked before the eyeglasses atom N_6051 because its raw effect is larger although its $t$-statistic is half as big); a terminal backward step (NES never removes an atom); regression-based conditional tests instead of strata built from atoms binarized at $>0$ (which leave about 36% of a mediated effect in place, measured); the correct error statement $\liminf\mathbb{P}(\text{exact})\ge 1-\alpha$ instead of "FWER $\to0$"; mediated and spouse structure handled explicitly; cluster-robust inference for video.

**Sanity simulation (measured, linear-Gaussian toy with a mediator, a descendant spouse and a pre-treatment spouse, 100 seeds, Appendix B).** With exactly aligned atoms DINES recovers the direct set exactly in 99/100 runs at $n=1000$, $4000$ and $16000$. Its forward phase contains on average 1.9–2.1 non-direct atoms (the two spouses), which the terminal step removes. With noisy principal atoms (approximate alignment) exact recovery falls with $n$: 96, 96, 84 of 100 — the predicted failure mode, which only a relevance margin $\delta$ removes (100/100 at $n\ge4000$ with $\delta=0.05$, but recall 0.85 at $n=1000$).

---

## 2. Setup and notation

- **Units.** $n$ i.i.d. units (for video: i.i.d. clusters, §7.6). Each unit has a treatment $T\in\{0,1\}$ with $\mathbb{P}(T=1)=e\in(0,1)$ fixed by design, optional pre-treatment covariates $W$ (so $T\perp\!\!\!\perp W$), and a post-treatment measurement $X$.
- **Dictionary.** $\psi:\mathcal{X}\to\mathbb{R}^m$ is fixed before the experiment is analysed (not fitted on the test units; §7.9). $Z=\psi(X)$, $Z^j$ its $j$-th atom, $[m]=\{1,\dots,m\}$, $Z^A=(Z^k)_{k\in A}$, $Z^{\emptyset}$ a constant. We assume $\mathbb{E}|Z^j|<\infty$ for all $j$ (second moments where tests need them).
- **Conditional quantities.** For $A\subseteq[m]$ and $j\notin A$:
  - propensity given atoms: $\pi_A(Z^A):=\mathbb{P}(T=1\mid Z^A)$;
  - atom regression: $\mu_{j,A}(Z^A):=\mathbb{E}[Z^j\mid Z^A]$;
  - arm contrast: $\Delta_{j,A}(Z^A):=\mathbb{E}[Z^j\mid T=1,Z^A]-\mathbb{E}[Z^j\mid T=0,Z^A]$ on $\{0<\pi_A<1\}$, and $0$ elsewhere.
- **The null hypothesis used everywhere (mean null).**
$$H_0(j\mid A):\quad \mathbb{E}[Z^j\mid T,Z^A]=\mathbb{E}[Z^j\mid Z^A]\quad\text{a.s.}$$
  When covariates $W$ are used, $Z^A$ is replaced by $(Z^A,W)$ throughout. $p_j(A)$ denotes a $p$-value for $H_0(j\mid A)$.
- **Graph terms** (for a DAG $\mathcal{G}$): $\mathrm{Pa}(v)$ parents, $\mathrm{Ch}(v)$ children, $\mathrm{An}(v)$ ancestors (including $v$), $\mathrm{De}(v)$ *strict* descendants, $\mathrm{Sp}(T):=\bigcup_{c\in\mathrm{Ch}(T)}\mathrm{Pa}(c)\setminus\{T\}$ the co-parents of $T$'s children ("spouses"), $\mathrm{MB}(T):=\mathrm{Ch}(T)\cup\mathrm{Sp}(T)$ the Markov blanket of a root $T$. $\perp_d$ is d-separation.

**Lemma 1 (forms of the mean null).** Fix $A$ and $j\notin A$.
1. $\mathbb{E}[Z^j\mid T,Z^A]-\mu_{j,A}(Z^A)=(T-\pi_A)\,\Delta_{j,A}$ a.s.
2. $H_0(j\mid A)$ holds iff $\pi_A(1-\pi_A)\Delta_{j,A}=0$ a.s., i.e. iff the arms have the same conditional mean wherever both arms have positive conditional probability.
3. If $\mathbb{E}[(Z^j)^2]<\infty$: $\ \mathbb{E}\big[(T-\pi_A)(Z^j-\mu_{j,A})\big]=\mathbb{E}\big[\pi_A(1-\pi_A)\Delta_{j,A}\big]$.
4. $T\perp\!\!\!\perp Z^j\mid Z^A$ implies $H_0(j\mid A)$; the converse is false.

*Proof.* (1) On $\{0<\pi_A<1\}$, write $m_t=\mathbb{E}[Z^j\mid T=t,Z^A]$. Then $\mathbb{E}[Z^j\mid T,Z^A]=Tm_1+(1-T)m_0$ and $\mu_{j,A}=\pi_Am_1+(1-\pi_A)m_0$; the difference is $(T-\pi_A)(m_1-m_0)$. On $\{\pi_A=1\}$ we have $T=1$ a.s., so $\mathbb{E}[Z^j\mid T,Z^A]=\mathbb{E}[Z^j\mid Z^A]$ a.s. there, matching $\Delta=0$; likewise on $\{\pi_A=0\}$. (2) Given $Z^A$ on $\{0<\pi_A<1\}$, $T-\pi_A$ takes the non-zero values $1-\pi_A$ and $-\pi_A$ with positive conditional probabilities, so $(T-\pi_A)\Delta_{j,A}=0$ a.s. iff $\Delta_{j,A}=0$ a.s. on that set, iff $\pi_A(1-\pi_A)\Delta_{j,A}=0$ a.s. (3) $\mathbb{E}[(T-\pi_A)\mu_{j,A}]=\mathbb{E}[\mu_{j,A}\,\mathbb{E}[T-\pi_A\mid Z^A]]=0$, and by the tower rule and (1), $\mathbb{E}[(T-\pi_A)Z^j]=\mathbb{E}[(T-\pi_A)(\mu_{j,A}+(T-\pi_A)\Delta_{j,A})]=\mathbb{E}[\pi_A(1-\pi_A)\Delta_{j,A}]$. (4) Conditional independence gives equality of the conditional laws, hence of the conditional means. For the converse, let $Z^j\mid T\sim\mathcal{N}(0,1+T)$ with $A=\emptyset$. $\square$

Item 3 is the quantity a residual-product (GCM) test estimates: an *overlap-weighted average* of the conditional arm contrast, analogous to the overlap-weighted average treatment effect. Item 4 says the mean null is weaker than conditional independence. Since $T$ is binary, conditional independence is itself a mean statement *about $T$*: $T\perp\!\!\!\perp Z^j\mid Z^A\iff \mathbb{E}[T\mid Z^j,Z^A]=\mathbb{E}[T\mid Z^A]$. We use the mean null *on the atom* $Z^j$, as decided: the effect type of interest is a change in the mean of an atom. For a zero-inflated sparse-autoencoder code, the mean $\mathbb{E}[Z^j]=\mathbb{P}(Z^j>0)\,\mathbb{E}[Z^j\mid Z^j>0]$ moves when either the firing rate or the firing magnitude moves, so a mean test covers both. A treatment that changes only the spread of an atom is invisible to this target.

---

## 3. Targets

### 3.1 Tier 1 — the direct set (statistical, any dictionary)

**Definition 1 (direct set; total-effect set).**
$$\mathrm{Adj}_Z(T):=\{\,j\in[m]:\ H_0(j\mid A)\ \text{is false for every } A\subseteq[m]\setminus\{j\}\,\},\qquad \mathrm{De}_Z(T):=\{\,j:\ H_0(j\mid\emptyset)\ \text{is false}\,\}.$$
$\mathrm{De}_Z(T)$ is the set of atoms with a non-zero average treatment effect. Taking $A=\emptyset$ shows $\mathrm{Adj}_Z(T)\subseteq\mathrm{De}_Z(T)$.

*Reading.* An atom is direct if *no* set of other atoms accounts for its mean shift. "Direct" is relative to the dictionary, exactly as adjacency in causal discovery is relative to the measured variables: if a mediator is missing from the dictionary, its downstream atoms become direct (§4.3, counterexample (b)). With covariates, every conditioning set also contains $W$; Proposition 14 shows this does not change the target under Tier 2.

*Why the mean null, and what the tests see.* The target uses the mean null (decision of the user; §2). Two test families are relevant (§7.2):
- the generalised covariance measure (GCM) estimates $\mathbb{E}[\pi_A(1-\pi_A)\Delta_{j,A}]$ (Lemma 1.3). It is valid under the mean null but has power only when this overlap-weighted average is non-zero, so it misses alternatives where the conditional contrast changes sign across values of $Z^A$;
- the projected covariance measure (PCM) with response $Z^j$, covariate $T$ and conditioning set $Z^A$ tests the mean null itself and has power against any alternative its learned projection captures (it estimates $\mathbb{E}[\pi_A(1-\pi_A)\,h\,\Delta_{j,A}]$ for a learned $h\approx\Delta_{j,A}$).

The theorems below are stated for a generic test (Assumption A5); the faithfulness-type assumptions must then hold *in the sense the chosen test can see*. For the GCM this means "the overlap-weighted contrast is non-zero" wherever a hypothesis is assumed false.

### 3.2 Tier 2 — the causal model

**Assumption A1 (latent structural model).** There is a finite set of latent variables $L=(L_1,\dots,L_r)$ ("factors"; they may include variables no atom represents) and an acyclic structural causal model over $\{T\}\cup L$ with mutually independent exogenous noises, in which $T$ is exogenous (a root: randomization). Call its DAG $\mathcal{G}_L$. The atoms are generated by a measurement map
$$Z=H(L,N^Z),\qquad N^Z\perp\!\!\!\perp(T,L),$$
where the measurement noise $N^Z$ may be arbitrarily dependent across atoms. In particular $T$ acts on $X$, hence on $Z$, only through the factors (a visible treatment marker is itself a factor, a child of $T$). We write $P(j)\subseteq L$ for a minimal set of factors on which $H_j$ depends.

**Assumption A2 (alignment on the Markov blanket of $T$).** For every factor $k\in\mathrm{MB}(T)$ (every child of $T$ and every co-parent of a child) there is an atom $j_k$, distinct for distinct $k$, such that
- (a) *determination*: $L_k=a_k(Z^{j_k})$ a.s. for a measurable $a_k$;
- (b) *purity*: $Z^{j_k}=h_k(L_k,N^Z)$, i.e. the atom depends on the latent world only through its own factor.

Write $J_C:=\{j_c:c\in\mathrm{Ch}(T)\}$ (principal atoms of the children), $J_{Sp}:=\{j_s:s\in\mathrm{Sp}(T)\setminus\mathrm{Ch}(T)\}$, and $B_Z:=J_C\cup J_{Sp}$.

**Assumption A3 (mean adjacency faithfulness at the children).** For every $c\in\mathrm{Ch}(T)$ and every $A\subseteq[m]\setminus\{j_c\}$, $H_0(j_c\mid A)$ is false.

*Relation to the framing given in the brief.* The brief asked for (i) alignment that "nearly determines" each affected factor and each co-parent of an affected factor, (ii) no hidden common cause among affected factors outside the dictionary, and (iii) factor-level faithfulness. Three differences:

1. **Less alignment is needed.** Only the Markov blanket of $T$ (children and their co-parents) must be aligned. Descendants further down, and hidden common causes of two *non-child* affected factors, do no harm (they are handled by the proof of Theorem 6). The brief's (i)+(ii) imply A2 but are stronger.
2. **(ii) is the special case that matters.** A hidden common cause of a child and another affected factor is a co-parent of that child, so A2 forbids it; this is exactly the counterexample (a) in §4.3. A hidden common cause of two grandchildren is allowed.
3. **Factor-level faithfulness is not enough.** A3 is a statement about atoms. It fails for a faithful factor model whenever another atom duplicates a child's principal atom (sparse-autoencoder feature splitting or absorption), or whenever some atoms almost reveal $T$ (§4.3 (e)). We therefore state the atom-level condition, as NEXIS does with its Eq. 12.
4. **"Nearly determines" is not enough for the theorem.** Determination must be exact; a noisy proxy does not block a mediated path (§4.3 (c)). Approximate alignment is treated in §7.8.
5. **Purity is new.** NEXIS's Principal Alignment allows polysemantic principal coordinates. Here the principal atoms of spouses must be pure (§4.3 (d)), and so must the blanket atoms used to separate them (the set $D^\star(s)$ of Lemma 5, which can contain children): an impure child atom that also reads a common child of $T$ and $s$ re-opens the collider it was meant to leave closed. Purity is not needed for Lemma 3.

### 3.3 Tier 3 — the mechanisms that the treatment changes

Under A1, write the mechanism of factor $k$ as the conditional law $q_k(\,\cdot\mid \mathrm{pa}_k)$ of $L_k$ given its parents. An intervention sets $T=t$:
$$\mathbb{P}(L\mid do(T=t))=\prod_{k}q_k\big(L_k\mid L_{\mathrm{Pa}(k)\setminus\{T\}},\,t\big),$$
where $t$ enters only the factors with $T\in\mathrm{Pa}(k)$. Define the **shifted mechanisms** $\mathrm{Shift}(T):=\{k:\ q_k(\cdot\mid \ell,1)\neq q_k(\cdot\mid \ell,0)$ for a set of parent values $\ell$ of positive probability$\}$.

**Proposition 8 (the direct set is the minimal set of changed mechanisms).** Under A1:
1. $\mathrm{Shift}(T)\subseteq\mathrm{Ch}(T)$, and
$$\mathbb{P}(L\mid do(T=t))=\prod_{k\in\mathrm{Shift}(T)}q_k(L_k\mid \mathrm{pa}_k,t)\prod_{k\notin\mathrm{Shift}(T)}q_k(L_k\mid\mathrm{pa}_k).$$
2. For $K\subseteq[r]$: every mechanism outside $K$ is the same in both arms iff $\mathrm{Shift}(T)\subseteq K$. So $\mathrm{Shift}(T)$ is the unique smallest set of mechanisms whose change explains the intervention.
3. If the edges out of $T$ are causally minimal (each mechanism with $T$ as a parent really uses it), $\mathrm{Shift}(T)=\mathrm{Ch}(T)$. With Theorem 6, $\mathrm{Adj}_Z(T)=\{j_k:k\in\mathrm{Shift}(T)\}$.

*Proof.* (1) A mechanism without $T$ as a parent cannot depend on $t$, so $\mathrm{Shift}(T)\subseteq\mathrm{Ch}(T)$; for $k\in\mathrm{Ch}(T)\setminus\mathrm{Shift}(T)$ the two kernels agree almost everywhere, so dropping $t$ from them changes no factor of the product. (2) "Only if": a $k\in\mathrm{Shift}(T)\setminus K$ is, by definition, a mechanism outside $K$ that differs between arms. "If": mechanisms outside $K\supseteq\mathrm{Shift}(T)$ do not depend on $t$ by definition. (3) Causal minimality of the edge $T\to k$ means $q_k$ is not a.e. constant in $t$, i.e. $k\in\mathrm{Shift}(T)$. $\square$

This is the *sparse mechanism shift* view of an intervention (Schölkopf et al., 2021; Perry et al., 2022). In the language of Joint Causal Inference (Mooij et al., 2020) $T$ is a context variable and its children are the intervention targets; DINES estimates the targets of a known intervention when the target variables are latent and seen only through a dictionary.

*Mean caveat.* Theorem 6 uses A3, a mean statement. A child whose mechanism changes only the spread of its atom is in $\mathrm{Shift}(T)$ but violates A3 and is not recovered.

### 3.4 Why this is not the Markov boundary: the hat/hair example

The smallest set $S$ with $T\perp\!\!\!\perp Z^{[m]\setminus S}\mid Z^S$ (the Markov boundary of $T$ in the dictionary) is the natural target of a pre-treatment search such as NEXIS. With a post-treatment dictionary it equals children *plus spouses* (Proposition 7), while the mechanisms changed are the children only. The two coincide exactly when every co-parent of a child of $T$ is itself a child (in particular when children have no other factor parents).

```
        T  (hat given)                    L (long hair, pre-treatment)
         \                                  /
          \                                /
           v                              v
              V  (hair visible in the photo)
```

Let $L\sim\mathrm{Bern}(1/2)$ and $\mathbb{P}(V=1\mid T,L)=0.9$ if $T=0$ or $L=1$ (long hair sticks out under the hat), and $0.1$ otherwise. Atoms: $Z^V=V$, $Z^L=L$.
- The treatment changes one mechanism, $q_V$. The direct set is $\{Z^V\}$.
- $T\perp\!\!\!\perp L$ (pre-treatment), so $Z^L$ has no average effect.
- But given $V=1$: $\mathbb{E}[L\mid T=0,V=1]=1/2$ while $\mathbb{E}[L\mid T=1,V=1]=0.45/(0.45+0.05)=0.9$. Conditioning on the child opens $T\to V\leftarrow L$, so $Z^L$ is needed to make $T$ independent of the rest: the Markov boundary is $\{Z^V,Z^L\}$.

The same structure is in NES's own CelebA design (Appendix D of the NES paper): $T\to\text{Eyeglasses}\leftarrow\text{Smiling}$, $T\to\text{Hat}$. With ATE $=0.7$ the design gives $\mathbb{E}[\text{Smiling}\mid T=1,\text{Eyeglasses}=1]=0.85/1.75=0.486$ versus $0.15/0.35=0.429$ under control: a small but real collider dependence. Smiling is in the Markov boundary but not in the direct set.

**Proposition 7 (Markov boundary under Tier 2).** Under A1–A2, $B_Z$ is a Markov blanket of $T$ in the dictionary: $T\perp\!\!\!\perp Z^{[m]\setminus B_Z}\mid Z^{B_Z}$. If moreover $T\not\perp\!\!\!\perp Z^j\mid Z^{[m]\setminus\{j\}}$ for every $j\in B_Z$, then every Markov blanket contains $B_Z$, so $B_Z$ is the unique Markov boundary.

*Proof.* First claim: Lemma 3 below with $Y=Z^{[m]\setminus B_Z}$. Second: let $S$ be a Markov blanket and $j\in B_Z\setminus S$. Then $T\perp\!\!\!\perp Z^{[m]\setminus S}\mid Z^S$ and weak union give $T\perp\!\!\!\perp Z^j\mid Z^{[m]\setminus\{j\}}$, a contradiction. $\square$

---

## 4. Identification (Tier 2)

### 4.1 Two lemmas

**Lemma 2 (conditioning on an atom that determines a factor).** Let $T,Y,G,M$ be random elements with $T\perp\!\!\!\perp (Y,M)\mid G$. Let $\tilde Z=\varphi(G,M)$ for a measurable $\varphi$, and suppose $G=a(\tilde Z)$ a.s. Then $T\perp\!\!\!\perp Y\mid \tilde Z$.

*Proof.* $(Y,\tilde Z)$ is a measurable function of $(Y,M,G)$, so $T\perp\!\!\!\perp (Y,\tilde Z)\mid G$. Weak union gives $T\perp\!\!\!\perp Y\mid (G,\tilde Z)$. Since $G=a(\tilde Z)$ a.s., $\sigma(G,\tilde Z)=\sigma(\tilde Z)$ up to null sets, so $T\perp\!\!\!\perp Y\mid\tilde Z$. $\square$

The lemma is where *determination* enters: conditioning on a principal atom is as good as conditioning on its factor, plus independent measurement noise that does not matter.

**Lemma 3 (the aligned blanket screens off everything).** Under A1–A2, for any measurable function $Y$ of $(L,N^Z)$ (for example any collection of atoms outside $B_Z$), $T\perp\!\!\!\perp Y\mid Z^{B_Z}$.

*Proof.* In $\mathcal{G}_L$, every path from the root $T$ starts with an edge $T\to c$, $c\in\mathrm{Ch}(T)\subseteq\mathrm{MB}(T)$. If $c$ is a non-collider on the path it is blocked by conditioning on $L_{\mathrm{MB}}$. If $c$ is a collider, the next vertex $x$ is a parent of $c$ other than $T$, so $x\in\mathrm{Sp}(T)\subseteq\mathrm{MB}(T)$, and $x$ is a non-collider (the edge $x\to c$ leaves $x$), so it is blocked. Hence $T\perp_d L\setminus(\mathrm{MB}\cup\{T\})\mid \mathrm{MB}$, and by the global Markov property of an SCM with independent noises, $T\perp\!\!\!\perp L_{\mathrm{rest}}\mid L_{\mathrm{MB}}$. Since $N^Z\perp\!\!\!\perp(T,L)$, we have $T\perp\!\!\!\perp N^Z\mid (L_{\mathrm{rest}},L_{\mathrm{MB}})$, and contraction gives $T\perp\!\!\!\perp (L_{\mathrm{rest}},N^Z)\mid L_{\mathrm{MB}}$. Now apply Lemma 2 with $G=L_{\mathrm{MB}}$, $M=(L_{\mathrm{rest}},N^Z)$, $\tilde Z=Z^{B_Z}$ (a function of $(L,N^Z)$ by A1, which determines $G$ by A2(a)). $\square$

Purity is not used in Lemma 3. It is used in the next one.

**Lemma 4 (atoms of the separating set).** Under A1–A2, let $D\subseteq\mathrm{MB}(T)$ and $V\subseteq L$ with $T\perp_d V\mid D$ in $\mathcal{G}_L$. Then for every $Y=f(L_V,N^Z)$, $T\perp\!\!\!\perp Y\mid Z^{J_D}$ where $J_D=\{j_k:k\in D\}$.

*Proof.* The Markov property gives $T\perp\!\!\!\perp L_V\mid L_D$; with $N^Z\perp\!\!\!\perp(T,L)$ and contraction, $T\perp\!\!\!\perp(L_V,N^Z)\mid L_D$. By purity A2(b), $Z^{J_D}=\varphi(L_D,N^Z)$; by determination A2(a), $L_D$ is a function of $Z^{J_D}$. Lemma 2 with $G=L_D$, $M=(L_V,N^Z)$ concludes. $\square$

### 4.2 The separator lemma and the identification theorem

**Lemma 5 (a separator inside the blanket of a root).** Let $\mathcal{G}$ be a DAG in which $T$ has no parents, and let $s\notin\mathrm{Ch}(T)\cup\{T\}$. Put
$$D^\star(s):=\mathrm{MB}(T)\setminus\big(\{s\}\cup\mathrm{De}(s)\big).$$
Then $T\perp_d s\mid D^\star(s)$.

*Proof.* We use the moralization criterion (Lauritzen et al., 1990): $T\perp_d s\mid D$ iff $D$ separates $T$ from $s$ in the moral graph $(\mathcal{G}_{\mathcal{A}})^m$ of the ancestral set $\mathcal{A}:=\mathrm{An}(\{T,s\}\cup D)$. Take $D=D^\star:=D^\star(s)$.

*Step 1: $\mathcal{A}$ contains no strict descendant of $s$.* Let $x\in\mathrm{De}(s)\cap\mathcal{A}$. $x\in\mathrm{An}(T)=\{T\}$ is impossible ($x=T$ would make $s$ an ancestor of the root $T$). $x\in\mathrm{An}(s)$ would create a cycle. $x\in\mathrm{An}(d)$ for some $d\in D^\star$ would make $d=x$ or $d\in\mathrm{De}(x)\subseteq\mathrm{De}(s)$, and both contradict $d\in D^\star$.

*Step 2: every neighbour of $T$ in $(\mathcal{G}_{\mathcal{A}})^m$ lies in $D^\star$.* $T$ has no parents, so its neighbours are (i) its children inside $\mathcal{A}$ and (ii) co-parents $p$ of a child $c\in\mathcal{A}$ (moral edges). (i) A child $c\in\mathcal{A}$ is not $s$ ($s\notin\mathrm{Ch}(T)$) and not in $\mathrm{De}(s)$ (Step 1), and $c\in\mathrm{MB}(T)$; so $c\in D^\star$. (ii) $p\in\mathcal{A}$ because $\mathcal{A}$ is ancestral; $p\neq s$, since otherwise $c\in\mathrm{Ch}(s)\subseteq\mathrm{De}(s)$, contradicting Step 1; $p\notin\mathrm{De}(s)$ by Step 1; and $p\in\mathrm{Sp}(T)\subseteq\mathrm{MB}(T)$. So $p\in D^\star$.

*Step 3.* Every path from $T$ to $s$ in the moral graph starts with an edge to a neighbour of $T$, which lies in $D^\star$; and $s\notin D^\star$. So $D^\star$ separates $T$ from $s$. $\square$

*Example.* $T\to A\to B\to W\to C\leftarrow T$. $\mathrm{MB}(T)=\{A,C,W\}$; $W$ is a spouse and a descendant of $T$. $D^\star(W)=\{A,C,W\}\setminus(\{W\}\cup\{C\})=\{A\}$, and indeed $\{A\}$ blocks $T\to A\to B\to W$ while $T\to C\leftarrow W$ stays blocked because $C$ is not conditioned on. $\{A,C\}$ does *not* separate. Measured on a linear-Gaussian version ($n=20{,}000$, GCM test, Appendix B): $p=0.14$ given $\{A\}$, $p=0.15$ given $\{A,H\}$, $p=0.37$ given $\{A,B\}$, but $p=3.6\times10^{-70}$ given $\emptyset$ and $p=2.2\times10^{-56}$ given $\{A,C\}$.

*Remark (the root matters).* The proof uses only that $T$ has no parents. For a non-root target the statement fails as written: in $X\to P\to T\to C\leftarrow X$ the vertex $X$ is a spouse, $D^\star(X)$ excludes $P\in\mathrm{De}(X)$, yet $P$ is needed to block $X\to P\to T$. This is the situation behind the known incompleteness of MMPC-type rules (Peña et al., 2007); randomization removes it.

**Theorem 6 (identification of the direct set).** Under A1, A2 and A3,
$$\mathrm{Adj}_Z(T)=J_C=\{\,j_c:\ c\in\mathrm{Ch}(T)\,\}.$$
Moreover, every $j\notin J_C$ has a fixed separating set $A_j\subseteq B_Z\setminus\{j\}$ with $T\perp\!\!\!\perp Z^j\mid Z^{A_j}$ (hence $H_0(j\mid A_j)$ true):
$$A_j=B_Z\ \text{ if } j\notin B_Z,\qquad A_j=J_{D^\star(s)}\ \text{ if } j=j_s,\ s\in\mathrm{Sp}(T)\setminus\mathrm{Ch}(T).$$

*Proof.* "$\supseteq$" is A3. For "$\subseteq$" and the second claim, take $j\notin J_C$.
- If $j\notin B_Z$: $Z^j$ is a function of $(L,N^Z)$, so Lemma 3 gives $T\perp\!\!\!\perp Z^j\mid Z^{B_Z}$, and $j\notin B_Z=A_j$.
- If $j=j_s$ with $s$ a spouse that is not a child: by Lemma 5 (in $\mathcal{G}_L$, where $T$ is a root and $s\notin\mathrm{Ch}(T)$), $T\perp_d s\mid D^\star(s)$, with $D^\star(s)\subseteq\mathrm{MB}(T)$. By purity, $Z^{j_s}=h_s(L_s,N^Z)$, so Lemma 4 with $V=\{s\}$, $D=D^\star(s)$ gives $T\perp\!\!\!\perp Z^{j_s}\mid Z^{J_{D^\star(s)}}$. Since $s\notin D^\star(s)$ and principal atoms are distinct, $j_s\notin A_j$.

In both cases conditional independence implies $H_0(j\mid A_j)$ (Lemma 1.4), so $j\notin\mathrm{Adj}_Z(T)$. $\square$

### 4.3 Each assumption is needed: counterexamples

Each example satisfies all assumptions but one and the conclusion of Theorem 6 fails. In (a)–(c) all noises are independent standard Gaussians unless stated, $T\sim\mathrm{Bern}(1/2)$, and conditional dependence is shown by exhibiting a conditional mean that depends on $T$ (in a Gaussian model given $T$, this is exact).

**(a) Hidden co-parent of a child (violates A2 coverage; this is the brief's "hidden common cause").** $U\to F_1$, $U\to F_2$, $T\to F_1\to F_2$, with $U$ not represented: $F_1=\beta T+U+\varepsilon_1$, $F_2=\gamma F_1+U+\varepsilon_2$, $Z^1=F_1$, $Z^2=F_2$, $\beta,\gamma\neq0$. Given $\emptyset$, $\mathbb{E}[Z^2\mid T]=\gamma\beta T$. Given $Z^1$: $\mathbb{E}[U\mid T,F_1]=\tfrac12(F_1-\beta T)$, so $\mathbb{E}[Z^2\mid T,Z^1]=(\gamma+\tfrac12)F_1-\tfrac{\beta}{2}T$, which depends on $T$. These are the only conditioning sets, so $Z^2\in\mathrm{Adj}_Z(T)$ although $F_2$ is not a child. (Measured: OLS coefficient on $T$ given $F_1$ $=-0.2475$ at $\beta=0.5$; formula $-0.25$.)

**(b) Missing mediator (violates A2 coverage of a child).** $T\to M\to F_2$, $M$ has no atom, $Z^1=F_2$. Then $Z^1\in\mathrm{Adj}_Z(T)$ although $F_2$ is a grandchild. This is not an error of the procedure but a reminder that "direct" is relative to the dictionary.

**(c) Noisy proxy of a mediator (violates determination A2(a)).** $F_1=\beta T+\sigma_1\varepsilon_1$, $Z^1=F_1+\sigma_\eta\eta$, $F_2=\gamma F_1+\varepsilon_2$, $Z^2=F_2$. Given $T$, $(F_1,Z^1)$ is jointly Gaussian, so
$$\mathbb{E}[Z^2\mid T,Z^1]=\gamma\Big(\tfrac{\sigma_1^2}{\sigma_1^2+\sigma_\eta^2}Z^1+\tfrac{\sigma_\eta^2}{\sigma_1^2+\sigma_\eta^2}\beta T\Big).$$
The coefficient on $T$, $\gamma\beta\,\sigma_\eta^2/(\sigma_1^2+\sigma_\eta^2)$, is non-zero for every $\sigma_\eta>0$, so $Z^2\in\mathrm{Adj}_Z(T)$. It is small when the proxy is good — the *leakage* that §7.8 is about. (Measured: $0.0323$ for $\beta=0.5,\gamma=0.8,\sigma_1=1,\sigma_\eta=0.3$; formula $0.0330$.) This is why NEXIS's one-sided screening form of alignment is not enough here: a proxy that screens its factor *from the rest of the dictionary* but does not determine it still leaves the path $T\to F_1\to F_2$ open.

**(d) Impure principal atom of a spouse (violates purity A2(b)).** $T\to V\leftarrow L$ with $L\in\{0,1\}$ and a principal atom for $L$ that also reads $V$: $Z^L=L+\tfrac12\,\mathrm{sigmoid}(V)$. It determines $L$ ($L=\lfloor Z^L\rfloor$), but given $Z^V$ it is a function of $L$, and $T\not\perp\!\!\!\perp L\mid V$ (collider); given $\emptyset$ it depends on $V$, which depends on $T$. So $Z^L\in\mathrm{Adj}_Z(T)$. Real sparse-autoencoder atoms are often polysemantic in exactly this way (e.g. an atom for "hair" that also responds to "hat").

**(e) Failures of A3 (atom-level faithfulness).** (i) *Cancellation*: $F_1=\beta T+\varepsilon_1$, $F_2=aT+\gamma F_1+\varepsilon_2$ with $a=-\gamma\beta\neq0$; then $Z^2=F_2$ is independent of $T$ given $\emptyset$, so the child $F_2$ is missed. (ii) *Duplicate atom* (feature splitting/absorption): if $Z^p=Z^{j_c}$ for some $p\neq j_c$, then $H_0(j_c\mid\{p\})$ holds and *neither* atom is direct. (iii) *Near-determinism of $T$*: if a child is almost a copy of $T$ (a strongly visible treatment), $\pi_{\{j_c\}}\in\{0,1\}$ almost surely, every other child $c'$ satisfies $H_0(j_{c'}\mid\{j_c\})$ trivially (Lemma 1.2), and only one child can be recovered. In the limit $F_1=F_2=T$, the direct set is empty. Strong effects therefore *hurt* the identifiability of multiple direct effects; the benchmark must vary effect strength.

**(f) A3 with the GCM.** With a GCM test, recall needs the overlap-weighted contrast $\mathbb{E}[\pi_A(1-\pi_A)\Delta_{j_c,A}]$ to be non-zero for all $A$. Conditioning on a collider can make $\Delta_{j_c,A}$ change sign across values of $Z^A$; then the GCM has no power at any $n$ even though A3 holds. The PCM (§7.2) removes this blind spot at the cost of a sample split.

---

## 5. The indirect output

The name DINES says that the method returns the *structure* of the effect: direct atoms, and indirect atoms filed under the direct atoms that carry their effect. Marginal screening converges to $\mathrm{De}_Z(T)$ (NES Theorem 3.1, the "paradox"); DINES does not discard these atoms but organizes them.

**Definition 2 (indirect set and carriers).**
- Indirect set: $\mathrm{Ind}_Z(T):=\mathrm{De}_Z(T)\setminus\mathrm{Adj}_Z(T)$ — atoms with a non-zero average effect that are not direct.
- Causal carriers (Tier 2): for $j\in\mathrm{Ind}_Z(T)$, $\mathrm{Car}(j):=\{\,j_c:\ c\in\mathrm{Ch}(T),\ c\in\mathrm{An}(v)\text{ for some } v\in P(j)\,\}$ — the direct mechanisms from which a directed causal path reaches a factor read by atom $j$.
- Observable carriers: $\mathrm{Car}_B(j):=\{\,k\in J_C:\ H_0(j\mid B_Z\setminus\{k\})\text{ is false}\,\}$ — holding every other blanket atom fixed, releasing $k$ re-opens the effect of $T$ on $Z^j$.

Note $\mathrm{Car}(j)\neq\emptyset$ for $j\in\mathrm{Ind}_Z(T)$: if no child were an ancestor of $P(j)$, then $P(j)$ would contain no descendant of $T$, so $T\perp\!\!\!\perp Z^j$ and $j\notin\mathrm{De}_Z(T)$.

**Assumption N1 (no directed path between blanket members).** In $\mathcal{G}_L$ there is no directed path from a child of $T$ to another member of $\mathrm{MB}(T)$. (Allowed: spouses that are not descendants of $T$, as in the NES CelebA design; mediator chains below the children, $T\to Y_1\to Y_2$. Excluded: a child causing another child or a spouse.)

**Assumption CF (carrier faithfulness).** For $j\in\mathrm{Ind}_Z(T)$ and $j_c\in\mathrm{Car}(j)$, $H_0(j\mid B_Z\setminus\{j_c\})$ is false.

**Proposition 10 (carriers are identified).** Under A1–A3, N1 and CF, $\mathrm{Car}_B(j)=\mathrm{Car}(j)$ for every $j\in\mathrm{Ind}_Z(T)$, and every $j\in\mathrm{Ind}_Z(T)$ satisfies $H_0(j\mid B_Z)$ ("fully carried by the blanket").

*Proof.* The last claim is Lemma 3. $\mathrm{Car}(j)\subseteq\mathrm{Car}_B(j)$ is CF. For the converse take $c\in\mathrm{Ch}(T)$ with $c\notin\mathrm{An}(v)$ for all $v\in P(j)$, and let $C:=\mathrm{MB}(T)\setminus\{c\}$. We show $T\perp_d P(j)\mid C$. Take a path from $T$ to some $v\in P(j)$; its first edge is $T\to c'$ with $c'$ a child.
- If $c'\neq c$: $c'\in C$. If $c'$ is a non-collider the path is blocked. If it is a collider, the next vertex $x$ is a parent of $c'$; $x\neq c$ by N1 (else $c\to c'$ is a directed path between blanket members), so $x\in\mathrm{Sp}(T)\setminus\{c\}\subseteq C$ is a conditioned non-collider; blocked.
- If $c'=c$: $c\notin C$. If $c$ is a collider ($T\to c\leftarrow x$), it is open only if $c$ or a descendant of $c$ is in $C$; a descendant of $c$ in $C$ would be a directed path from a child to another blanket member, excluded by N1; blocked. Otherwise the path continues $c\to u_1$. Let $u_i$ be the first vertex after which an edge points backwards ($c\to u_1\to\dots\to u_i\leftarrow u_{i+1}$). Then $u_i$ is a collider, open only if $u_i$ or a descendant of it is in $C$; since $u_i\in\mathrm{De}(c)$, this again contradicts N1. So an open path must be directed all the way, $c\to\dots\to v$, which contradicts $c\notin\mathrm{An}(v)$.

So $T\perp_d P(j)\mid C$. Since $Z^j=H_j(L_{P(j)},N^Z)$, Lemma 4 (with $V=P(j)$, $D=C\subseteq\mathrm{MB}(T)$) gives $T\perp\!\!\!\perp Z^j\mid Z^{B_Z\setminus\{j_c\}}$, hence $H_0(j\mid B_Z\setminus\{j_c\})$ and $j_c\notin\mathrm{Car}_B(j)$. $\square$

Without N1 the two notions differ in a predictable way: for $T\to c_1\to c_2\to Z^j$ (child causes child), $\mathrm{Car}(j)=\{j_{c_1},j_{c_2}\}$ but $\mathrm{Car}_B(j)=\{j_{c_2}\}$ — the observable notion reports the *last* direct mechanism on the path. A general observable characterization is Open question O4.

**Estimation.** The population blanket $B_Z$ is unknown; DINES estimates it by the shrink step of IAMB applied to the forward output $\tilde S$, at a level $\alpha_n\to0$:
$$\hat B:=\{\,j\in\tilde S:\ p_j(\tilde S\setminus\{j\})\le\alpha_n\,\},\qquad \widehat{\mathrm{Ind}}:=\{\,j\notin\hat S:\ p_j(\emptyset)\le \tfrac{\alpha_2}{2m}\,\},$$
$$\widehat{\mathrm{Car}}(j):=\{\,k\in\hat S:\ p_j(\hat B\setminus\{k\})\le \tfrac{\alpha_2}{2m\max(1,|\hat S|)}\,\}\quad (j\in\widehat{\mathrm{Ind}}).$$

**Assumption MTF (mean total-conditioning faithfulness on the blanket).** For every $S$ with $B_Z\subseteq S\subseteq[m]$ and every $j\in B_Z$, $H_0(j\mid S\setminus\{j\})$ is false.

**Proposition 13 (guarantee for the full output).** Assume the conditions of Corollary 12, N1, CF and MTF. Assume the tests are consistent at fixed alternatives also at the vanishing level $\alpha_n$ (i.e. $\mathbb{P}(p_j(A)\le\alpha_n)\to1$; for asymptotically normal statistics whose mean grows like $\sqrt n$ under a fixed alternative this typically holds when $\log(1/\alpha_n)=o(n)$, but we assume it rather than prove it), consistent against $H_0(j\mid\emptyset)$ for $j\in\mathrm{Ind}_Z(T)$, and valid at the fixed nulls used below. Then
$$\liminf_n\ \mathbb{P}\Big(\hat S=J_C,\ \widehat{\mathrm{Ind}}=\mathrm{Ind}_Z(T),\ \widehat{\mathrm{Car}}(j)=\mathrm{Car}(j)\ \forall j\in\mathrm{Ind}_Z(T)\Big)\ \ge\ 1-\alpha-\alpha_2 .$$

*Proof.* (i) $\mathbb{P}(\hat B=B_Z)\to1$: by step (i) of the proof of Theorem 11, $\mathbb{P}(B_Z\subseteq\tilde S)\to1$. $\tilde S$ takes finitely many values. For a fixed $S\supseteq B_Z$: if $j\in S\setminus B_Z$, Lemma 3 and weak union give $T\perp\!\!\!\perp Z^j\mid Z^{S\setminus\{j\}}$, so $H_0(j\mid S\setminus\{j\})$ holds, and asymptotic validity gives $\limsup\mathbb{P}(p_j(S\setminus\{j\})\le\alpha_n)\le t$ for every fixed $t>0$ (since eventually $\alpha_n\le t$), i.e. this probability tends to $0$; if $j\in B_Z$, MTF and consistency at level $\alpha_n$ give $\mathbb{P}(p_j(S\setminus\{j\})\le\alpha_n)\to1$. A finite union bound over $(S,j)$ gives the claim. (ii) Let $E:=\{\hat S=J_C,\ \hat B=B_Z\}$; by Corollary 12 and (i), $\limsup\mathbb{P}(E^c)\le\alpha$. On $E$ every remaining test is at a fixed conditioning set: $p_j(\emptyset)$ and $p_j(B_Z\setminus\{k\})$, with $|\hat S|=|J_C|$ fixed. Errors on $E$ are of four kinds: a $j\notin\mathrm{De}_Z(T)\cup J_C$ with $p_j(\emptyset)\le\alpha_2/(2m)$ — at most $m$ fixed true nulls, total $\limsup\le\alpha_2/2$; a $j\in\mathrm{Ind}_Z(T)$ missed — finitely many fixed alternatives, probability $\to0$; a $k\notin\mathrm{Car}(j)$ included — by Proposition 10 a fixed true null $H_0(j\mid B_Z\setminus\{k\})$, at most $m|J_C|$ of them at level $\alpha_2/(2m|J_C|)$, total $\limsup\le\alpha_2/2$; a $k\in\mathrm{Car}(j)$ missed — fixed alternatives (CF), probability $\to0$. Summing, $\limsup\mathbb{P}(\text{error})\le\alpha+\alpha_2$. $\square$

*Stability warning.* The indirect output is less robust than the direct set. Under approximate alignment (§7.8), conditioning on noisy proxies of the children leaves part of every mediated effect, so $H_0(j\mid B_Z)$ fails for many indirect atoms and carriers are over-reported; and spurious direct atoms (leaked atoms) change $\hat S$ and hence every carrier test. Report carriers as hypotheses, together with the residual test $p_j(\hat B)$ ("is $j$ fully carried?").

---

## 6. Algorithm and selection guarantee

### 6.1 Algorithm

```
DINES(data (T, Z, optional W), level α, indirect level α2, shrink level α_n, test)
  # every p_j(A) tests H0(j | A): E[Z^j | T, Z^A, W] = E[Z^j | Z^A, W]

  # 1. Forward phase (IAMB grow phase)
  S ← ∅
  while |S| < m:
      R ← [m] \ S
      j* ← argmin_{j ∈ R} p_j(S)               # smallest p-value, NOT largest raw effect
      if p_{j*}(S) ≤ α / |R|: S ← S ∪ {j*}      # gate; Bonferroni here is optional
      else: break
  S̃ ← S

  # 2. Terminal backward phase (direct set)
  Ŝ ← { j ∈ S̃ : p_j(A) ≤ α/m for EVERY A ⊆ S̃ \ {j} (∅ included) }   # all removals simultaneous

  # 3. Indirect output (optional)
  B̂ ← { j ∈ S̃ : p_j(S̃ \ {j}) ≤ α_n }
  Înd ← { j ∉ Ŝ : p_j(∅) ≤ α2 / (2m) }
  for j in Înd:
      Ĉar(j) ← { k ∈ Ŝ : p_j(B̂ \ {k}) ≤ α2 / (2m · max(1,|Ŝ|)) }
      report also the residual p-value p_j(B̂)
  return Ŝ, {(j, Ĉar(j)) : j ∈ Înd}
```

**Cost.** Forward: at most $\sum_{r=0}^{|\tilde S|}(m-r)$ tests. Terminal: at most $|\tilde S|\,2^{|\tilde S|-1}$ tests; a candidate is dropped at its first non-significant subset, so ordering the subsets well (e.g. $\emptyset$ and small sets first for atoms with weak marginal effect, large sets first otherwise) saves most of them. Under Tier 2, $\tilde S\supseteq B_Z$, which contains the spouses, so $\tilde S$ is larger than in NEXIS (in the toy of Appendix B, $|\tilde S|\approx4$). The terminal level is $\alpha/m$ regardless of how many subsets are tested.

**Ordering.** The ordering rule does not enter the proof of Theorem 11: any rule that admits only atoms passing the gate gives the same asymptotic guarantee. We order by $p$-value (equivalently by $|$standardized statistic$|$ when all tests share a reference distribution) for finite-sample reasons: it does not depend on the scale of each atom (sparse-autoencoder atoms have very different activation scales), it is IAMB's heuristic, and it keeps $\tilde S$ small. Ordering by raw magnitude is what NES needed its Principal–Max assumption for, and it fails in practice (§8.2).

**Forward versus terminal tests.** The forward phase needs only *consistency* (to reach the pool); validity is needed only in the terminal phase at fixed nulls. So a fast screening test (e.g. an OLS interaction $t$-test) may be used in the forward phase, and a careful valid test (§7.2) in the terminal phase, without changing the guarantee — provided the fast test is consistent against the alternatives in Assumption P2.

### 6.2 Assumptions for the statistical guarantee

**Assumption P (separator pool).** There is a fixed set $B\subseteq[m]$ such that
- (P1) every $j\notin\mathrm{Adj}_Z(T)$ has a fixed $A_j\subseteq B\setminus\{j\}$ with $H_0(j\mid A_j)$ true;
- (P2) (forward completeness, mean version) for every $S\subseteq[m]$ with $B\not\subseteq S$ there is $j\notin S$ with $H_0(j\mid S)$ false.

P is a property of the joint law of $(T,Z)$ only. P1 alone is trivial with $B=[m]$ (by definition of $\mathrm{Adj}$); the content is that a *small* pool exists *and* the forward phase is forced to reach it. P2 is the mean analogue of the composition property that IAMB's grow phase needs. Under Tier 2, P1 holds with $B=B_Z$ (Theorem 6); P2 with $B=B_Z$ is then an assumption we call **A4**. A4 says: if some child atom is missing from $S$, some atom is still mean-dependent on $T$ given $S$ (this already follows from A3), and if all children but some spouse atom are in $S$, some atom (typically that spouse) is mean-dependent on $T$ given $S$ (the collider dependence of §3.4).

**Assumption A5 (tests).** (a) *Validity at fixed nulls:* for each pair $(j,A_j)$ of P1, $\limsup_n\mathbb{P}(p_j(A_j)\le t)\le t$ for all $t\in(0,1)$. (b) *Consistency at fixed alternatives:* for every fixed pair $(j,A)$ with $H_0(j\mid A)$ false that is used below — all $(j,A)$ with $j\in\mathrm{Adj}_Z(T)$, and one witness $j$ for each $S$ in P2 — $\mathbb{P}(p_j(A)\le t)\to1$ for every $t>0$.

### 6.3 Main theorem

**Theorem 11 (selection consistency, Tier 1).** Fix $m$ and $\alpha\in(0,1)$. Under Assumptions P and A5, the DINES direct set $\hat S_n$ satisfies
$$\lim_n\mathbb{P}\big(\mathrm{Adj}_Z(T)\subseteq\hat S_n\big)=1,\qquad \limsup_n\mathbb{P}\big(\hat S_n\not\subseteq\mathrm{Adj}_Z(T)\big)\le\frac{m-|\mathrm{Adj}_Z(T)|}{m}\,\alpha,$$
hence $\liminf_n\mathbb{P}\big(\hat S_n=\mathrm{Adj}_Z(T)\big)\ge1-\alpha$.

*Proof.* Write $\mathcal{A}:=\mathrm{Adj}_Z(T)$.

(i) *The forward phase reaches the pool and the direct set.* Let $\mathcal{S}:=\{S\subseteq[m]:(B\cup\mathcal{A})\not\subseteq S\}$. For each $S\in\mathcal{S}$ pick a witness $w_S\notin S$ with $H_0(w_S\mid S)$ false: if $B\not\subseteq S$ use P2; otherwise some $j\in\mathcal{A}\setminus S$ exists and $H_0(j\mid S)$ is false by the definition of $\mathcal{A}$. By A5(b), $\mathbb{P}(p_{w_S}(S)\le\alpha/|[m]\setminus S|)\to1$. Let $\mathcal{E}_n:=\bigcap_{S\in\mathcal{S}}\{\min_{j\notin S}p_j(S)\le\alpha/|[m]\setminus S|\}$; a finite intersection, so $\mathbb{P}(\mathcal{E}_n)\to1$. The forward phase adds one atom per round and never removes one, so it stops after at most $m$ rounds; on $\mathcal{E}_n$ it cannot stop at any $S\in\mathcal{S}$. Hence on $\mathcal{E}_n$, $B\cup\mathcal{A}\subseteq\tilde S$.

(ii) *Direct atoms survive the terminal phase.* For $j\in\mathcal{A}$ and every $A\subseteq[m]\setminus\{j\}$, $H_0(j\mid A)$ is false. These are finitely many fixed alternatives, so by A5(b) and a union bound $\mathbb{P}(\forall j\in\mathcal{A},\forall A:\ p_j(A)\le\alpha/m)\to1$. The terminal phase decides on $j$ using only $p_j(A)$ with $A\subseteq\tilde S\setminus\{j\}\subseteq[m]\setminus\{j\}$, and removals are simultaneous, so on this event and on $\{\mathcal{A}\subseteq\tilde S\}$ every $j\in\mathcal{A}$ is kept. With (i), $\mathbb{P}(\mathcal{A}\subseteq\hat S_n)\to1$.

(iii) *Precision with fixed nulls only.* Let $\mathcal{R}_n:=\{B\subseteq\tilde S\}$. Fix $j\notin\mathcal{A}$. On $\mathcal{R}_n$, if $j\in\hat S_n$ then $j\in\tilde S$, and $A_j\subseteq B\setminus\{j\}\subseteq\tilde S\setminus\{j\}$ is one of the subsets at which $j$ was tested, so $p_j(A_j)\le\alpha/m$. The null $H_0(j\mid A_j)$ is true (P1) and $A_j$ is fixed, not chosen from the data. By A5(a) and a union bound,
$$\limsup_n\mathbb{P}\big(\mathcal{R}_n,\ \hat S_n\not\subseteq\mathcal{A}\big)\le\sum_{j\notin\mathcal{A}}\limsup_n\mathbb{P}\big(p_j(A_j)\le\alpha/m\big)\le\frac{m-|\mathcal{A}|}{m}\alpha.$$
Since $\mathbb{P}(\mathcal{R}_n^c)\to0$ by (i), the same bound holds for $\mathbb{P}(\hat S_n\not\subseteq\mathcal{A})$.

(iv) $\mathbb{P}(\hat S_n\neq\mathcal{A})\le\mathbb{P}(\mathcal{A}\not\subseteq\hat S_n)+\mathbb{P}(\hat S_n\not\subseteq\mathcal{A})$; the first term $\to0$ by (ii), the second has $\limsup\le\alpha$ by (iii). $\square$

No step uses validity at a data-dependent conditioning set: (i) and (ii) use consistency at finitely many fixed pairs, and (iii) uses validity at the $m-|\mathcal{A}|$ fixed pairs $(j,A_j)$. The forward Bonferroni gate is not needed for the bound (any fixed positive threshold works at fixed $m$); the terminal level $\alpha/m$ is what controls precision. A data-dependent terminal level such as $\alpha/|\tilde S|$ is *not* covered by this proof.

**Corollary 12 (causal selection, Tier 2).** Under A1–A5 (with A4 = P2 for $B=B_Z$),
$$\liminf_n\mathbb{P}\big(\hat S_n=\{j_c:c\in\mathrm{Ch}(T)\}\big)\ge1-\alpha,$$
and the recovered set indexes the shifted mechanisms $\mathrm{Shift}(T)$ under causal minimality.

*Proof.* Theorem 6 gives $\mathrm{Adj}_Z(T)=J_C$ and P1 with $B=B_Z$ and the stated $A_j$; A4 is P2. Apply Theorem 11, then Proposition 8.3. $\square$

### 6.4 The separating-set question, answered

*Question.* In NEXIS, each false coordinate is bounded by its test at the fixed set $\mathcal{S}^\star$ (the target), which lies in the forward output. Here the target $J_C$ does not separate spouse atoms (conditioning on a child opens $T\to c\leftarrow s$). Does every non-direct atom have a separating set inside the forward output?

*Answer: yes, under Tier 2 and A4, and the separating set is fixed.* On the event $B_Z\subseteq\tilde S$ (probability $\to1$ by A4):
- atoms that read no descendant of $T$: $\emptyset$ separates (also $B_Z$ does);
- atoms outside $B_Z$: $B_Z$ separates (Lemma 3, a Markov-blanket argument);
- principal atoms of spouses, *including spouses that are descendants of $T$*: $J_{D^\star(s)}$ with $D^\star(s)=\mathrm{MB}(T)\setminus(\{s\}\cup\mathrm{De}(s))$ separates (Lemma 5). The key step is that, $T$ being a root, all its moral neighbours in the relevant ancestral set are children or co-parents that are not descendants of $s$.

No enlargement of the subset pool, no symmetry (AND) check à la Peña et al. (2007), and no test over all affected atoms is needed. Two caveats. (1) Without Tier 2 — for a general dictionary whose atom-level dependence structure is only an ancestral graph with $T$ a root — we do not know whether a separator always exists inside a Markov blanket of $T$, and the set $D^\star$ as defined can fail there (a collider path $T\to c\leftrightarrow X$ through a blanket member $c$ that is not an ancestor of $X$ is opened by conditioning on $c$); this is Open question O1, and is why Theorem 11 carries P as an assumption. (2) A4 (forward completeness) is an assumption, the mean analogue of composition.

---

## 7. Conditional tests in practice

### 7.1 What is tested

$H_0(j\mid A)$: $\mathbb{E}[Z^j\mid T,Z^A,W]=\mathbb{E}[Z^j\mid Z^A,W]$. The tested atom $Z^j$ stays continuous. The conditioning atoms stay continuous.

### 7.2 Residualization instead of stratification: the GCM test

For a pair $(j,A)$, with conditioning variables $V_A:=(Z^A,W)$:
1. Split the units (for video: the clusters) into $K$ folds. On each fold, use models fitted on the other folds:
   - $\hat\pi(V_A)$, an estimate of $\mathbb{P}(T=1\mid V_A)$ (a classifier);
   - $\hat\mu_j(V_A)$, an estimate of $\mathbb{E}[Z^j\mid V_A]$ (a regression).
2. $R_i:=(T_i-\hat\pi(V_{A,i}))\,(Z^j_i-\hat\mu_j(V_{A,i}))$, $\ T_n:=\sqrt n\,\bar R/\hat\sigma_R$, two-sided normal $p$-value.

*What it estimates.* By Lemma 1.3, $\mathbb{E}[R]\approx\mathbb{E}[\pi_A(1-\pi_A)\Delta_{j,A}]$: the conditional arm contrast averaged with overlap weights.

*Validity.* Under $H_0(j\mid A)$ the population residual product has conditional mean zero, $\mathbb{E}[(T-\pi_A)(Z^j-\mu_{j,A})\mid V_A]=\pi_A(1-\pi_A)\Delta_{j,A}=0$. The standard decomposition of the GCM statistic (Shah & Peters, 2020) uses the null only through this identity and through $\mathbb{E}[T-\pi_A\mid V_A]=\mathbb{E}[Z^j-\mu_{j,A}\mid V_A]=0$, so the test is asymptotically valid under the *mean* null under conditions of their type: the product of the two regression errors is $o_P(n^{-1/2})$ in $L_2$, each error vanishes (also when weighted by the other residual's conditional variance), and $\mathbb{E}[R^2]$ is bounded away from zero with a slightly higher moment bounded. (Scheidegger et al., 2022, and Lundborg et al., 2024, discuss the GCM as a test of this weaker, covariance-type null.)

*The propensity is not known.* In NEXIS the conditioning set is pre-treatment, so $\mathbb{P}(T=1\mid Z^A)=e$ by design. Here $Z^A$ is post-treatment, so $\pi_A$ depends on $Z^A$ and must be learned — it is a classifier of the treatment from the selected atoms. This is the price of a post-treatment dictionary. Only the first step ($A=\emptyset$) has a known propensity.

*No free lunch.* Shah & Peters (2020) show that no conditional-independence test with continuous conditioning variables can have uniformly valid size and non-trivial power against all alternatives. The regression rate conditions above are the assumption we pay with.

*Power.* The GCM has power against fixed alternatives with $\mathbb{E}[\pi_A(1-\pi_A)\Delta_{j,A}]\neq0$, at the usual $\sqrt n$ rate. It is blind when the contrast changes sign across $V_A$ (§4.3 (f)). Remedies: the PCM of Lundborg et al. (2024) with response $Z^j$, covariate $T$, conditioning set $V_A$ (learn $\hat h\approx\Delta_{j,A}$ on one half, test $\mathbb{E}[(T-\hat\pi)\hat h(Z^j-\hat\mu)]$ on the other), equivalently a weighted GCM (Scheidegger et al., 2022). We suggest the GCM by default and the PCM in the terminal phase when sign changes are plausible.

*Positivity.* When $Z^A$ nearly reveals $T$, $\hat\pi$ is near 0 or 1, residuals $T-\hat\pi$ are tiny, and power vanishes (this is §4.3 (e)(iii) at finite $n$). Report the effective sample size $\sum_i\hat\pi_i(1-\hat\pi_i)$ for every terminal test.

**Why binned stratification fails.** NES (Algorithm 2; code: atoms binarized at $>0$) compares arms within cells defined by binarized selected atoms. A binarized atom $b(Z^{j_c})$ does not determine the factor even if $Z^{j_c}$ does, so it is a noisy proxy and leaves mediated paths open (§4.3 (c)). For $T\to F_1\to F_2$ with $F_1\mid T\sim\mathcal{N}(\beta T,1)$ and strata $\{F_1>0\},\{F_1\le0\}$: within the upper stratum the arm difference of $\mathbb{E}[F_1]$ is $\beta+\lambda(\beta)-\lambda(0)$ with $\lambda=\varphi/\Phi$, which for small $\beta$ is $\beta(1-2/\pi)\approx0.36\beta$ (and symmetrically below). So stratification removes only about 64% of a mediated effect. Measured ($n=2\times10^6$, $F_2=F_1+\varepsilon$): retained fraction $0.366$, $0.379$, $0.418$ for $\beta=0.2,0.5,1.0$; small-$\beta$ limit $1-2/\pi=0.363$. In addition, with $|S|$ binarized atoms there are $2^{|S|}$ cells, and because the atoms are post-treatment, strong effects create cells with only one arm. NES drops those cells and renormalizes the weights (code lines 73–95), which silently changes the estimand.

### 7.3 Binarization

Do not binarize conditioning atoms (§7.2). Keep the tested atom continuous: for zero-inflated sparse-autoencoder codes, a mean test detects changes in firing rate and in firing magnitude. The regressions $\hat\mu,\hat\pi$ should be able to represent the point mass at zero (e.g. include $\mathbb{1}[Z^k>0]$ as a feature, or use trees). Binarize only to *interpret* a selected atom (e.g. report the change in firing rate).

### 7.4 Pre-treatment covariates and AIPW

**Proposition 14 (covariates do not change the target).** Let $W$ be observed pre-treatment variables that are vertices of $\mathcal{G}_L$ and non-descendants of $T$. Define $\mathrm{Adj}^W_Z(T)$ as in Definition 1 with $W$ added to every conditioning set. Under A1–A2, and A3 with $W$ added to every conditioning set, $\mathrm{Adj}^W_Z(T)=J_C$.

*Proof.* "$\supseteq$": A3 with $W$. "$\subseteq$": for $j\notin B_Z$, Lemma 3 with $Y=(Z^j,W)$ ($W$ is a function of $L$) gives $T\perp\!\!\!\perp(Z^j,W)\mid Z^{B_Z}$, and weak union gives $T\perp\!\!\!\perp Z^j\mid(Z^{B_Z},W)$. For $j=j_s$, $s$ a non-child spouse: we show $T\perp_d s\mid D^\star(s)\cup W$. Let $\mathcal{A}'=\mathrm{An}(\{T,s\}\cup D^\star(s)\cup W)$. A child $c$ of $T$ in $\mathcal{A}'$ is not an ancestor of $W$ (its descendants are descendants of $T$, and $W$ is not), so $c\in\mathrm{An}(\{T,s\}\cup D^\star(s))$; by Step 1 of Lemma 5, $c\notin\mathrm{De}(s)$, so $c\in D^\star(s)$; the co-parent argument of Step 2 is unchanged. So all moral neighbours of $T$ in $(\mathcal{G}_{\mathcal{A}'})^m$ lie in $D^\star(s)\cup W$, which separates $T$ from $s\notin D^\star(s)\cup W$. Lemma 4, with $L_D$ replaced by $(L_{D^\star},W)$ (observed directly, so trivially determined), gives $T\perp\!\!\!\perp Z^{j_s}\mid (Z^{J_{D^\star(s)}},W)$. $\square$

So in a randomized experiment covariates add *power*, not validity: they explain variance in the atoms and in the propensity. For the first step ($A=\emptyset$) the propensity $\mathbb{P}(T=1\mid W)=e$ is known, and $R_i=(T_i-e)(Z^j_i-\hat\mu_j(W_i))$ with cross-fitted $\hat\mu_j$ has mean $e(1-e)\,\mathrm{ATE}_j$ whatever $\hat\mu_j$ is (because $T\perp\!\!\!\perp W$): this is the covariate-adjusted (ANCOVA/augmented IPW) estimator of the average effect (Tsiatis et al., 2008; Lin, 2013), valid by randomization alone. In *observational* data $T$ is no longer a root; if $W$ contains all parents of $T$ and is always conditioned on, we expect the same results (the moral-graph argument then also has $W$ among $T$'s neighbours), but we have not written this proof out (Open question O8).

### 7.5 Clustered data (video)

Frames of one video, or of one pool of animals, are not independent, and the treatment is assigned per video or pool. The unit of independence is the cluster $g=1,\dots,G$. Use
$$T_G=\frac{\sum_g R_g}{\sqrt{\sum_g\big(R_g-n_g\bar R\big)^2}},\qquad R_g=\sum_{i\in g}R_i,$$
form the cross-fitting folds from whole clusters (otherwise the nuisance models memorize a video and the residuals are biased), and read the asymptotics as $G\to\infty$. With few clusters (tens of videos or pools) use a $t_{G-1}$ reference and small-sample corrections (Cameron & Miller, 2015). Two warnings. (1) Permutation of $T$ is exact only for the unconditional step $A=\emptyset$; for $A\neq\emptyset$ permuting $T$ breaks the dependence between $T$ and the post-treatment $Z^A$, and a conditional randomization test would need the unknown $\pi_A$. (2) The target is defined for a randomly drawn frame; decide whether frames or videos are weighted equally, and keep it fixed. Averaging atoms over a video first is a different dictionary (an average of functions is not a function of averaged factors), so alignment must hold for the averaged atoms.

### 7.6 Forward-phase tests

The forward phase may use a cheap test (OLS of $Z^j$ on $T$, $Z^S$, $W$ with a $t$-test on $T$, or a linear GCM), because only consistency matters there (§6.1). The terminal phase should use the cross-fitted GCM/PCM.

### 7.7 The $p$-value gate and the terminal level

Forward gate: Bonferroni $\alpha/|R|$ by default (not needed for the theorem; a looser gate admits more atoms and makes the terminal phase more expensive). Terminal level: $\alpha/m$, with $m$ the number of atoms entering the analysis. Atoms that never fire in the sample may be dropped before the analysis (this uses $Z$ only, not $T$), which lowers $m$.

### 7.8 Approximate alignment and the margin $\delta$ (practical, not part of the theory)

The theory uses the exact null. Under *approximate* alignment — a principal atom that only nearly determines its factor, which is the realistic case for sparse autoencoders (in NES's CelebA dictionary the best atom for "wearing hat" has F1 $0.94$ and the best for "eyeglasses" $0.74$) — §4.3 (c) shows that atoms downstream of a mediator keep a small but non-zero conditional effect given *every* set of atoms. These **leaked atoms really are in $\mathrm{Adj}_Z(T)$ for that dictionary**. So:
- the *statistical* guarantee (Theorem 11) is intact: DINES recovers $\mathrm{Adj}_Z(T)$, leaked atoms included, with probability $\ge1-\alpha$ asymptotically;
- the *causal* reading (Theorem 6) fails: $\mathrm{Adj}_Z(T)$ is larger than the set of child atoms, and the number of leaked atoms that pass the tests grows with $n$. Precision against the causal truth falls with $n$, for DINES as for NES.

The measured toy confirms this (Appendix B, proxy noise $\sigma_\eta=0.3$ on all principal atoms): exact recovery 96/100, 96/100, 84/100 at $n=1000,4000,16000$, with false atoms in the output rising from 0.01 to 0.17 per run (mostly the leaky mixed atom and the mediator's atom).

*Option.* The implementation exposes a relevance margin $\delta\ge0$ (default $0$): test $H_0^\delta(j\mid A):\ |\theta_{j,A}|\le\delta$ instead of $\theta_{j,A}=0$, with a standardized effect such as the residual correlation $\theta_{j,A}=\mathbb{E}[R]/\sqrt{\mathbb{E}[(T-\pi_A)^2]\,\mathbb{E}[(Z^j-\mu_{j,A})^2]}$, rejecting when $|\hat\theta|-\delta$ exceeds a one-sided normal quantile times its standard error. This is where effect *magnitude* legitimately re-enters — on a standardized scale, and as a declared tolerance rather than a ranking rule. NEXIS's spectral-gap gate $\rho$ is a relative version of the same idea. In the toy, $\delta=0.05$ gives 100/100 exact recovery at $n\ge4000$ under approximate alignment, but costs recall at small $n$ (recall 0.85 and 0.84 at $n=1000$ with and without proxy noise). The $\delta$ version changes the target to "atoms whose conditional effect exceeds $\delta$ given every set", which is not $\mathrm{Adj}_Z(T)$; we do not build it into the definition. Whether it is needed is decided empirically: **if DINES's precision on the CelebA benchmark falls with $n$, alignment is approximate and $\delta>0$ is warranted.** How to choose $\delta$ on a principled scale is Open question O5.

### 7.9 A dictionary fitted on the same data

All results treat $\psi$ as fixed. If the sparse autoencoder is trained on the experiment's own frames (as for ISTAnt), $\psi$ depends on the data, possibly on treatment-related structure. Safe options: train $\psi$ on held-out units, or cross-fit the dictionary. The effect of reusing the data is not covered (Open question O7).

---

## 8. What NES got wrong, and what DINES does instead

NES is "Neural Effect Search" from *Exploratory Causal Inference* (Mencattini, Cadei, Locatello; arXiv 2510.14073). Its central observation — marginal tests on an entangled dictionary flag ever more atoms as power grows (its Theorems 3.1–3.2) — is correct, and DINES keeps it: those atoms form $\mathrm{De}_Z(T)$. The problems are in the target, the selection rule, the conditioning device and the proof. Equation numbers refer to the arXiv version; code references are to `src/neuralscience/nes.py` in the NES project (`/nfs/scistore19/locatgrp/tmencatt/projects/neuralscience/`), which we read; the CelebA numbers are from the saved runs in `results/experiments-semi-synthetic/` of that project (seeds 420–429; Appendix B).

### 8.1 The target is never defined, and its proof target excludes mediation

*Claim (§2).* The goal is "the coarsest possible abstraction of the treatment effect in $X$", with $[Y,W]$ "the minimal and sufficient representation for $X$ that renders the treatment independent from the observations, i.e. $T\perp\!\!\!\perp X\mid[W,Y]$, where only $Y$ is causally affected"; the theorem targets "principal neurons of all $r$ affected outcomes".

*Why it fails.* (1) "Coarsest abstraction" is not given a formal definition. (2) The sufficiency condition $T\perp\!\!\!\perp X\mid[W,Y]$ is a Markov-blanket condition; with post-treatment variables it brings in unaffected spouses — which the paper puts in $W$ — and does not say whether a factor *caused by another affected factor* belongs to $Y$. (3) The theorem's target includes every factor with non-zero total effect, mediated ones included, but its Assumption A.3 (Eq. 28) states that after conditioning on the atoms of discovered factors the remaining contrast at atom $j$ is $\sum_{k\notin K(S)}\tau^Y_k v_{k,j}$, the sum of the *total* effects of undiscovered factors. In a mediator design $T\to Y_1\to Y_2$ with $Y_1$'s atom determining $Y_1$, conditioning on it removes all of $Y_2$'s effect, so the remaining contrast at $Y_2$'s atom is $0$, not $\tau^Y_2v_{2,j}$. Eq. 28 therefore silently assumes no causal relation among affected factors (and no spouses: in NES's own design Smiling is a co-parent of Eyeglasses, §3.4, and Eq. 28 predicts zero contrast at a Smiling atom after conditioning on Eyeglasses, whereas the collider makes it non-zero).

*DINES.* Definition 1 (direct set) with its causal reading (Theorem 6, Proposition 8); mediated factors are a separate, defined output (§5).

### 8.2 Selection by raw magnitude, and the Principal–Max assumption

*Claim (Algorithm 1 line 6; Eqs. 25–26, 31).* Among significant atoms NES adds the one with the largest raw $|\hat\tau_j|$ (code line 303). The theorem assumes the global argmax of $|\tau^Z|$ is a principal atom (Eq. 26) and argues that "replacing $\sum_{\ell=1}^r$ in Eq. 26 by a subsum over the remaining indices only reduces nonprincipal candidates; hence the argmax remains principal at every round".

*Why it fails.* (1) Raw effects are not comparable across atoms: sparse-autoencoder atoms have different activation scales, and the raw effect mixes effect size with scale. (2) The preservation argument is false, because removing a term from a signed sum can *increase* its absolute value. Counterexample satisfying A.1, Eq. 25, Eq. 26 and Eq. 31 (with any $\varepsilon<1$): three atoms, two factors, $v_1=(3,0,-1)$, $v_2=(0,1.5,2)$, $\tau^Y=(1,1)$. Round 1: $\tau^Z=(3,1.5,1)$, argmax atom 1, principal. Round 2 under Eq. 28: remaining contrasts $(\cdot,1.5,2)$, argmax atom 3, *not* principal. Measured on a Gaussian instance ($Y_k=T+\varepsilon_k$, $Z=(3Y_1,1.5Y_2,-Y_1+2Y_2+\varepsilon)$, $n=20{,}000$, OLS adjustment): in round 2 atom 3 has the larger effect ($2.0$ vs $1.5$) but the smaller $t$ ($56.5$ vs $62.8$); ordering by magnitude picks atom 3, after which atom 2 is still significant ($t=25.6$) and NES ends with $\{1,3,2\}$; ordering by $t$ picks atom 2, after which atom 3 is null ($t=0.34$). (3) *Evidence in the saved CelebA runs* (seed 420, Bonferroni-NES): in round 2, stratified on N_38 $>0$ (N_38 = hat atom), the mixed atom N_408 (F1 $0.48$ for hat, $0.45$ for eyeglasses) has a larger raw effect than the eyeglasses atom N_6051 but about half its $t$-statistic — $0.773$ ($t=4.8$) vs $0.691$ ($t=8.8$) at ATE $0.4$, $N=1000$; $1.324$ ($t=7.1$) vs $1.114$ ($t=12.4$) at ATE $0.7$, $N=1000$ — so NES selects N_408 before N_6051 (saved selection: 38, 408, 6051). Over 80 runs at $N=5000$ (ATE $0.1$–$0.8$, 10 seeds), N_408 is the most frequent false selection (50/80 runs).

*DINES.* Order by $p$-value; and the guarantee does not depend on the order at all (§6.1), because a terminal backward step removes whatever entered wrongly.

### 8.3 Eqs. 27–28 assume the conclusion

*Claim (Assumption A.3).* Conditioning within arms on the selected atoms "explains their contribution in expectation, leaving only undiscovered effects" (Eqs. 27–28).

*Why it fails.* This is the disentanglement property the theorem is supposed to establish, stated as an assumption. It is false in general for post-treatment conditioning: mediation (§8.1), colliders (§3.4), noisy proxies (§4.3 (c)), and binarized strata, which remove only about 64% of a mediated effect even with a perfectly aligned atom (§7.2, measured retained fraction $0.37$–$0.42$).

*DINES.* Replaces A.3 by structural assumptions (A1–A3) whose consequences are proved (Lemmas 2–5, Theorem 6), and by conditional tests whose validity conditions are explicit (§7.2).

### 8.4 The transport-error term of Eqs. 29–30 is identically zero

*Claim.* The arm/stratum discrepancy $\Delta_{j,t}(g):=\mathbb{E}[Z_j\mid G=g,T=t]-\mathbb{E}[Z_j\mid G=g,do(T=t)]$ (defined just before Eq. 29) captures the collider bias of post-treatment conditioning; Eq. 29 says it vanishes once all factors are found, and Eq. 30 bounds it by $\varepsilon$. Lemma A.1 then bounds the "leakage bias" $B_j$ by $\varepsilon$.

*Why it fails.* In a randomized experiment the joint law of all post-treatment variables under $do(T=t)$ equals their law given $T=t$; so $\mathbb{E}[Z_j\mid G=g,do(T=t)]=\mathbb{E}[Z_j\mid G=g,T=t]$ and $\Delta_{j,t}(g)=0$ for every $j,g,t$, always. Eq. 29 is trivially true, Eq. 30 holds with $\varepsilon=0$, and $B_j=0$. The real post-treatment problem is that $\mathbb{E}[Z_j\mid G=g,do(T=1)]-\mathbb{E}[Z_j\mid G=g,do(T=0)]$ compares two *different* subpopulations (units that land in stratum $g$ under treatment versus under control) and is not a causal effect. The paper's bound does not touch it; Eq. 28 assumes it away.

*DINES.* Does not compare strata as if they were causal contrasts. It asks a conditional-independence question (is $Z^j$'s mean shift explainable by other atoms?) and gives it a causal meaning only through Theorem 6.

### 8.5 "FWER → 0" is wrong at fixed $\alpha$

*Claim (Note after the Theorem in App. A.3).* $\mathbb{P}(S_{\rm final}=\{j_1,\dots,j_r\})\to1$, hence FWER and FDR converge to $0$. The termination step argues that after $r$ rounds the remaining $t$-statistics are $O_p(1)$, "hence, under Bonferroni control, no hypothesis is rejected with probability $\to1$".

*Why it fails.* $O_p(1)$ statistics compared with a *fixed* cutoff (Bonferroni at fixed $\alpha$ and $m$) exceed it with a probability that does not vanish: it converges to the size of the final-round Bonferroni test, which can be as large as about $\alpha$. So even when every other step is granted, the correct statement is $\liminf\mathbb{P}(\text{exact})\ge1-\alpha$ — the form of NEXIS's Theorem 4.1 and of Theorem 11 here. (Saved null runs, ATE $=0$: 1 of 80 runs selected an atom. This is consistent with an error rate of order $\alpha$; it neither proves nor disproves the limit claim, which is a mathematical error.)

*DINES.* States $\liminf_n\mathbb{P}(\hat S_n=\mathrm{Adj}_Z(T))\ge1-\alpha$ and proves it.

### 8.6 No backward step

*Claim (Algorithm 1).* Atoms are only added.

*Why it fails.* An atom admitted before the atoms that explain it is never re-tested. N_408 (§8.2) enters before N_6051; once N_6051 is in, N_408's remaining effect may vanish, but NES never checks. NEXIS showed the same point for its own forward step and fixed it with a terminal backward rule.

*DINES.* Terminal backward step over all subsets at level $\alpha/m$, with a fixed-null precision proof (Theorem 11).

### 8.7 The residualization in the paper is not run by the code

*Claim (App. A, "Residualization (optional)"; Algorithm 2 lines 2–10).* Arm-wise residualization of the tested atom on the selected atoms improves efficiency.

*What the code does.* In `neural_effect_search_base`, the residualized matrix `Y_mat` is computed (lines 205–224) but the stratified branch `test == "AF"` — the only branch NES uses once at least one atom is selected — calls `_stratified_ttests_lotp(data, sn, ...)` on the raw data frame (line 283), not on `Y_mat`. In the first round `sn` is empty and the residualization block is skipped. So the option never affects NES's output. (Had it been used with an unstratified contrast, within-arm OLS with an intercept centres each arm's residuals at zero, which removes the arm difference by construction.)

*DINES.* One test (cross-fitted GCM/PCM), used identically in the theory and the code.

### 8.8 Binarized strata and dropped cells

*Code.* Strata are built from selected atoms binarized at $0$ (`bin_threshold=0.0`), $2^{|S|}$ cells; cells without both arms are dropped and weights renormalized (lines 73–95).

*Why it matters.* Leakage of mediated effects (§7.2); and because the atoms are post-treatment, stronger effects produce more single-arm cells, so the estimand drifts with effect size.

*DINES.* Continuous conditioning by regression; overlap is reported, not silently trimmed.

### 8.9 The measured symptom: precision falls with $N$

Saved CelebA runs (Bonferroni-NES; ground truth = atoms 38 and 6051). At ATE $0.7$ (10 seeds): precision $0.80$ at $N=250$, $0.62$ at $500$, $0.56$ at $1000$, $0.45$ at $2500$, $0.49$ at $5000$; mean selection size $2.7\to4.2$; recall $1.0$ from $N=250$. Pooled over ATE $0.1$–$0.8$ (80 runs per $N$): precision $0.656$ ($N=250$) $\to0.566$ ($N=5000$), mean $|S|$ $1.58\to3.85$, exact recovery $26/80\to11/80$. (Marginal Bonferroni at ATE $0.7$: precision $0.28\to0.03$, mean $|S|$ $7.5\to77.3$ — the paradox.)

**Hypothesis (empirical, untested).** The fall has several causes that this report identifies separately: (a) magnitude ordering admits mixed atoms early (N_408); (b) no backward step keeps them; (c) binarized strata leak mediated and shared signal; (d) approximate alignment makes leaked atoms truly adjacent in the dictionary (§7.8), so their number grows with $N$; (e) **the benchmark design violates Tier 2**: Smiling is a co-parent of Eyeglasses (§3.4), but no atom is aligned with Smiling (the best-F1 atom for Smiling is the *hat* atom 38, F1 $0.48$). By §4.3 (a), atoms reading both eyeglasses and smiling are then genuinely direct in this dictionary. DINES fixes (a)–(c) by construction; (d) and (e) would affect DINES too. The benchmark in §9.2 separates them.

### 8.10 Summary table

| NES element (paper / code) | Problem | DINES |
|---|---|---|
| Target "coarsest abstraction" (§2) | undefined; Markov-blanket sufficiency mixes in spouses; proof target forbids mediation (Eq. 28) | $\mathrm{Adj}_Z(T)$ (Def. 1), causal reading Thm 6, mechanisms Prop 8, indirect output §5 |
| Raw $\lvert\hat\tau\rvert$ ordering, Principal–Max (Eqs. 25–26, 31; code l. 303) | scale-dependent; "argmax stays principal" false (counterexample §8.2); N_408 picked before N_6051 | $p$-value ordering; guarantee independent of ordering |
| A.3 (Eqs. 27–28) | assumes the disentanglement it should prove; false under mediation, colliders, noisy proxies, binning | structural A1–A3 with proved consequences |
| Transport error (Eqs. 29–30) | identically zero in an RCT; bound vacuous; real post-treatment bias unaddressed | conditional-independence formulation |
| "FWER → 0" (App. A.3 Note) | false at fixed $\alpha$ | $\liminf\mathbb{P}(\text{exact})\ge1-\alpha$ (Thm 11) |
| No backward step (Alg. 1) | early wrong atoms never removed | terminal subset rule at $\alpha/m$ |
| Binarized strata, dropped cells (Alg. 2; code l. 36, 73–95) | ~36% of mediated effect retained; estimand drifts | cross-fitted GCM/PCM, continuous conditioning |
| Residualization (App. A; Alg. 2) | never executed by the code (l. 205–224 vs 283) | one test, same in theory and code |
| i.i.d. units | frames are clustered | cluster-robust GCM, cluster folds |

---

## 9. Open questions, conjectures, and what the benchmark must test

### 9.1 Open questions and conjectures

- **O1 (separators without Tier 2).** For a general dictionary whose atom-level independence structure is an ancestral graph with $T$ a root, does every non-adjacent atom have a separating set inside every Markov blanket of $T$? Lemma 5's $D^\star$ can fail there (collider paths through blanket members); another subset might work. Unknown. Theorem 11 therefore keeps P as an assumption.
- **O2 (uniformity).** All results are pointwise in the distribution at fixed $m$. Uniform guarantees need a strong-faithfulness margin (Zhang & Spirtes, 2003; Uhler et al., 2013), and are impossible under plain faithfulness (Robins et al., 2003).
- **O3 (large $m$).** Dictionaries have $m\approx10^4$ atoms with $n\approx10^3$. The union bound over $m$ asymptotically valid $p$-values at level $\alpha/m$ needs error rates in the normal approximation that are $o(1/m)$; a growing-$m$ theory is open.
- **O4 (carriers in general).** An observable characterization of $\mathrm{Car}(j)$ without N1 (child-to-child or child-to-spouse paths).
- **O5 (choosing $\delta$).** A principled scale for the relevance margin.
- **Conjecture 1 (leakage bound).** Under approximate alignment, the conditional effect of a non-direct atom given the population separator is bounded by (the effect carried through the imperfectly aligned factor) × (the share of that factor's variance not explained by its principal atom). In the linear-Gaussian chain this is exact: $\gamma\beta\,\sigma_\eta^2/(\sigma_1^2+\sigma_\eta^2)$ (§4.3 (c)). A general version would justify a $\delta$ tied to measured alignment quality (e.g. probe $R^2$ of each principal atom).
- **O6 (feature splitting).** When a factor is determined only by a *group* of atoms, A3 fails for each member. The natural target is then a set of atom groups; not developed.
- **O7 (dictionary trained on the same data).** §7.9.
- **O8 (observational data).** §7.4.
- **O9 (distributional effects).** The mean null misses effects on spread only (§3.3).
- **O10 (checking A4).** Forward completeness is not testable directly; how often it fails for real dictionaries is unknown.

### 9.2 What the CelebA benchmark must test

Each design is generated as in NES (attributes sampled from a structural model, a real image matched to the sampled attributes), and reports, as functions of $N$ and of effect size: exact recovery rate (against the line $1-\alpha$), precision, recall, size of $\tilde S$, and, where relevant, the indirect output and its carriers. Compare DINES ($\delta=0$ and $\delta>0$), NES, the IAMB forward phase alone (a Markov-boundary estimate), NEXIS-style forward + terminal on the same tests, and marginal Bonferroni.

1. **Original NES design** ($T\to$ Eyeglasses, $T\to$ Hat, Smiling $\to$ Eyeglasses). Expect the forward phase to pick up spouse-like atoms; the question is whether DINES's precision stays near $1-\alpha$ or falls with $N$. Since no Smiling atom exists (§8.9), also run a variant where the spouse is an attribute with a well-aligned atom.
2. **Mediator** $T\to Y_1\to Y_2$ ($Y_2$ sampled from $Y_1$ only). Truth: direct $\{Y_1\}$, indirect $\{Y_2\}$ with carrier $Y_1$.
3. **Spouse** $T\to Y_1\leftarrow S$ ($S$ pre-treatment, well aligned), and the **descendant spouse** $T\to A\to\dots\to S\to C\leftarrow T$ that needs Lemma 5. Truth: direct = children only; Markov boundary adds $S$.
4. **Hidden confounder / hidden co-parent** $U\to Y_1$, $U\to Y_2$, $T\to Y_1\to Y_2$ with $U$ an attribute that has no aligned atom (or is not visible). Expected: $Y_2$'s atom becomes direct (§4.3 (a)) — a check that the Tier 2 assumption is necessary in practice.
5. **Approximate-alignment stress.** Weaker dictionaries (smaller codes, fewer active atoms, or attributes whose best atom has F1 around $0.5$). Expected: precision falls with $N$ at $\delta=0$; test whether $\delta>0$ restores it and at what recall cost.
6. **Null design** (ATE $=0$): rate of runs with any selection, to be compared with $\alpha$.
7. **Strong effects / positivity.** Two children, one with effect near deterministic: check that the second child is lost (§4.3 (e)(iii)).
8. **Clustered version** (several images per synthetic "video" sharing a cluster effect): check calibration of the cluster-robust test against the naive one.

### 9.3 Things in the brief we think are wrong or need care

Collected here so they are easy to review: (1) the alignment set needed is the Markov blanket of $T$, not all affected factors and their co-parents (§3.2); (2) "no hidden common cause" matters only as "no hidden co-parent of a child" (§3.2, §4.3 (a)); (3) factor-level faithfulness does not imply the atom-level condition the proof needs (§3.2, §4.3 (e)); (4) "nearly determines" does not suffice for the theorem (§4.3 (c)); (5) purity of spouse atoms is an extra requirement (§4.3 (d)); (6) ordering by $p$-value is a finite-sample choice, not a requirement of the guarantee (§6.1); (7) with the GCM, "mean faithfulness" must hold in the overlap-weighted sense (§4.3 (f)); (8) NES's own benchmark contains a spouse without an aligned atom (§8.9).

### 9.4 Reproducibility of the numbers in this report

All scripts were run once; see Appendix B. Nothing in §§1–8 depends on numbers we did not measure, except the brief's description of NES precision, which we re-measured (§8.9).

---

## 10. References

- Aliferis, C. F., Statnikov, A., Tsamardinos, I., Mani, S., Koutsoukos, X. D. (2010). Local causal and Markov blanket induction for causal discovery and feature selection for classification, Parts I–II. *JMLR* 11.
- Cadei, R., Otchere, F., Tirivayi, N., Angeles Tagliaferro, G., Bargagli-Stoffi, F. J., Locatello, F. (2026). From Tokens to Policy: Causal and Interpretable Heterogeneous Treatment Effects Identification (NEXIS). arXiv 2606.17010.
- Cameron, A. C., Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *J. Human Resources* 50(2).
- Lauritzen, S. L., Dawid, A. P., Larsen, B. N., Leimer, H.-G. (1990). Independence properties of directed Markov fields. *Networks* 20.
- Lin, W. (2013). Agnostic notes on regression adjustments to experimental data. *Ann. Appl. Stat.* 7(1).
- Lundborg, A. R., Kim, I., Shah, R. D., Samworth, R. J. (2024). The projected covariance measure for assumption-lean variable significance testing. *Ann. Statist.*
- Mencattini, T., Cadei, R., Locatello, F. (2025). Exploratory Causal Inference (NES). arXiv 2510.14073.
- Mooij, J. M., Magliacane, S., Claassen, T. (2020). Joint causal inference from multiple contexts. *JMLR* 21.
- Pearl, J. (2009). *Causality*, 2nd ed. Cambridge University Press.
- Peña, J. M., Nilsson, R., Björkegren, J., Tegnér, J. (2007). Towards scalable and data efficient learning of Markov boundaries. *Int. J. Approx. Reasoning* 45(2).
- Perry, R., von Kügelgen, J., Schölkopf, B. (2022). Causal discovery in heterogeneous environments under the sparse mechanism shift hypothesis. *NeurIPS*.
- Richardson, T., Spirtes, P. (2002). Ancestral graph Markov models. *Ann. Statist.* 30(4).
- Robins, J. M., Scheines, R., Spirtes, P., Wasserman, L. (2003). Uniform consistency in causal inference. *Biometrika* 90(3).
- Scheidegger, C., Hörrmann, J., Bühlmann, P. (2022). The weighted generalised covariance measure. *JMLR* 23.
- Schölkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., Bengio, Y. (2021). Toward causal representation learning. *Proc. IEEE* 109(5).
- Shah, R. D., Peters, J. (2020). The hardness of conditional independence testing and the generalised covariance measure. *Ann. Statist.* 48(3).
- Spirtes, P., Glymour, C., Scheines, R. (2000). *Causation, Prediction, and Search*, 2nd ed. MIT Press.
- Tsamardinos, I., Aliferis, C. F., Statnikov, A. (2003). Algorithms for large scale Markov blanket discovery (IAMB). *FLAIRS*.
- Tsiatis, A. A., Davidian, M., Zhang, M., Lu, X. (2008). Covariate adjustment for two-sample treatment comparisons in randomized clinical trials. *Stat. Med.* 27.
- Uhler, C., Raskutti, G., Bühlmann, P., Yu, B. (2013). Geometry of the faithfulness assumption in causal inference. *Ann. Statist.* 41(2).
- Zhang, J., Spirtes, P. (2003). Strong faithfulness and uniform consistency in causal inference. *UAI*.

---

## Appendix A. Index of results

| Label | Statement | Status |
|---|---|---|
| Lemma 1 | forms of the mean null; GCM functional | proved |
| Lemma 2 | conditioning on a determining atom | proved |
| Lemma 3 | aligned blanket screens off everything | proved |
| Lemma 4 | separation transfers to atoms (needs purity) | proved |
| Lemma 5 | separator $D^\star(s)$ inside $\mathrm{MB}(T)$ for a root $T$ | proved |
| Theorem 6 | $\mathrm{Adj}_Z(T)=J_C$ under A1–A3, with fixed separators | proved |
| Proposition 7 | Markov boundary = children + spouses | proved |
| Proposition 8 | direct set = minimal set of shifted mechanisms | proved |
| §4.3 (a)–(f) | necessity counterexamples | proved by computation |
| Proposition 10 | carriers identified under N1, CF | proved |
| Theorem 11 | $\liminf\mathbb{P}(\hat S_n=\mathrm{Adj}_Z(T))\ge1-\alpha$ under P, A5 | proved |
| Corollary 12 | causal version under A1–A5 | proved |
| Proposition 13 | full-output guarantee $\ge1-\alpha-\alpha_2$ | proved |
| Proposition 14 | covariates do not change the target | proved |
| Conjecture 1 | leakage bound under approximate alignment | conjecture |
| §8.9 | causes of NES's precision fall | hypothesis (empirical) |
| O1–O10 | | open |

## Appendix B. Measurements used in this report

All runs on Slurm CPU nodes (none on the login node beyond sub-minute checks).

**B.1 Toy DINES simulation.** Factor DAG $T\to A\to B\to W\to C\leftarrow T$, $H\to C$, $A\to D$, $L$ isolated; $T\sim\mathrm{Bern}(1/2)$; $A=0.5T+\varepsilon$, $B=0.8A+\varepsilon$, $W=0.8B+\varepsilon$, $D=0.8A+\varepsilon$, $C=0.5T+0.8W+0.8H+\varepsilon$, $H,L,\varepsilon\sim\mathcal{N}(0,1)$. Atoms: principal atoms of $A,B,W,C,D,H,L$ (each equal to its factor plus $\sigma_\eta\,\mathcal{N}(0,1)$), and three non-principal atoms $0.5A+0.5D+\varepsilon$, $0.5C+0.5H+\varepsilon$, $L+\varepsilon$; $m=10$. Truth: direct $\{A,C\}$, Markov blanket $\{A,C,W,H\}$. Test: GCM with 2-fold cross-fitted logistic propensity and OLS regression (with the fitted propensity as an extra feature, which makes both models correctly specified in this model); $\alpha=0.05$; forward gate $\alpha/|R|$; terminal $\alpha/m$. With $\delta>0$ the relevance test of §7.8 replaces the exact test in both phases. 100 seeds per cell.

| alignment | $\delta$ | $n$ | exact recovery | mean non-direct atoms in $\tilde S$ | mean false atoms in $\hat S$ | recall |
|---|---|---|---|---|---|---|
| exact ($\sigma_\eta=0$) | 0 | 1000 | 99/100 | 1.86 | 0.00 | 0.99 |
| exact | 0 | 4000 | 99/100 | 2.10 | 0.01 | 1.00 |
| exact | 0 | 16000 | 99/100 | 2.03 | 0.01 | 1.00 |
| approximate ($\sigma_\eta=0.3$) | 0 | 1000 | 96/100 | 1.56 | 0.01 | 0.98 |
| approximate | 0 | 4000 | 96/100 | 2.15 | 0.04 | 1.00 |
| approximate | 0 | 16000 | 84/100 | 2.36 | 0.17 | 1.00 |
| approximate | 0.05 | 1000 | 70/100 | 0.35 | 0.01 | 0.85 |
| approximate | 0.05 | 4000 | 100/100 | 1.64 | 0.00 | 1.00 |
| approximate | 0.05 | 16000 | 100/100 | 2.00 | 0.00 | 1.00 |
| exact | 0.05 | 1000 | 70/100 | 0.56 | 0.00 | 0.84 |
| exact | 0.05 | 4000 | 100/100 | 1.94 | 0.00 | 1.00 |
| exact | 0.05 | 16000 | 100/100 | 2.00 | 0.00 | 1.00 |

False atoms kept at $\sigma_\eta=0.3$, $\delta=0$, $n=16000$: mixed atom $0.5A+0.5D$ (10 runs), $D$ (4), $W$ (3) — all downstream of the noisy proxy of $A$. Separating sets (exact alignment, $n=20{,}000$, one run): $T$ vs $W$: $p=3.6\times10^{-70}$ given $\emptyset$; $0.14$ given $\{A\}$; $0.15$ given $\{A,H\}$; $0.37$ given $\{A,B\}$; $2.2\times10^{-56}$ given $\{A,C\}$; $9.8\times10^{-122}$ given $\{A,C,H\}$. $T$ vs $H$: $0.68$ given $\emptyset$; $1.1\times10^{-59}$ given $\{C\}$.

**B.2 Closed-form checks** ($n=2\times10^6$): binarized-stratum retained fraction (§7.2); noisy-proxy coefficient (§4.3 (c)); hidden co-parent coefficient (§4.3 (a)); Principal–Max counterexample (§8.2, $n=20{,}000$).

**B.3 Saved NES CelebA runs** (read-only): selections from `sneurons_{420..429}.json`; round-2 statistics recomputed with the project's own `_stratified_ttests_lotp` on `experiment_M3_ATE:{0.4,0.7}_size:{250,1000,5000}_seed:420.csv`; atom–attribute F1 computed on `experiment_M3_ATE:0.4_size:1000_seed:420.csv` (firing indicator vs attribute).
