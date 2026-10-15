# DINES — Direct–Indirect Neural Effect Search: theory report

*Draft 2, 2026-10-16 (R. Cadei; adversarial check by Codex). Replaces draft 1 (commit 07f7c11). Appendix A maps draft-1 result numbers to the new ones.*

**Labels.** *Theorem*, *Proposition*, *Lemma* and *Corollary* have a complete proof in this document. *Conjecture* = believed, not proved. *Open question* = unknown. "Measured" numbers come from the runs in Appendix B.

---

## 0. Summary (this page is self-contained)

**Setting.** A binary treatment $T$ is randomized. After treatment we observe an image or video frame and summarize it with a dictionary fixed in advance: $m$ atoms $Z=(Z^1,\dots,Z^m)$, for example sparse-autoencoder codes.

**Objective.** Find the atoms the treatment changes, excluding those that change only as a consequence of other changed atoms.

**Running example: the frog.** A treatment paralyses a leg ($P$) and changes skin colour ($C$). Movement speed ($V$) drops only because the leg is paralysed.
- The target is $\{P,C\}$, not $V$.
- If the treatment also made the frog lethargic, $V$ would change directly and should be reported.
- Marginal tests flag all three atoms. NES may flag $V$ first and never remove it (§4).

DINES holds atoms fixed and asks which shifts survive whatever is held fixed. Formally, the target is the atoms of the treatment's *children* in a latent causal model (§2). The statistical object is the **direct set** $\mathrm{Adj}$: the atoms whose mean $T$ shifts given every set of other atoms that leaves both arms represented (§1).

**This is a mediation question, and it has a price.** A single randomized treatment identifies *total* effects only. Splitting them into direct and indirect is a mediation question, and it needs untestable assumptions (Robins & Greenland, 1992; Pearl, 2001; sequential ignorability, Imai et al., 2010):
- **(a)** the mediator is recorded by the dictionary in the detail its consequences depend on. This fails if the atom says only "leg drags yes/no" while speed depends on the degree of paralysis;
- **(b)** there is no hidden common cause of a changed factor and its consequences. This fails if frog size affects both paralysis severity and speed and no atom records size.

If either fails, $V$ still differs between the arms when the paralysis atom is held fixed. The data are then exactly as compatible with "the treatment slows the frog directly" (Proposition 3). **This is the price of the question, not of DINES: any method that drops $V$ relies on (a) and (b), NES included.**

**Realistic version.** Exact alignment is unrealistic. NEXIS rejects it on CelebA at $p<10^{-14}$, while the principal atom still carries 94–96% of the held-out information. With an imperfect atom for $P$, a small part of $V$'s shift *leaks* through, and it is detected once $n$ is large. This is the likely cause of NES's precision falling with $N$ (0.80 → 0.49), and the same failure appears for DINES in our toy model. DINES therefore tests "$|\rho|\le\delta$", where $\rho$ is a partial correlation between $T$ and the atom given the conditioning atoms. It recovers the target when every leak is below $\delta$ and every direct effect is above it. The margin $\delta$ belongs to the algorithm and the assumption; the target does not change.

**Overlap.** Skin colour may reveal the treatment. Holding colour fixed then leaves no control frogs, and every other shift is "explained" trivially. The direct set is therefore defined over conditioning sets that leave both arms represented. With this definition:
- $P$ stays in the set;
- $V$ is still dropped, because holding $P$ fixed explains it;
- atoms that change only through colour cannot be told apart from atoms the treatment changes directly, because colour *is* the treatment as far as the data can tell. They are reported as "undetermined". If colour is an exact copy of $T$, this label also applies to $P$: "changed by $T$" and "changed by colour" are then the same statement in the data. This is a real identification limit (Proposition 4).

**Why it works (proof sketch).**
- *Recall.* A direct atom is shifted given every overlap set, so consistent tests admit it and keep it.
- *Precision.* The forward phase reaches the atoms around $T$. Every non-direct atom has one **fixed** separating set among them, and the terminal phase tests it there. A union bound over at most $m$ fixed nulls gives error at most $\alpha$; no test at a data-chosen set needs to be valid.
- *Causal reading.*
  - Because $T$ is randomized, a non-child factor is cut off from $T$ by those blanket factors that are its ancestors (separator lemma, Lemma 3).
  - Aligned atoms let atom-level conditioning replace factor-level conditioning (Lemma 4).
  - So the direct set is the set of child atoms (Theorem 1).
- *The δ version.* The same argument applies with the null $|\rho|\le\delta$ (Theorem 3).

**Clean formulation.**
- **Objective.** The atoms the treatment changes, excluding those that change only as a consequence of other changed atoms.
- **Assumptions.**
  1. $T$ is randomized.
  2. *Alignment* with the changed factors and the needed conditioning factors, in the NEXIS sense. Each has a principal atom that screens it off: once the principal atom is known, no other atom tells more about the factor. The atom of an "other cause" must also read nothing on the treatment side. In the approximate version, leak $<\delta<$ direct effect.
  3. *No hidden common cause* of a changed factor and its consequences.
  4. *No exact cancellation and valid tests.* Direct atoms stay shifted given every overlap set. The search never stalls on an exact cancellation. The regressions used to hold atoms fixed are learned well enough, on a dictionary fixed before the analysis.
- **Guarantee.** As $n\to\infty$ with $m$ fixed, DINES returns exactly the atoms of the directly changed factors with probability at least $1-\alpha$, and with probability tending to 1 when leaks are strictly below $\delta$ (Corollary 2, Theorem 3).
- **What you still get without 2–3.**
  - Total effects are always valid.
  - Every directly changed factor whose atom stays shifted is reported (recall needs only assumption 4).
  - The output is the dictionary-relative direct set $\mathrm{Adj}$, with error at most $\alpha$. This holds provided the atoms around $T$ suffice to explain away the rest (assumption L, §2.6), which is not automatic for an arbitrary dictionary.
  - But an atom reported as direct may be a consequence seen through a coarse or noisy mediator atom, or a consequence confounded by an unrecorded common cause. No method can tell the difference from these data.

---

## 1. Setup and the target

- **Units.** $n$ i.i.d. units (for video: i.i.d. clusters, §3.4). Treatment $T\in\{0,1\}$ with $\mathbb{P}(T=1)=e\in(0,1)$ fixed by design. Optional pre-treatment covariates $W$ (so $T\perp\!\!\!\perp W$).
- **Dictionary.** $\psi$ is fixed (not fitted on the analysed units; §3.6). $Z=\psi(X)\in\mathbb{R}^m$, $[m]=\{1,\dots,m\}$, $Z^A=(Z^k)_{k\in A}$, and $Z^\emptyset$ is a constant. Atoms have finite second moments.
- **Conditional quantities**, for $A\subseteq[m]$ and $j\notin A$:
  - propensity given atoms: $\pi_A:=\mathbb{P}(T=1\mid Z^A)$;
  - atom regression: $\mu_{j,A}:=\mathbb{E}[Z^j\mid Z^A]$;
  - arm contrast: $\Delta_{j,A}:=\mathbb{E}[Z^j\mid T=1,Z^A]-\mathbb{E}[Z^j\mid T=0,Z^A]$ on $\{0<\pi_A<1\}$, and $0$ elsewhere;
  - overlap of $A$: $V_A:=\mathbb{E}[\pi_A(1-\pi_A)]$.
- **Mean null.** $H_0(j\mid A):\ \mathbb{E}[Z^j\mid T,Z^A]=\mathbb{E}[Z^j\mid Z^A]$ a.s. With covariates, $Z^A$ is replaced by $(Z^A,W)$ everywhere.
- **Effect scale (partial correlation).**
  $$\rho_{j|A}:=\frac{\mathbb{E}[(T-\pi_A)(Z^j-\mu_{j,A})]}{\sqrt{\mathbb{E}[(T-\pi_A)^2]\ \mathbb{E}[(Z^j-\mu_{j,A})^2]}}\qquad(:=0\text{ if a factor of the denominator is }0).$$
  **Margin null:** $H_0^\delta(j\mid A):\ |\rho_{j|A}|\le\delta$, for $\delta\ge0$.
- **Overlap sets.** $\mathcal{O}:=\{A\subseteq[m]:\ 0<\pi_A<1\text{ a.s.}\}$, the conditioning sets that leave both arms possible for almost every unit. Properties:
  - $\emptyset\in\mathcal{O}$;
  - $\mathcal{O}$ is closed under subsets: if $A\subseteq A'$, then $\pi_A=\mathbb{E}[\pi_{A'}\mid Z^A]$ lies in $(0,1)$ when $\pi_{A'}$ does.
- **Graph terms** (DAG $\mathcal{G}$): $\mathrm{Pa},\mathrm{Ch}$; $\mathrm{An}(Y)$, the ancestors of $Y$ including $Y$; $\mathrm{De}$, *strict* descendants. $\mathrm{Sp}(T):=\bigcup_{c\in\mathrm{Ch}(T)}\mathrm{Pa}(c)\setminus\{T\}$ are the *other causes* of $T$'s children ("spouses"). $\mathrm{MB}(T):=\mathrm{Ch}(T)\cup\mathrm{Sp}(T)$ is the Markov blanket of a root $T$. $\perp_d$ denotes d-separation.

**Lemma 1 (forms of the null; the effect scale).** Fix $A$ and $j\notin A$.
1. $\mathbb{E}[Z^j\mid T,Z^A]-\mu_{j,A}=(T-\pi_A)\Delta_{j,A}$ a.s.
2. $H_0(j\mid A)$ holds iff $\pi_A(1-\pi_A)\Delta_{j,A}=0$ a.s. In particular, if $\pi_A\in\{0,1\}$ a.s., then $H_0(j\mid A)$ holds for every $j$.
3. $\mathbb{E}[(T-\pi_A)(Z^j-\mu_{j,A})]=\mathbb{E}[\pi_A(1-\pi_A)\Delta_{j,A}]$.
4. $T\perp\!\!\!\perp Z^j\mid Z^A$ implies $H_0(j\mid A)$; the converse is false.
5. $\rho_{j|A}=\mathbb{E}[\pi_A(1-\pi_A)\Delta_{j,A}]\big/\sqrt{V_A\,\mathbb{E}[\mathrm{Var}(Z^j\mid Z^A)]}$, $|\rho_{j|A}|\le1$, and $H_0(j\mid A)\Rightarrow\rho_{j|A}=0$; the converse is false.

*Proof.* (1) On $\{0<\pi_A<1\}$ let $m_t=\mathbb{E}[Z^j\mid T=t,Z^A]$. Then $\mathbb{E}[Z^j\mid T,Z^A]=Tm_1+(1-T)m_0$ and $\mu_{j,A}=\pi_Am_1+(1-\pi_A)m_0$, whose difference is $(T-\pi_A)(m_1-m_0)$. On $\{\pi_A=1\}$, $T=1$ a.s., so $\mathbb{E}[Z^j\mid T,Z^A]=\mu_{j,A}$ there; likewise on $\{\pi_A=0\}$.

(2) Given $Z^A$ on $\{0<\pi_A<1\}$, $T-\pi_A$ takes the two non-zero values $1-\pi_A$ and $-\pi_A$ with positive probability, so $(T-\pi_A)\Delta=0$ a.s. iff $\Delta=0$ a.s. there.

(3) $\mathbb{E}[(T-\pi_A)\mu_{j,A}]=0$, and $\mathbb{E}[(T-\pi_A)Z^j]=\mathbb{E}[(T-\pi_A)(\mu_{j,A}+(T-\pi_A)\Delta)]=\mathbb{E}[\pi_A(1-\pi_A)\Delta]$.

(4) Equal conditional laws give equal conditional means. Counterexample to the converse: $Z^j\mid T\sim\mathcal{N}(0,1+T)$ with $A=\emptyset$.

(5) $\mathbb{E}[(T-\pi_A)^2]=V_A$ and $\mathbb{E}[(Z^j-\mu_{j,A})^2]=\mathbb{E}\mathrm{Var}(Z^j\mid Z^A)$. The numerator is given by (3), and Cauchy–Schwarz gives $|\rho|\le1$. Under $H_0$ the numerator is $0$ by (2). The converse fails when $\Delta$ changes sign and averages to zero under the weights $\pi_A(1-\pi_A)$. $\square$

**Definition 1 (direct, total and indirect sets).**
$$\mathrm{Adj}_Z(T):=\{j:\ H_0(j\mid A)\text{ false for every }A\in\mathcal{O}\text{ with }j\notin A\},\qquad\mathrm{De}_Z(T):=\{j:\ H_0(j\mid\emptyset)\text{ false}\},$$
and $\mathrm{Ind}_Z(T):=\mathrm{De}_Z(T)\setminus\mathrm{Adj}_Z(T)$.

*Reading.*
- **Only overlap sets count.** By Lemma 1.2, a set that reveals $T$ satisfies every null vacuously, so it carries no information about the direct/indirect split. Draft 1 used every subset, so a single treatment-revealing atom emptied the direct set. The two definitions agree when every set has overlap.
- **"Direct" is relative to the dictionary**, just as adjacency in causal discovery is relative to the measured variables.
- **The null is about the mean.** For a zero-inflated code, $\mathbb{E}Z^j=\mathbb{P}(Z^j>0)\,\mathbb{E}[Z^j\mid Z^j>0]$, so a mean test sees changes in both firing rate and firing magnitude. A change in spread alone is invisible to it.
- **What the GCM test sees.** The GCM test (§3.1) detects only $\rho\neq0$, so with it "$H_0$ false" is read as "$\rho\ne0$". The difference from the mean null lies in sign changes that cancel exactly, which assumption 4 excludes.

---

## 2. Formal results

### 2.1 The latent model and the two causal assumptions

**Assumption A1 (latent model).** There are finite latent *factors* $L=(L_1,\dots,L_r)$ and an acyclic structural causal model over $\{T\}\cup L$ with mutually independent noises, in which $T$ is a root (randomization). Its DAG is $\mathcal{G}_L$. The atoms are $Z=H(L,N^Z)$ with $N^Z\perp\!\!\!\perp(T,L)$. The measurement noise may be dependent across atoms. $P(j)\subseteq L$ denotes a minimal set of factors read by atom $j$. The **changed factors** are $\mathrm{Ch}(T)$. Their consequences are their descendants.

**Assumption 3 (no hidden common cause; coverage).** There is a description satisfying A1 in which every factor of $\mathrm{MB}(T)$ — each changed factor and each other cause of a changed factor — has its own principal atom $j_k$, distinct for distinct $k$. We write $J_C:=\{j_c:c\in\mathrm{Ch}(T)\}$, $B_Z:=\{j_k:k\in\mathrm{MB}(T)\}$, and $J_D:=\{j_k:k\in D\}$.

**Assumption 2 (alignment, NEXIS screening form).** Write $D_s:=(\mathrm{MB}(T)\cap\mathrm{An}(s))\setminus\{s\}$ for each other cause $s\in\mathrm{Sp}(T)\setminus\mathrm{Ch}(T)$.
- (2a) *Screening off:* $L_D\perp\!\!\!\perp Z^{[m]\setminus J_D}\mid Z^{J_D}$ for $D=\mathrm{MB}(T)$ and for every $D=D_s$. Once the principal atoms are known, no other atom tells more about the factors.
- (2b) *Purity on the treatment side:* for each such $s$, the atoms $j_s$ and $J_{D_s}$ read only their own factors: $Z^{j_k}=h_k(L_k,N^Z)$.

Assumption 2a is NEXIS's Principal Alignment (its Eq. 6, $W^{\mathrm{dir},k}\perp\!\!\!\perp Z^{[m]\setminus\{j_k\}}\mid Z^{j_k}$) combined with its composition clause, applied to the changed factors and to the conditioning factors DINES needs. *Exact capture* means determination ($L_k=a_k(Z^{j_k})$) plus purity for every $k\in\mathrm{MB}(T)$. It implies assumption 2 for every $D$, and it is the convenient sufficient condition used in §2.10–2.12 and §3.2.

**Lemma 2 (folding: which other causes need an atom).** In a description satisfying A1:
- (i) a factor that no atom reads and that has no children can be deleted;
- (ii) a factor $s$ that no atom reads and that has exactly one child $c$ can be folded into $c$: $L_c=f_c(L_{\mathrm{Pa}(c)\setminus\{s\}},f_s(L_{\mathrm{Pa}(s)},N_s),N_c)$, with noise $(N_s,N_c)$.

In both cases the result satisfies A1, $T$ is still a root, and the law of $(T,Z)$ is unchanged.

*Proof.* (i) Deleting an unread sink leaves all other equations unchanged. (ii) The new noise is used only by $c$, so the noises stay independent, and substitution preserves the joint law of the other variables. A new edge $p\to c$ with $p\in\mathrm{Pa}(s)$ would close a cycle only if $c\in\mathrm{An}(p)$, but then $c\to\dots\to p\to s\to c$ was already a cycle. $T$ receives no parent. $\square$

After folding, the only other causes that remain are those read by an atom or linked to the atoms by a second route. So assumption 3 asks for an atom only for a cause of a changed factor that also reaches other atoms some other way. In the frog, size is such a cause: it reaches speed directly. A hidden common cause is exactly a needed conditioning factor without an atom. Folding an unread changed factor into its child makes that child "changed", in line with directness being relative to the dictionary. (The lethargy variant works this way: lethargy is unrecorded, so speed becomes a changed factor.)

### 2.2 Comparison with NEXIS: what changes when conditioning is post-treatment

- **Why NEXIS needs less.** NEXIS conditions only on pre-treatment variables, which are independent of $T$ and not caused by it. Holding them fixed therefore never creates spurious arm differences. Here the conditioning atoms are post-treatment. Holding paralysis fixed compares treated frogs with control frogs that reached the same severity *for other reasons*, such as size. That explaining-away dependence is why assumption 3, and the alignment of other causes, are needed.
- **Assumption (b).** NEXIS's analogue of (b) is built into its target: $W^{\mathrm{dir}}$ is *defined* as the direct modifiers, and $\tau\perp\!\!\!\perp Z\mid W^{\mathrm{dir}}$ is asserted. Here (b) must be stated (assumption 3), because the factor structure around $T$ is part of what is inferred.
- **Assumption (a)** is Principal Alignment applied to the changed and needed conditioning factors (assumption 2a).
- **Do the proofs need more than screening off?**
  - *For atoms explained away by the whole blanket: no.* Screening off, even in its mean form, is exactly what the proof uses, through contraction (Lemma 4a).
  - *For other-cause atoms and reduced separators: purity as well.* NEXIS never needs purity, because it never conditions on post-treatment variables. Example (d) of §2.3 shows that purity cannot be dropped here.
  - *Screening off is not weaker than determining the relevant part.* With independent measurement noise, screening off holds iff the principal atom determines the part of the factor that other atoms' means depend on (Lemma 5). The coarse "leg drags yes/no" atom violates it whenever speed depends on the degree of paralysis. Screening off is, however, weaker than full determination: the paralysis atom need not record aspects of paralysis on which nothing else depends.
- **Approximate version.** NEXIS's own CelebA evidence motivates the realistic version. Exact screening is rejected at $p<10^{-14}$, while the principal atom carries 94–96% of the held-out information. Hence the condition "leak $<\delta<$ direct effect" (Theorem 3), and Propositions 5–6, which say how large a leak is.

### 2.3 What breaks without each assumption (frog versions; each example satisfies everything else)

All noises are independent standard Gaussians unless stated, and $T\sim\mathrm{Bern}(1/2)$.

- **(a) Hidden common cause (fails assumption 3).** Size $S$, paralysis severity $P=\beta T+S+\varepsilon_1$, speed $V=-\gamma P+S+\varepsilon_2$, atoms $Z^1=P$ and $Z^2=V$, no size atom. Since $\mathbb{E}[S\mid T,P]=\tfrac12(P-\beta T)$,
  $$\mathbb{E}[Z^2\mid T,Z^1]=(\tfrac12-\gamma)P-\tfrac{\beta}{2}T,$$
  so $V$ is direct at every $n$ (measured coefficient $-0.2475$ at $\beta=0.5$; formula $-0.25$). A size atom, or size measured as a covariate (Proposition 10), restores $\mathbb{E}[Z^2\mid T,Z^1,S]=-\gamma P+S$.
- **(b) Coarse mediator atom (fails 2a).** $Z^1=\mathbb{1}[P>0]$ ("leg drags yes/no"), $V=-\gamma P+\varepsilon$. Within each stratum the arm difference in mean paralysis is $\beta+\lambda(\beta)-\lambda(0)$, with $\lambda=\varphi/\Phi$, which is about $(1-2/\pi)\beta\approx0.36\beta$ for small $\beta$. About 36% of $V$'s mediated shift survives (measured retained fraction 0.366, 0.379, 0.418 at $\beta=0.2,0.5,1.0$). This is exactly NES's binarized strata.
- **(c) Noisy mediator atom (fails 2a).** $P=\beta T+\sigma_1\varepsilon_1$, $Z^1=P+\sigma_\eta\eta$, $V=\gamma P+\varepsilon_2$. The coefficient on $T$ in $\mathbb{E}[Z^2\mid T,Z^1]$ is $\gamma\beta\sigma_\eta^2/(\sigma_1^2+\sigma_\eta^2)\neq0$, a *leak* (measured $0.0323$ at $\beta=0.5$, $\gamma=0.8$, $\sigma_1=1$, $\sigma_\eta=0.3$; formula $0.0330$).
- **(d) Impure atom of an other cause (fails 2b, though screening holds).** $T\to P\leftarrow S$ with $S\in\{0,1\}$, and a size atom that also reads paralysis: $Z^S=S+\tfrac12\mathrm{sigmoid}(P)$. It determines $S$ ($S=\lfloor Z^S\rfloor$), so it screens $S$ off from everything. Yet $Z^S\in\mathrm{Adj}$. Given $Z^P$ it is a function of $S$, and $T\not\perp\!\!\!\perp S\mid P$ (explaining away); given $\emptyset$ it reads $P$.
- **(e) Exact cancellation (fails 4).** $F_2=aT+\gamma F_1+\varepsilon_2$ with $a=-\gamma\beta$ makes a changed factor invisible. Redundant atoms explain each other away (§2.10).
- **(f) GCM blind spot.** If $\Delta_{j,A}$ changes sign across values of $Z^A$, the GCM has no power at any $n$. The PCM (§3.1) removes this.

### 2.4 Identification

**Lemma 3 (separator lemma).** Let $T$ have no parents in a DAG $\mathcal{G}$.
- (a) For any set $Y\not\ni T$, put $D:=\mathrm{MB}(T)\cap\mathrm{An}(Y)$. Then $T\perp_d Y\setminus D\mid D$.
- (b) For any $s\notin\mathrm{Ch}(T)\cup\{T\}$, put $D_s:=(\mathrm{MB}(T)\cap\mathrm{An}(s))\setminus\{s\}$. Then $T\perp_d s\mid D_s$.

*Proof.* We use the moralization criterion (Lauritzen et al., 1990): $X\perp_d Y\mid D$ iff $D$ separates $X$ from $Y$ in the moral graph of $\mathrm{An}(X\cup Y\cup D)$.

(a) Here $D\subseteq\mathrm{An}(Y)$ and $\mathrm{An}(T)=\{T\}$, so the ancestral set is $\mathcal{A}=\{T\}\cup\mathrm{An}(Y)$. Since $T$ has no parents, its moral neighbours in $\mathcal{A}$ are of two kinds.
- Its children $c\in\mathcal{A}$. Such a $c$ lies in $\mathrm{An}(Y)\cap\mathrm{MB}(T)=D$.
- Co-parents $p$ of a child $c\in\mathcal{A}$. Such a $p$ lies in $\mathrm{An}(c)\subseteq\mathrm{An}(Y)$ and in $\mathrm{MB}(T)$, hence in $D$.

Every path from $T$ to $Y\setminus D$ starts at a neighbour in $D$.

(b) Here $\mathcal{A}=\{T\}\cup\mathrm{An}(s)$.
- A child $c\in\mathcal{A}$ is in $\mathrm{An}(s)\cap\mathrm{MB}$ and differs from $s$ (because $s$ is not a child), so $c\in D_s$.
- A co-parent $p$ of a child $c\in\mathrm{An}(s)$ is in $\mathrm{An}(s)\cap\mathrm{MB}$. It differs from $s$, because $s\to c$ with $c\in\mathrm{An}(s)$ would be a cycle. So $p\in D_s$.

Since $s\notin D_s$, every moral path from $T$ to $s$ passes through $D_s$. $\square$

*Examples.*
- *Frog.* $\mathrm{MB}(T)\cap\mathrm{An}(V)=\{P\}$, or $\{P,S\}$ if size causes both $P$ and $V$. So $V$ is separated by $P$ (and size) and *not* by colour, which matters when colour reveals $T$.
- *Toy.* In $T\to A\to B\to W\to C\leftarrow T$, the other cause $W$ is also a descendant of $T$, and $D_W=\{A\}$. The set $\{A,C\}$ does *not* separate. Measured (linear-Gaussian, $n=20{,}000$, GCM): $p=0.14$ given $\{A\}$, $3.6\times10^{-70}$ given $\emptyset$, $2.2\times10^{-56}$ given $\{A,C\}$.
- *Draft 1's set.* Draft 1 used the larger $D^\star(s)=\mathrm{MB}(T)\setminus(\{s\}\cup\mathrm{De}(s))\supseteq D_s$, which also separates. The ancestral version is smaller and avoids changed factors that are not ancestors of $s$.

*The root matters.* For a non-root target, parents of $T$ are moral neighbours too. In $X\to P\to T\to C\leftarrow X$, blocking $X$ needs $P$. Randomization removes this case, which is the known incompleteness of MMPC-type rules (Peña et al., 2007).

**Lemma 4 (from factors to atoms).** Let $D\subseteq L$, let $J\subseteq[m]$, and let $Y$ be a random vector.
- Assume (i) $T\perp\!\!\!\perp Y\mid(L_D,Z^J)$ and (ii) $L_D\perp\!\!\!\perp Y\mid Z^J$ (screening). Then $T\perp\!\!\!\perp Y\mid Z^J$.
- *Mean version:* if (i′) $\mathbb{E}[Y\mid T,L_D,Z^J]=\mathbb{E}[Y\mid L_D,Z^J]$ and (ii′) $\mathbb{E}[Y\mid L_D,Z^J]=\mathbb{E}[Y\mid Z^J]$, then $\mathbb{E}[Y\mid T,Z^J]=\mathbb{E}[Y\mid Z^J]$.

Under A1, condition (i) holds in either of two cases:
- (a) $D=\mathrm{MB}(T)$ and $Y,Z^J$ are any functions of $(L,N^Z)$ (no purity needed);
- (b) $T\perp_d V\setminus D\mid D$ in $\mathcal{G}_L$, $Y=f(L_V,N^Z)$ and $Z^J=\phi(L_D,N^Z)$ (purity of $J$).

The lemma holds verbatim with $Z^J$ replaced by any vector $Q$ of functions of $(L,N^Z)$, such as atoms together with observed covariates. In (b), this requires $Q=\phi(L_D,N^Z)$.

*Proof.* Contraction applied to (ii) and (i) gives $Y\perp\!\!\!\perp(L_D,T)\mid Z^J$, hence $Y\perp\!\!\!\perp T\mid Z^J$. For the mean version: $\mathbb{E}[Y\mid T,Z^J]=\mathbb{E}[\mathbb{E}[Y\mid T,L_D,Z^J]\mid T,Z^J]=\mathbb{E}[\mathbb{E}[Y\mid Z^J]\mid T,Z^J]$.

(a) Every path from the root $T$ starts with $T\to c$. If $c$ is a non-collider, conditioning on $\mathrm{MB}$ blocks it. If $c$ is a collider, the next vertex is a co-parent in $\mathrm{MB}$ and a non-collider. Hence $T\perp_d L_{\rm rest}\mid L_{\mathrm{MB}}$, and by the Markov property and $N^Z\perp\!\!\!\perp(T,L)$, $T\perp\!\!\!\perp(L_{\rm rest},N^Z)\mid L_{\mathrm{MB}}$. So $T\perp\!\!\!\perp(Y,Z^J)\mid L_{\mathrm{MB}}$, and weak union gives (i).

(b) The Markov property gives $T\perp\!\!\!\perp L_V\mid L_D$. With $N^Z\perp\!\!\!\perp(T,L)$, $T\perp\!\!\!\perp(L_V,N^Z)\mid L_D$. So $T\perp\!\!\!\perp(Y,Z^J)\mid L_D$, and weak union gives (i). $\square$

**Lemma 5 (screening off = the principal atoms determine the relevant part).** Let $Z^J=\phi(L_D,\eta)$ with $\eta\perp\!\!\!\perp(L,N^j)$ (pure atoms, independent noise), and $Z^j=f(L_V,N^j)$ with $N^j\perp\!\!\!\perp(T,L)$. Write $g(L_D):=\mathbb{E}[Z^j\mid L_D]$. Then mean screening, $\mathbb{E}[Z^j\mid L_D,Z^J]=\mathbb{E}[Z^j\mid Z^J]$, holds iff $g(L_D)$ is a.s. a function of $Z^J$.

*Proof.* $\eta\perp\!\!\!\perp(L_V,N^j)\mid L_D$, so $Z^j\perp\!\!\!\perp Z^J\mid L_D$ and $\mathbb{E}[Z^j\mid L_D,Z^J]=g(L_D)$.
- If mean screening holds, then $g(L_D)=\mathbb{E}[Z^j\mid Z^J]$, a function of $Z^J$.
- Conversely, if $g(L_D)=h(Z^J)$, then $\mathbb{E}[Z^j\mid Z^J]=\mathbb{E}[g\mid Z^J]=h(Z^J)=g(L_D)$. $\square$

**Theorem 1 (identification).** Assume A1, assumptions 2 and 3, and *no cancellation at the children*: $H_0(j_c\mid A)$ is false for every $c\in\mathrm{Ch}(T)$ and every $A\in\mathcal{O}$ with $j_c\notin A$. Then:
1. $J_C\subseteq\mathrm{Adj}_Z(T)$.
2. Every $j\notin B_Z$ satisfies $T\perp\!\!\!\perp Z^j\mid Z^{B_Z}$.
3. Every other-cause atom $j_s$ ($s\in\mathrm{Sp}(T)\setminus\mathrm{Ch}(T)$) satisfies $T\perp\!\!\!\perp Z^{j_s}\mid Z^{J_{D_s}}$.
4. Suppose in addition that the atoms of $D_j:=\mathrm{MB}(T)\cap\mathrm{An}(P(j))$ are pure and that screening (2a) holds for $D_j$. Then every $j\notin B_Z$ also satisfies $T\perp\!\!\!\perp Z^j\mid Z^{J_{D_j}}$.

Consequently, a non-child atom with one of these separators in $\mathcal{O}$ is not in $\mathrm{Adj}$. If $B_Z\in\mathcal{O}$ (no set of blanket atoms reveals $T$), then
$$\mathrm{Adj}_Z(T)=J_C .$$

*Proof.*
1. This is the no-cancellation assumption.
2. Apply Lemma 4(a) with $D=\mathrm{MB}$, $J=B_Z$, $Y=Z^j$; screening is 2a.
3. Lemma 3(b) gives $T\perp_d s\mid D_s$. Apply Lemma 4(b) with $V=\{s\}$, using purity 2b ($Z^{j_s}=h_s(L_s,N^Z)$ and $Z^{J_{D_s}}=\phi(L_{D_s},N^Z)$) and screening 2a for $D_s$. Since $s\notin D_s$, $j_s\notin J_{D_s}$.
4. Lemma 3(a) with $Y=P(j)$ gives $T\perp_d P(j)\setminus D_j\mid D_j$. Apply Lemma 4(b).

Conditional independence implies the mean null (Lemma 1.4), so a separator in $\mathcal{O}$ excludes $j$. If $B_Z\in\mathcal{O}$, every subset of $B_Z$ is in $\mathcal{O}$, so the separators in items 2 and 3 are all overlap sets. $\square$

The separators are fixed sets inside $B_Z$. That is what the precision proof needs (Theorem 2).

**Proposition 1 (the Markov boundary keeps the other causes; the direct set does not).** Under A1 and assumptions 2a and 3, $T\perp\!\!\!\perp Z^{[m]\setminus B_Z}\mid Z^{B_Z}$. If moreover $T\not\perp\!\!\!\perp Z^j\mid Z^{[m]\setminus\{j\}}$ for each $j\in B_Z$, then $B_Z$ is the unique Markov boundary.

*Proof.* The first claim is Lemma 4(a) with $Y=Z^{[m]\setminus B_Z}$. For uniqueness: if $S$ is a Markov blanket and $j\in B_Z\setminus S$, weak union gives $T\perp\!\!\!\perp Z^j\mid Z^{[m]\setminus\{j\}}$, a contradiction. $\square$

*Frog.* Take $S\sim\mathrm{Bern}(1/2)$ (large frog), $P\in\{0,1\}$, and $\mathbb{P}(P=1\mid T,S)$ equal to $0.9$ for treated small frogs, $0.5$ for treated large frogs, and $0.05$ for controls. Then $T\perp\!\!\!\perp S$, but $\mathbb{E}[S\mid T=1,P=1]=0.25/0.70=0.357$, while $\mathbb{E}[S\mid T=0,P=1]=0.5$.
- So the Markov boundary includes size, but the direct set does not: size is a conditioning tool only.
- NES's CelebA design has the same structure: Smiling $\to$ Eyeglasses $\leftarrow T$.

**Proposition 2 (the direct set is the minimal set of changed mechanisms).** Under A1, let $q_k(\cdot\mid\mathrm{pa}_k)$ be the mechanism of $L_k$, and let $\mathrm{Shift}(T):=\{k:q_k(\cdot\mid\ell,1)\ne q_k(\cdot\mid\ell,0)$ on parent values of positive probability$\}$.
1. $\mathrm{Shift}(T)\subseteq\mathrm{Ch}(T)$.
2. Every mechanism outside $K$ is the same in both arms iff $\mathrm{Shift}(T)\subseteq K$.
3. If each edge out of $T$ is used (causal minimality), $\mathrm{Shift}(T)=\mathrm{Ch}(T)$, so under Theorem 1 with $B_Z\in\mathcal{O}$, $\mathrm{Adj}=\{j_k:k\in\mathrm{Shift}(T)\}$.

*Proof.* (1) A mechanism without $T$ as a parent cannot depend on $t$. (2) "Only if": an element of $\mathrm{Shift}\setminus K$ is a changed mechanism outside $K$. "If": mechanisms outside $K\supseteq\mathrm{Shift}$ do not depend on $t$. (3) Minimality of the edge $T\to k$ means $q_k$ is not a.e. constant in $t$. $\square$

This is the sparse-mechanism-shift view of an intervention (Schölkopf et al., 2021; Perry et al., 2022). In Joint Causal Inference terms (Mooij et al., 2020), DINES estimates the targets of a known intervention when the targets are latent. A child whose mechanism changes only the spread of its atom is not recovered by mean tests.

**Proposition 3 (without (a) and (b), mediation is not identified).** Let $(T,Z^P,Z^V)$ have any law with $T\sim\mathrm{Bern}(e)$ and real-valued atoms, and suppose $\mathbb{E}[Z^V\mid T,Z^P]$ depends on $T$. Then the same law is generated by a model $M'$ satisfying A1, assumption 2 (exact capture) and assumption 3 (no hidden factor), in which the speed factor is a child of $T$.

*Proof.* Let $F_{P|T}$ and $F_{V|T,P}$ be regular conditional distribution functions with quantile functions $Q$. Let $U_1,U_2$ be i.i.d. uniform and independent of $T$, and put $P'=Q_{P|T}(U_1\mid T)$, $V'=Q_{V|T,P}(U_2\mid T,P')$, $Z^P=P'$, $Z^V=V'$. Then $(T,P',V')$ has the given law, $T$ is a root, the noises are independent, and the atoms are the factors themselves. In $M'$ we have $\mathbb{E}[V'\mid T,P']=\mathbb{E}[Z^V\mid T,Z^P]$, which depends on $T$. So $V'$'s equation must use $T$: $T\in\mathrm{Pa}(V')$. $\square$

The size world of §2.3(a) and the coarse-atom world of §2.3(b) generate laws of exactly this kind, with speed *not* changed directly. $M'$ (the lethargic frog) generates the same laws with speed changed directly. Every procedure that uses only the law of $(T,Z)$ is therefore wrong in one of the two worlds. Dropping $V$ is justified only by assumptions (a) and (b).

**Proposition 4 (a factor that copies the treatment).** Under A1, let $c\in\mathrm{Ch}(T)$ have $\mathrm{Pa}(c)=\{T\}$ and $L_c=f(T)$ for an injective $f$, as with skin colour revealing the arm. Then:
- (i) for every $k\in\mathrm{Ch}(c)$, replacing $L_c$ by $f(T)$ in $k$'s equation yields a model with the same law of $(T,L,Z)$ in which $k$ is a child of $T$;
- (ii) for every $k\in\mathrm{Ch}(T)\setminus\{c\}$, replacing $T$ by $f^{-1}(L_c)$ yields a model with the same law in which $k$ is a child of $c$ and not of $T$.

Both models satisfy A1 with $T$ a root.

*Proof.* Since $L_c=f(T)$ a.s., each substitution changes no variable's value. Noises are untouched. (i) adds the edge $T\to k$, and $T$ is a root, so no cycle appears. (ii) replaces $T\to k$ by $c\to k$. A cycle would need $k\in\mathrm{An}(c)=\{c,T\}$, which is impossible. $\square$

So with a perfect copy of $T$, whether an atom is changed "by the treatment" or "only through colour" is not identified, in either direction: a sheen atom changed only through colour, and the paralysis atom $P$, are symmetric in the data. DINES keeps both: neither has a separator in $\mathcal{O}$, and the only sets that make their nulls hold contain the colour atom. It still drops $V$ via $\{P\}\in\mathcal{O}$, and it flags $P$ and sheen as **undetermined** (§2.9). What *is* identified is that neither is explained by any overlap set.

### 2.5 Algorithm

```
DINES(T, Z, optional W; level α; margin δ ≥ 0; overlap ε, f)
  # p_j(A): p-value for "ρ_{j|A} = 0" (δ = 0) or "|ρ_{j|A}| ≤ δ" (δ > 0), conditioning on (Z^A, W)
  # A set A is SKIPPED if more than a fraction f of the units have estimated π_A outside [ε, 1−ε]

  1. Forward phase
     S ← ∅; K ← ∅                                  # K: treatment-revealing atoms, never conditioned on
     loop:
        R ← [m] \ (S ∪ K);  if R = ∅: break
        j* ← argmin_{j ∈ R} p_j(S)                 # smallest p-value, NOT largest raw effect
        if p_{j*}(S) > α / |R|: break              # Bonferroni gate
        if S ∪ {j*} is skipped: K ← K ∪ {j*}  else: S ← S ∪ {j*}
     F ← S ∪ K

  2. Terminal phase (every j ∈ F judged on subsets of the same F)
     j is DROPPED       if p_j(A) > α/m for some non-skipped A ⊆ F \ {j}
     j is UNDETERMINED  if not dropped, and p_j(A) > α/m for some skipped A ⊆ F \ {j}
     j is KEPT          otherwise
     report: KEPT ∪ UNDETERMINED (the estimate of Adj), with the flag

  3. Indirect output (optional, §2.12)
```

- **Implementation status.** `src/eci/dines.py` applies the overlap rule in the terminal phase only (so $K=\emptyset$ there), with defaults $\varepsilon=0.02$ and $f=0.10$. The forward set-aside step is what Theorem 2 needs. Without it, a treatment-revealing atom can stall the forward phase, which is exactly what happened to NES on the mice data (§2.9).
- **Cost.** At most $|F|\,2^{|F|-1}$ terminal tests, at level $\alpha/m$ however many are run.
- **Ordering.** The order of entry does not enter any proof. We order by $p$-value because that is scale-free, it is IAMB's heuristic, and it keeps $F$ small.
- **Two speeds.** The forward phase needs only consistency, so a fast linear test may be used there. The terminal phase needs validity at fixed nulls, so it uses the cross-fitted test (§3.1).

### 2.6 Guarantees

**Theorem 2 (selection guarantee; generic).**

*Setting.*
- Let $\bar{\mathcal{O}}\ni\emptyset$ be a family of conditioning sets, and assume **(O)**: with probability $\to1$, the algorithm skips exactly the sets outside $\bar{\mathcal{O}}$. Under the rule of §2.5, (O) holds with $\bar{\mathcal{O}}=\{A:q_A<f\}$, where $q_A:=\mathbb{P}(\pi_A\notin[\varepsilon,1-\varepsilon])$, provided that $q_A\neq f$ and $\hat q_A\to q_A$ in probability for every $A$.
- Fix a null family $\mathcal{H}$: the mean null $H_0$, or the margin null $H_0^\delta$ with $\delta\ge0$.
- Fix a target $\mathcal{A}\subseteq[m]$ and a pool $B\subseteq[m]$.

*Assumptions.*
- **(T1) No cancellation at the target.** For $j\in\mathcal{A}$, $H(j\mid A)$ is false for every $A\in\bar{\mathcal{O}}$ with $j\notin A$.
- **(T2) Fixed separators in the pool.** Every $j\notin\mathcal{A}$ has a fixed $A_j\in\bar{\mathcal{O}}$ with $A_j\subseteq B\setminus\{j\}$ and $H(j\mid A_j)$ true.
- **(T3) Forward completeness.** For every $S\in\bar{\mathcal{O}}$ and every $K\subseteq[m]\setminus S$ with $B\not\subseteq S\cup K$, some $j\notin S\cup K$ has $H(j\mid S)$ false.
- **(T4) Tests.**
  - (a) $\limsup_n\mathbb{P}(p_j(A_j)\le t)\le t$ for $j\notin\mathcal{A}$ and all $t\in(0,1)$.
  - (b) $\mathbb{P}(p_j(A)\le t)\to1$ for every $t>0$ and every fixed pair with $A\in\bar{\mathcal{O}}$ and $H(j\mid A)$ false.

*Conclusion.* Let $\hat R_n$ be the reported set ($=$ kept $\cup$ undetermined). Then:
- $\mathbb{P}(\mathcal{A}\subseteq\hat R_n)\to1$, using only (T1), (T4b) and (O);
- $\limsup_n\mathbb{P}(\hat R_n\not\subseteq\mathcal{A})\le\frac{m-|\mathcal{A}|}{m}\alpha$;
- hence $\liminf_n\mathbb{P}(\hat R_n=\mathcal{A})\ge1-\alpha$;
- if (T4c) $\mathbb{P}(p_j(A_j)\le t)\to0$ for each $t>0$ and $j\notin\mathcal{A}$, then $\mathbb{P}(\hat R_n=\mathcal{A})\to1$;
- if $\bar{\mathcal{O}}$ contains every set, nothing is skipped with probability $\to1$, and the reported set is the kept set.

*Proof.* Let $E_O$ be the event in (O). On $E_O$ the algorithm skips exactly the sets outside $\bar{\mathcal{O}}$. In particular $S$ stays in $\bar{\mathcal{O}}$, because an atom is added to $S$ only if $S\cup\{j^*\}$ is not skipped.

(i) *The forward phase reaches $B\cup\mathcal{A}$.* Consider any state $(S,K)$ with $S\in\bar{\mathcal{O}}$ and $B\cup\mathcal{A}\not\subseteq S\cup K$. It has a witness $w\notin S\cup K$ with $H(w\mid S)$ false: use (T3) if $B\not\subseteq S\cup K$, and otherwise any $j\in\mathcal{A}\setminus(S\cup K)$, by (T1). By (T4b) each witness passes the gate with probability $\to1$, and there are finitely many states. On the intersection of these events, the phase cannot stop at such a state. Each round adds one atom to $S\cup K$. So $F\supseteq B\cup\mathcal{A}$. Without (T3), the same argument still gives $F\supseteq\mathcal{A}$.

(ii) *Recall.* For $j\in\mathcal{A}$, every non-skipped $A\subseteq F\setminus\{j\}$ lies in $\bar{\mathcal{O}}$, so $H(j\mid A)$ is false there. These are finitely many fixed alternatives, so with probability $\to1$ all are rejected at level $\alpha/m$, and $j$ is not dropped.

(iii) *Precision with fixed nulls only.* On $\{B\subseteq F\}\cap E_O$, a $j\notin\mathcal{A}$ is tested at $A_j\subseteq B\setminus\{j\}\subseteq F\setminus\{j\}$. This set is in $\bar{\mathcal{O}}$, hence not skipped, and it is not chosen from the data. So $j\in\hat R_n$ implies $p_j(A_j)\le\alpha/m$. By (T4a) and a union bound,
$$\limsup_n\mathbb{P}(\hat R_n\not\subseteq\mathcal{A})\le\sum_{j\notin\mathcal{A}}\limsup_n\mathbb{P}(p_j(A_j)\le\alpha/m)\le\tfrac{m-|\mathcal{A}|}{m}\alpha .$$
Under (T4c) each term is 0.

(iv) Combine (ii) and (iii). $\square$

No step needs validity at a data-dependent conditioning set. The terminal level $\alpha/m$ controls precision; the forward gate does not matter for the bound. A data-dependent terminal level such as $\alpha/|F|$ is not covered. The split between kept and undetermined uses tests at skipped sets, which carry no guarantee; only the reported union is covered.

**Corollary 1 (statistical target).** Take $\mathcal{H}=H_0$ (with the GCM, "$\rho=0$"), $\mathcal{A}=\mathrm{Adj}_Z(T)$, and $\bar{\mathcal{O}}=\mathcal{O}$. Every set either has overlap and $q_A<f$, or lacks overlap and $q_A>f$; in plain words, there are no borderline sets. Then (T1) holds by definition, and:
- **recall always holds:** $\mathbb{P}(\mathrm{Adj}\subseteq\hat R_n)\to1$;
- under **assumption L (local separation)** — (T2) and (T3) for some fixed pool $B$ — $\liminf_n\mathbb{P}(\hat R_n=\mathrm{Adj})\ge1-\alpha$.

**Corollary 2 (causal target).** Assume:
- A1, assumptions 2 and 3, and no cancellation at the children;
- every non-child atom has one of the separators of Theorem 1 in $\mathcal{O}$ (for example, $B_Z\in\mathcal{O}$);
- (T3) with $B=B_Z$ (no exact cancellation of the explaining-away witnesses), (T4) and (O).

Then $\liminf_n\mathbb{P}(\hat R_n=J_C)\ge1-\alpha$, and $J_C$ indexes the changed mechanisms (Proposition 2). *Proof.* Theorem 1 gives $\mathrm{Adj}=J_C$, together with (T2) for $B=B_Z$ and fixed separators in $\mathcal{O}$. Apply Theorem 2. $\square$

**Assumption L survives for an arbitrary dictionary.** Take $B$ to be a Markov blanket of $T$ among the atoms ("the atoms around the treatment").
- Atoms outside $B$ are always explained away by $B$.
- Non-direct atoms *inside* $B$ (the other causes) need a separator made of other blanket atoms. Under assumptions 2–3 this is Theorem 1. For a general dictionary we do not know whether one exists (Open question O1).
- (T3) is the mean analogue of the composition property that IAMB's grow phase needs, and it is not testable directly.

So, in plain words, assumption L is: *the atoms around the treatment are enough to explain away everything else, and the search does not stop before collecting them.* Under assumptions 2–3, only the second half remains, and it is a no-cancellation condition. In the toy model this is the explaining-away shift $|\rho_{H|\{A,C\}}|=0.077$.

**How the plain assumptions of §0 map onto the formal ones.**

| §0 | Formal | Used in |
|---|---|---|
| 1 randomized $T$ | $T$ a root of $\mathcal{G}_L$ | Lemma 3; known propensity at $A=\emptyset$ |
| 2 alignment | 2a screening (NEXIS Eq. 6, jointly) + 2b purity on the treatment side; realistic: (δ-a) | Lemmas 4–5, Theorem 1, Theorem 3 |
| 3 no hidden common cause | coverage of $\mathrm{MB}(T)$ after folding (Lemma 2) | Theorem 1 |
| 4 no exact cancellation, valid tests | (T1), (T3), (T4), (O); fixed dictionary | Theorem 2 |
| — (without 2–3) | assumption L = (T2) + (T3) | Corollary 1 |

**Theorem 3 (realistic alignment).** Assume A1. For each $k\in\mathrm{MB}(T)$, choose a distinct atom $j_k$ (its best atom, with no alignment required). Take the separators $A_j$ of Theorem 1, and assume each lies in $\mathcal{O}$. Fix $\delta>0$ and assume:
- **(δ-a) Leaks below the margin.** $|\rho_{j|A_j}|\le\delta$ for every $j\notin J_C$.
- **(δ-b) Direct effects above the margin.** $|\rho_{j_c|A}|>\delta$ for every child $c$ and every $A\in\mathcal{O}$ with $j_c\notin A$.
- **(δ-c) Forward completeness at the margin.** (T3) holds with $H_0^\delta$ and $B=B_Z$.
- **Tests and overlap.** (T4) holds for $H_0^\delta$ (valid on the boundary $|\rho|=\delta$, consistent beyond it), and (O) holds with $\bar{\mathcal{O}}=\mathcal{O}$.

Then $\liminf_n\mathbb{P}(\hat R_n=J_C)\ge1-\alpha$. If every leak is strict and $\hat\rho\to\rho$ with standard error $\to0$, then $\mathbb{P}(\hat R_n=J_C)\to1$.

*Proof.* Apply Theorem 2 with $\mathcal{H}=H_0^\delta$, $\mathcal{A}=J_C$ and $B=B_Z$.
- (T1) is (δ-b) and (T2) is (δ-a). The containment $A_j\subseteq B_Z\setminus\{j\}$ holds as in Theorem 1.
- For strict leaks, $(|\hat\rho|-\delta)/\widehat{\rm se}\to-\infty$, which gives (T4c). $\square$

So $\delta$ enters the algorithm and the assumption, never the target. Under exact alignment every leak is 0, so (δ-a) holds for any $\delta\ge0$. With $\delta>0$, an other-cause atom is collected only if its explaining-away shift exceeds $\delta$, which makes (δ-c) the upper limit on $\delta$.

### 2.7 How large are leaks?

**Proposition 5 (exact leak identity).** Fix $j$ and $A$. Suppose some $F$ (the factors that $A$ was meant to align with) satisfies $\mathbb{E}[Z^j\mid T,F,Z^A]=\mathbb{E}[Z^j\mid F]=:g(F)$, with $\mathbb{E}g^2<\infty$. Then $\Delta_{j,A}=\Delta_{g,A}$ a.s., and
$$\rho_{j|A}=\rho_{g|A}\cdot\sqrt{\frac{\mathbb{E}[\mathrm{Var}(g(F)\mid Z^A)]}{\mathbb{E}[\mathrm{Var}(Z^j\mid Z^A)]}},\qquad\text{where the square root is}\le1 .$$
**In words, leak = (treatment signal left in the carried part once the atoms are held fixed) × √(share of the atom's leftover variance that is carried signal the atoms miss).** Both factors are at most 1. The second is 0 exactly when the atoms determine $g(F)$, which is screening (Lemma 5).

*Proof.* $\mathbb{E}[Z^j\mid T,Z^A]=\mathbb{E}[g(F)\mid T,Z^A]$ by the tower rule, so the arm contrasts agree. By Lemma 1.5 the numerators of $\rho_{j|A}$ and $\rho_{g|A}$ agree, and the denominators are $\sqrt{V_A\,\mathbb{E}\mathrm{Var}(\cdot\mid Z^A)}$; if $\mathbb{E}\mathrm{Var}(g\mid Z^A)=0$, both numerators vanish. Averaging over $T$ gives $\mathbb{E}[Z^j\mid F,Z^A]=g(F)$, so $Z^j-g(F)$ is uncorrelated with every function of $(F,Z^A)$. Hence $\mathrm{Var}(Z^j\mid Z^A)=\mathrm{Var}(g\mid Z^A)+\mathbb{E}[(Z^j-g)^2\mid Z^A]$. $\square$

*When does the condition hold?* For pure atoms with independent noise, $Z^A=\phi(L_D,\eta)$ with $\eta\perp\!\!\!\perp(T,L,N^j)$, and with $T\perp_d P(j)\setminus D\mid D$ (as for the Theorem 1 separators): take $F=L_D$. For impure atoms it can fail even with screening (example (d)). The leak then has no general bound; it must simply be at most $\delta$.

**Proposition 6 (linear-Gaussian leak).** Assume:
- $L_D\mid T\sim\mathcal{N}(\nu+T\tau_D,\Sigma)$ with $\Sigma\succ0$;
- $Z^A=L_D+\eta$, with $\eta\sim\mathcal{N}(0,\Omega)$ independent of $(T,L)$ and of $Z^j$'s noise;
- $(L_D,Z^j)$ is jointly Gaussian given $T$, with a $T$-free covariance;
- $\mathbb{E}[Z^j\mid T,L_D]=a+\gamma^\top L_D$.

Then the arm contrast is the constant $\theta_{j|A}=\gamma^\top\Omega(\Sigma+\Omega)^{-1}\tau_D$. Moreover:
- *Uncorrelated factors:* $\theta=\sum_k\gamma_k\tau_k(1-R^2_k)$, where $R^2_k=\Sigma_{kk}/(\Sigma_{kk}+\Omega_{kk})$. **Each conditioning factor contributes the effect it carries times its unaligned share.**
- *In general:* $|\theta|\le u\,\|\Sigma^{1/2}\gamma\|\,\|\Sigma^{-1/2}\tau_D\|$, where $u=\max_ww^\top\Omega w/w^\top(\Sigma+\Omega)w$.
- *On the $\rho$ scale:* $\rho_{j|A}=\theta\sqrt{V_A}/\sqrt{\sigma^2_{j|A}+\theta^2V_A}$, so $|\rho|\le|\theta|\sqrt{V_A}/\sigma_{j|A}\le|\theta|/(2\sigma_{j|A})$, where $\sigma^2_{j|A}=\mathrm{Var}(Z^j\mid T,Z^A)$.

*Proof.* Proposition 5's condition holds with $F=L_D$, so $\mathbb{E}[Z^j\mid T,Z^A]=a+\gamma^\top\mathbb{E}[L_D\mid T,Z^A]$. Here $\mathbb{E}[L_D\mid T,Z^A]=\nu+T\tau_D+\Sigma(\Sigma+\Omega)^{-1}(Z^A-\nu-T\tau_D)$, whose coefficient on $T$ is $\Omega(\Sigma+\Omega)^{-1}\tau_D$. The diagonal case is immediate.

For the general bound, put $N=\Sigma^{-1/2}\Omega\Sigma^{-1/2}$. Then $\Omega(\Sigma+\Omega)^{-1}=\Sigma^{1/2}N(I+N)^{-1}\Sigma^{-1/2}$, and the symmetric matrix $N(I+N)^{-1}$ has operator norm $\nu_{\max}/(1+\nu_{\max})=u$. Cauchy–Schwarz concludes.

For $\rho$: $\mathrm{Var}(Z^j\mid T,Z^A)$ is constant, so $\mathbb{E}\mathrm{Var}(Z^j\mid Z^A)=\sigma^2_{j|A}+\theta^2V_A$, and Lemma 1.5 applies. Finally $V_A=\mathbb{E}\mathrm{Var}(T\mid Z^A)\le\tfrac14$. $\square$

The scalar case is example (c). Measured checks: the matrix formula matches direct Gaussian conditioning to 6 decimals on a random 3-factor instance, and the identity of Proposition 5 holds to 5 decimals at every separator of the toy (Appendix B.4).

**Conjecture 1 (general leak bound).** For pure atoms with independent noise, beyond the linear-Gaussian case, the first factor in Proposition 5 is at most a constant × (the standardized effect of $T$ on the aligned factors) × (their unaligned share). Proposition 6 proves this in the linear-Gaussian case. Proposition 5 alone gives only $|\rho|\le\sqrt{\text{share}}$, which is loose: 0.14–0.23 against true leaks of 0.003–0.015 in the toy.

**The toy at the population level** (Appendix B.4; atoms are factors plus noise of s.d. 0.3):
- leaks at the full-blanket separator are 0.003–0.015, the largest being the mixed atom $0.5A+0.5D$;
- leaks at the smaller ancestral separators are 0.002–0.016; these are slightly larger, because the extra blanket atoms help predict $A$;
- the smallest direct effect over all $2^9$ conditioning sets is 0.120;
- forward completeness at margin $\delta$ holds iff $\delta<0.090$.

So every condition of Theorem 3 holds for $\delta\in[0.015,0.090)$, and $\delta=0.05$ lies inside this window. At $\delta=0$, a leak of 0.0147 is 1.86 standard errors at $n=16{,}000$. It therefore passes the terminal threshold (2.81 standard errors) in roughly 17% of runs, and in about 3% at $n=4000$. The measured rates are the same order: the mixed atom is falsely kept in 10/100 runs at $n=16{,}000$, and there are 0.04 false atoms per run at $n=4000$.

### 2.8 Choosing δ

1. **The window.** $\delta$ must be above every leak, and below every direct effect and every forward witness. All of these are on one scale: the partial correlation between $T$ and the atom given the conditioning atoms. For example, $|\rho|=0.05$ means that $T$ explains 0.25% of the atom's leftover variance.
2. **Why $n$ matters at $\delta=0$.** Near $\rho=0$, the standard error of $\hat\rho$ is about $1/\sqrt n$, so the effective threshold is about $z/\sqrt n$ with $z=\Phi^{-1}(1-\alpha/2m)$. That threshold shrinks with $n$, and a leak $\ell$ starts to be selected around $n\approx(z/\ell)^2$. With $\delta>0$ the threshold is about $\delta+z/\sqrt n$. For example, with $m=10^4$ and $\alpha=0.05$, $z\approx4.56$, and at $n=5000$ the $\delta=0$ threshold is about 0.064.
3. **Estimating the leak.** Measure alignment where labels exist: probe $R^2$, or the share of held-out information, as in NEXIS's 94–96%. Plug it into Proposition 6, which gives $|\theta|\le\sum_k|\gamma_k\tau_k|(1-R^2_k)$ in the uncorrelated case, and convert to $\rho$. Outside the linear-Gaussian case this is a heuristic (Conjecture 1).
4. **Recall cost.** A direct effect must exceed $\delta$ given every overlap set, including sets that almost reveal $T$, where $\rho$ shrinks like $\sqrt{V_A}$. Finite-sample recall needs about $(\min|\rho_{\rm direct}|-\delta)\sqrt n\gtrsim z$. In the toy, $(0.120-0.05)\sqrt{1000}\approx2.2<2.81$, which matches the measured recall of 0.85 at $n=1000$.
5. **Declare $\delta$ and check it.**
   - Fix $\delta$ before looking at the data, like an equivalence margin.
   - Report the output over a small grid ($0, 0.02, 0.05, 0.1$), with each kept atom's smallest $|\hat\rho|$.
   - An atom that disappears when $\delta$ increases slightly is a leak suspect.
   - A principled automatic choice is Open question O6.

### 2.9 Overlap in practice: when one atom reveals the treatment

"Strong" here means *predictive of $T$*, not large in raw magnitude. A large shift of a noisy atom is harmless. An atom that reveals the arm makes every null given it hold vacuously (Lemma 1.2).

The population target (Definition 1) therefore ignores non-overlap sets. The atoms this leaves unresolved are flagged **undetermined**. With an exact copy of $T$ they include both the paralysis atom and any atom changed only through colour, which are symmetric in the data (Proposition 4).

- **Mice case.** In the paired NES run on the mice data, round 1 picks the "bag" neuron. Round 2 then fails with "no overlap", because the bag changes in 34–36 of the 36 pools, so only one neuron can be returned per switch.
- **Finite-sample rule** (§2.5): skip a set when more than a fraction $f$ of the units have $\hat\pi_A\notin[\varepsilon,1-\varepsilon]$.
- **What Theorem 2 then guarantees,** under (O) with $\bar{\mathcal{O}}=\mathcal{O}$, that is, no borderline sets:
  1. no atom that some overlap set explains away is reported (error at most $\alpha$);
  2. every direct atom is reported, as kept or undetermined;
  3. undetermined atoms may be changed directly or only through the revealing factor, and no sample size separates the two (Proposition 4).

If a set has partial overlap, between the two cases, the rule treats it as revealing. That choice is conservative: it may add undetermined atoms, but it never drops a direct one.

### 2.10 Feature splitting

**Proposition 7 (groups of atoms).** Replace the single principal atom by disjoint groups $G_k$ with exact capture: $L_k=a_k(Z^{G_k})$ and $Z^{G_k}=h_k(L_k,N^Z)$. Assume $B_Z^G:=\bigcup_{k\in\mathrm{MB}}G_k\in\mathcal{O}$. Then:
1. $\mathrm{Adj}\subseteq\bigcup_{c\in\mathrm{Ch}(T)}G_c$. **There is no false causal claim:** every reported atom belongs to the group of a changed factor. Every other atom has a fixed separator: $B_Z^G$, or $\bigcup_{k\in D_s}G_k$ for atoms of an other cause $s$. So Theorem 2's precision applies with $B=B^G_Z$.
2. Whether each member of $G_c$ is direct depends on the split.
   - *Clean split:* $Z^1=UF$ and $Z^2=(1-U)F$, with $U\sim\mathrm{Bern}(1/2)$ independent measurement noise, $F\ne0$ a.s., and $\mathbb{E}[F\mid T]$ non-constant. Both atoms are direct: the set is inflated but correct.
   - *Redundant split:* add $Z^3=F$. Then none of the three atoms is direct. The factor vanishes from the output, a loss of recall rather than a false claim.

*Proof.* (1) Determination implies screening for every set, so Lemma 4 applies with vector-valued groups in place of single atoms. The proof of Theorem 1, items 2 and 3, then goes through. Since $s\notin D_s$ and the groups are disjoint, $j\notin A_j$.

(2) *Clean split.* Given $\emptyset$, $\mathbb{E}[Z^1\mid T]=\tfrac12\mathbb{E}[F\mid T]$ varies with $T$. Given $Z^2$: on $\{Z^2=0\}=\{U=1\}$ (probability ½ in both arms), $Z^1=F$, and $\mathbb{E}[F\mid T,U=1]=\mathbb{E}[F\mid T]$ varies with $T$. $Z^2$ is symmetric.

*Redundant split.* Each atom is a function of the other two. If $Z^j=f(Z^A)$, then $\mathbb{E}[Z^j\mid T,Z^A]=f(Z^A)=\mathbb{E}[Z^j\mid Z^A]$, so the null holds at $A$ (which has overlap when $F$ does not reveal $T$). Downstream atoms are still explained away by the group, by part (1). $\square$

*Recommendation.* Score per factor: group atoms before the analysis, using $Z$ only (for example, correlated atoms or similar decoder directions), and test the groups with a vector-valued test. **The real damage comes from mixed atoms, not from splitting.** A mixed atom that is used for conditioning, or one that reads a changed factor together with an unrecorded other cause, leaks, and Theorem 3 covers every leak regardless of its source. Near-redundant groups at $\delta>0$ also lose all their members, which is one more reason to group.

### 2.11 When can DINES win, and when can nobody?

- **DINES wins** when the effect has consequences like $V$ and alignment is approximately right (leak $<\delta<$ direct). Marginal tests report $V$, because its total effect is real. NES may pick $V$ first if its raw shift is largest, its binarized strata leak, and it never removes an atom. DINES drops $V$ through one fixed test given $P$'s atom, with error controlled at $\alpha$.
- **Nobody wins** when $V$ is informative beyond the dictionary:
  - $P$'s atom is too coarse or too noisy to carry what speed depends on;
  - a common cause such as size is not recorded;
  - an atom copies the treatment.

  The law of $(T,Z)$ is then the same as in a world where $V$ (or the sheen atom) is changed directly (Propositions 3–4). A better dictionary, or measuring size as a covariate (Proposition 10), changes the situation; a better algorithm does not.

### 2.12 The indirect output

This section assumes exact capture on $\mathrm{MB}(T)$ (determination and purity) and $B_Z\in\mathcal{O}$.

**Definition 2.** For $j\in\mathrm{Ind}_Z$:
- *causal carriers:* $\mathrm{Car}(j):=\{j_c:c\in\mathrm{Ch}(T),\ c\in\mathrm{An}(v)\text{ for some }v\in P(j)\}$;
- *observable carriers:* $\mathrm{Car}_B(j):=\{k\in J_C:H_0(j\mid B_Z\setminus\{k\})\text{ false}\}$.

$\mathrm{Car}(j)\ne\emptyset$: otherwise $P(j)$ would contain no descendant of $T$.

**Assumption N1.** No directed path runs from a child of $T$ to another member of $\mathrm{MB}(T)$.
**Assumption CF.** For $j\in\mathrm{Ind}_Z$ and $j_c\in\mathrm{Car}(j)$, $H_0(j\mid B_Z\setminus\{j_c\})$ is false.

**Proposition 8 (carriers are identified).** Under the conditions of Corollary 2 with exact capture, N1 and CF, $\mathrm{Car}_B(j)=\mathrm{Car}(j)$, and $H_0(j\mid B_Z)$ holds for every $j\in\mathrm{Ind}_Z$.

*Proof.* The last claim is Theorem 1 item 2, and "$\supseteq$" is CF. For "$\subseteq$", take a child $c$ that is an ancestor of no $v\in P(j)$, and let $C:=\mathrm{MB}(T)\setminus\{c\}$. We show $T\perp_d P(j)\mid C$. A path from $T$ starts with $T\to c'$.
- *If $c'\neq c$:* $c'\in C$. If $c'$ is a non-collider, the path is blocked. If it is a collider, the next vertex $x$ is a parent of $c'$, and $x\ne c$ by N1, so $x\in C$ is a conditioned non-collider.
- *If $c'=c$:* a collider at $c$ is open only through a descendant of $c$ in $C$, which N1 excludes. Otherwise the path continues $c\to u_1\to\dots$. The first backward edge creates a collider in $\mathrm{De}(c)$, which N1 keeps closed. An open path is therefore directed, $c\to\dots\to v$, which contradicts the choice of $c$.

Determination and purity give screening for $C$, and Lemma 4(b) with $V=P(j)$ gives $H_0(j\mid B_Z\setminus\{j_c\})$. $\square$

Without N1 the observable notion reports the *last* changed factor on a path.

**Estimation.**
- Blanket estimate: $\hat B:=\{j\in F:p_j(F\setminus\{j\})\le\alpha_n\}$.
- Indirect set: $\widehat{\mathrm{Ind}}:=\{j\notin\hat R:p_j(\emptyset)\le\alpha_2/2m\}$.
- Carriers: $\widehat{\mathrm{Car}}(j):=\{k\in\hat R:p_j(\hat B\setminus\{k\})\le\alpha_2/(2m\max(1,|\hat R|))\}$.

**Assumption MTF.** For every $S$ with $B_Z\subseteq S$ and every $j\in B_Z$, $H_0(j\mid S\setminus\{j\})$ is false.

**Proposition 9 (full output).** Assume the conditions of Proposition 8 and MTF. Assume also that the tests are consistent at fixed alternatives at the level $\alpha_n\to0$ (assumed; for $\sqrt n$-consistent statistics this typically holds when $\log(1/\alpha_n)=o(n)$), consistent against $H_0(j\mid\emptyset)$ for $j\in\mathrm{Ind}_Z$, and valid at the fixed nulls used. Then
$$\liminf_n\mathbb{P}(\hat R=J_C,\ \widehat{\mathrm{Ind}}=\mathrm{Ind}_Z,\ \widehat{\mathrm{Car}}(j)=\mathrm{Car}(j)\ \forall j)\ge1-\alpha-\alpha_2 .$$

*Proof.* (i) $\mathbb{P}(\hat B=B_Z)\to1$. By Theorem 2(i), $B_Z\subseteq F$ with probability $\to1$. Fix $S\supseteq B_Z$.
- For $j\in S\setminus B_Z$, the null is true (Theorem 1 item 2 and weak union). Validity gives $\limsup\mathbb{P}(p\le\alpha_n)\le t$ for every $t>0$, so this probability tends to 0.
- For $j\in B_Z$, MTF and consistency give probability $\to1$.

Take a finite union bound over the pairs $(S,j)$.

(ii) On $E:=\{\hat R=J_C,\hat B=B_Z\}$ (with $\limsup\mathbb{P}(E^c)\le\alpha$), every remaining test is at a fixed set. The possible errors are:
- false indirect atoms: at most $m$ fixed nulls at level $\alpha_2/2m$;
- false carriers: fixed nulls by Proposition 8, at total level $\alpha_2/2$;
- missed indirect atoms or missed carriers: probability $\to0$. $\square$

Under realistic alignment, carriers are over-reported, because noisy child atoms leave part of every mediated effect. Report carriers as hypotheses, together with the residual $p_j(\hat B)$.

---

## 3. Conditional tests in practice

### 3.1 The test: residualization (GCM), not stratification

For a pair $(j,A)$ with $V_A=(Z^A,W)$:
1. Cross-fit $\hat\pi(V_A)\approx\mathbb{P}(T=1\mid V_A)$ and $\hat\mu_j(V_A)\approx\mathbb{E}[Z^j\mid V_A]$, using folds made of whole clusters for video.
2. Form $R_i=(T_i-\hat\pi_i)(Z^j_i-\hat\mu_{j,i})$ and the statistic $\sqrt n\bar R/\hat\sigma_R$, with a two-sided normal $p$-value.

Properties:
- **What it estimates.** The numerator of $\rho_{j|A}$ (Lemma 1.3). Its bias is the product of the two regression errors, whether or not the null holds, because the other cross terms have conditional mean zero. This is also why $\hat\rho$ is asymptotically normal on the boundary $|\rho|=\delta$.
- **Validity under the mean null** (Shah & Peters, 2020) requires the product of the regression errors to be $o_P(n^{-1/2})$, each error to vanish, and $\mathbb{E}R^2$ to be bounded away from 0. No conditional-independence test with continuous conditioning variables has uniform size and power, so these rate conditions are the price.
- **The propensity is not known.** $Z^A$ is post-treatment, so $\pi_A$ must be learned: it is a classifier of $T$ from the atoms. Only $A=\emptyset$ without covariates has the known value $e$.
- **Sign changes.** Use the PCM (Lundborg et al., 2024), equivalently a weighted GCM (Scheidegger et al., 2022), when sign changes are plausible.
- **Margin test.** $\hat\rho=\bar R/\sqrt{\overline{(T-\hat\pi)^2}\,\overline{(Z^j-\hat\mu)^2}}$, with a delta-method standard error. The implementation uses $p=\Phi((\delta-|\hat\rho|)/\widehat{\rm se})+\Phi((-\delta-|\hat\rho|)/\widehat{\rm se})$. This is asymptotically exact on the boundary, conservative inside it, and equal to the usual two-sided $p$ at $\delta=0$.
- **Report overlap.** For every terminal test, report $\sum_i\hat\pi_i(1-\hat\pi_i)$ and the fraction of extreme $\hat\pi$.
- **Do not binarize conditioning atoms.** A binarized atom is a coarse mediator atom (§2.3(b)): it leaks about 36% of a mediated effect. Keep atoms continuous, and let $\hat\mu,\hat\pi$ represent the point mass at zero (indicator features or trees).

### 3.2 Covariates

**Proposition 10 (covariates; a measured common cause needs no atom).** Let $W$ be observed pre-treatment vertices of $\mathcal{G}_L$ (non-descendants of $T$), added to every conditioning set. Assume:
- every $k\in\mathrm{MB}(T)\setminus W$ has an exactly capturing atom (determination + purity); members of $\mathrm{MB}(T)$ inside $W$, such as a measured frog size, need none;
- $B'\cup W$ has overlap, where $B':=\{j_k:k\in\mathrm{MB}(T)\setminus W\}$;
- no cancellation at the children with $W$ added.

Then $\mathrm{Adj}^W=J_C$.

*Proof.* "$\supseteq$" is the assumption. In both remaining cases we apply Lemma 4 with the conditioning vector $Q$ made of atoms and $W$.
- *$j\notin B'$:* take $Q=(Z^{B'},W)$, a function of $(L,N^Z)$. By determination, $L_{\mathrm{MB}}$ is a function of $Q$, so screening (ii) holds trivially. (i) holds by Lemma 4(a). Hence $T\perp\!\!\!\perp Z^j\mid(Z^{B'},W)$.
- *$j=j_s$ with $s\in\mathrm{Sp}(T)\setminus(\mathrm{Ch}(T)\cup W)$:* let $\mathcal{A}'=\{T\}\cup\mathrm{An}(s)\cup\mathrm{An}(W)$. A child of $T$ is not in $\mathrm{An}(W)$, since otherwise a member of $W$ would descend from $T$. So, as in Lemma 3(b), every moral neighbour of $T$ lies in $D_s$, and $D_s\cup W$ separates $T$ from $s$. Take $Q=(Z^{J_{D_s\setminus W}},W)$. By purity it is a function of $(L_{D_s\cup W},N^Z)$, and by determination $L_{D_s\cup W}$ is a function of $Q$. Lemma 4(b) with $D=D_s\cup W$ concludes.

Both separators are fixed and have overlap. $\square$

In a randomized experiment covariates add power, not validity. At $A=\emptyset$, $R_i=(T_i-e)(Z^j_i-\hat\mu_j(W_i))$ has mean $e(1-e)\mathrm{ATE}_j$ whatever $\hat\mu_j$ is (Tsiatis et al., 2008; Lin, 2013).

### 3.3 Levels

- The forward gate $\alpha/|R|$ is a default; the theorems do not use it.
- The terminal level is $\alpha/m$, where $m$ counts the atoms entering the analysis. Atoms that never fire may be removed beforehand (that screen uses $Z$ only).
- The forward phase may use a linear GCM.

### 3.4 Clustered data (video)

The unit of independence is the cluster (a video or a pool). Use $T_G=\sum_gR_g/\sqrt{\sum_g(R_g-n_g\bar R)^2}$, build folds from whole clusters, and use a $t_{G-1}$ reference when there are few clusters (Cameron & Miller, 2015). Permuting $T$ is exact only at $A=\emptyset$. Averaging atoms over a video yields a different dictionary.

### 3.5 Mean only

All results concern the mean of each atom. Effects on spread only are invisible (O10).

### 3.6 A dictionary fitted on the same data

All results take $\psi$ as fixed. Train the autoencoder on held-out units, or cross-fit it. Reusing the same data is not covered (O8).

---

## 4. What NES got wrong, and what DINES does instead

NES is "Neural Effect Search" (Mencattini, Cadei, Locatello, *Exploratory Causal Inference*, arXiv 2510.14073). Its central observation is correct, and DINES keeps it: marginal tests on an entangled dictionary flag more and more atoms as power grows (Theorems 3.1–3.2), and those atoms form $\mathrm{De}_Z(T)$. Equation numbers refer to the arXiv version. Code references are to `src/neuralscience/nes.py` in `/nfs/scistore19/locatgrp/tmencatt/projects/neuralscience/`. CelebA numbers come from that project's saved runs (Appendix B.3).

### 4.1 NES's hidden assumptions, on the frog

1. **Eq. 24 (additivity) excludes mediation.** The model writes each atom's shift as an additive combination of factor effects. A chain $P\to V$ has no place in it, so NES never says whether $V$ is in its target.
2. **Eqs. 27–28 assume that stratifying on the selected atoms removes exactly the explained effect.** That silently assumes (a) and (b): a paralysis atom that records what speed depends on, and no hidden size. In practice the strata are binarized at $>0$, which is the coarse "leg drags yes/no" atom. About 36% of a mediated effect leaks (measured; §2.3(b)).
3. **Eq. 26 (Principal–Max).** $V$ can have the largest raw shift (raw shifts depend on each atom's scale). It is then picked first and never removed, because there is no backward step. The claim that "the argmax stays principal" after conditioning is false. Counterexample satisfying A.1 and Eqs. 25, 26 and 31: $v_1=(3,0,-1)$, $v_2=(0,1.5,2)$, $\tau^Y=(1,1)$. Round 1 picks atom 1, which is principal; under Eq. 28, round 2 picks atom 3, which is not. In the saved CelebA runs, the mixed atom N_408 (F1 0.48 for hat, 0.45 for eyeglasses) is picked before the eyeglasses atom N_6051: its raw effect is larger but its $t$ is about half (0.773 with $t=4.8$, vs 0.691 with $t=8.8$, at ATE 0.4, $N=1000$). It is the most frequent false selection (50/80 runs at $N=5000$).
4. **Termination needs exact cancellation.** NES stops when every remaining stratified contrast is zero. With leaks, more atoms pass as $n$ grows. In the saved CelebA runs at ATE 0.7, precision is 0.80, 0.62, 0.56, 0.45, 0.49 at $N=250$ to $5000$, with recall 1.0. Pooled over ATE 0.1–0.8, exact recovery falls from 26/80 to 11/80.
5. **"FWER → 0" is wrong at fixed $\alpha$.** $O_p(1)$ statistics compared with a fixed Bonferroni cutoff exceed it with non-vanishing probability, up to about $\alpha$. The correct form is $\liminf\mathbb{P}(\text{exact})\ge1-\alpha$, as in NEXIS's Theorem 4.1 and Theorem 2 here.

**The real biases are assumed away, not bounded.** The transport-error term before Eq. 29 is $\Delta_{j,t}(g):=\mathbb{E}[Z_j\mid G=g,T=t]-\mathbb{E}[Z_j\mid G=g,do(T=t)]$, and Eq. 30 bounds it by $\varepsilon$. This term is identically zero in a randomized experiment, because the law of post-treatment variables under $do(T=t)$ equals their law given $T=t$. The actual problems — strata that hold different frogs in the two arms, coarse strata, hidden size — never appear in the bound.

### 4.2 Smaller points

- The paper's optional residualization is computed in the code (lines 205–224), but the stratified branch, which is the one used, runs on raw data (line 283).
- Cells without both arms are dropped and the weights renormalized (lines 73–95), so the estimand drifts with effect size.
- Frames are treated as i.i.d. units.

| NES element | Problem | DINES |
|---|---|---|
| Target "coarsest abstraction"; Eq. 24 | undefined; additivity excludes mediation | Def. 1, Theorem 1, Prop. 2; mediation framing (Prop. 3) |
| Eqs. 27–28 | assume (a) and (b) silently; binarized strata leak ~36% | assumptions 2–3 stated; continuous conditioning; Theorem 3 for leaks |
| Eq. 26 Principal–Max | scale-dependent; "argmax stays principal" false | $p$-value ordering; guarantee independent of order |
| No backward step | early wrong atoms stay | terminal subset rule at $\alpha/m$ |
| Termination | needs exact cancellation; precision falls with $N$ | fixed-null precision proof; margin $\delta$ |
| Eqs. 29–30 | identically zero in an RCT | conditional-independence formulation |
| "FWER → 0" | false at fixed $\alpha$ | $\liminf\mathbb{P}(\text{exact})\ge1-\alpha$ |
| Dropped cells | estimand drifts | overlap built into the target; undetermined flag |

*Hypothesis (empirical, untested).* NES's precision fall has several causes: items 2–4 above, approximate alignment, and the design itself — Smiling is an other cause of Eyeglasses with no aligned atom (its best-F1 atom is the hat atom 38, F1 0.48). DINES removes the algorithmic causes; $\delta$ addresses approximate alignment; the design issue affects any method.

---

## 5. FAQ

**Why are the other causes needed if they are never output?**
Holding paralysis fixed compares treated and control frogs that reached the same severity by different routes: on average the control frogs that are as paralysed as the treated ones differ in size. That is explaining away (Proposition 1). Anything that size affects, such as speed, then looks shifted. Holding size fixed as well removes the dependence. Other causes are conditioning tools. They are not output, because the treatment does not change their mechanism (Proposition 2).

**Why is "no hidden common cause" needed, and why did NES need it too?**
A hidden common cause is a variable that must be held fixed but cannot be. Without a size atom, $\mathbb{E}[\text{speed}\mid T,\text{paralysis}]$ has the coefficient $-\beta/2$ on $T$ (§2.3(a)), so speed is direct in that dictionary at every $n$. By Proposition 3, no method can do better from these data. NES also conditions on selected atoms, so it faces the same problem. Its Eqs. 27–28 assume the problem away, and the Eq. 29 term meant to cover it is identically zero (§4.1).

**What does "aligned" mean, and is it realistic?**
It is NEXIS's screening form: once the principal atom is known, no other atom tells more about the factor. For DINES this is equivalent to the atom determining the part of the factor that other atoms depend on (Lemma 5). The atoms of other causes must, in addition, not read changed factors. This is not realistic exactly: NEXIS rejects it at $p<10^{-14}$, while the principal atom carries 94–96% of the information. What the method needs is the realistic version, leaks below $\delta$ (Theorem 3). A leak is the treatment signal left in the carried part times the square root of the share the atoms miss (Proposition 5).

**What happens with feature splitting?**
- *Clean split:* if one factor is spread cleanly over two atoms, both are reported. That inflates the set but makes no false causal claim (Proposition 7).
- *Redundant split:* if a full atom coexists with its parts, each atom is explained by the others, and the factor disappears from the output.

Group atoms using $Z$ only, and score per factor. The real damage comes from mixed atoms, which Theorem 3 covers.

**What about strong effects?**
"Strong" in the harmful sense means predictive of $T$, not large. If colour reveals the treatment, holding colour fixed leaves no control frogs. The direct set is therefore defined over overlap sets only. $P$ stays, and $V$ is still dropped via $P$. Atoms that change only through colour cannot be separated from atoms the treatment changes directly; in the data, colour is the treatment. Such atoms are reported as undetermined (Proposition 4, §2.9). This is the mice "bag" case: NES stopped after the bag neuron because round 2 had no overlap.

**DINES vs NES on data?**
The CelebA comparison is running separately, and its numbers will be added here. It covers the NES designs, DINES at $\delta=0$ and $\delta>0$, NES, IAMB forward only, NEXIS-style forward + terminal, and marginal Bonferroni. On the toy model of Appendix B:
- with exact alignment, DINES recovers the direct set in 99/100 runs at every $n$;
- with noisy atoms at $\delta=0$, exact recovery falls to 84/100 at $n=16{,}000$;
- with noisy atoms at $\delta=0.05$, it stays at 100/100 for $n\ge4000$.

---

## 6. Open questions and conjectures

- **O1 (separators without alignment).** For a general dictionary, does every non-direct atom inside a Markov blanket of $T$ have a separator made of other blanket atoms? This is why assumption L stays for arbitrary dictionaries.
- **O2 (affected-only exemption).** An other cause that reaches only atoms the treatment does not shift probably needs no atom. Not proved in general.
- **O3 (uniformity).** All results are pointwise at fixed $m$. Uniform versions need a strong-faithfulness margin (Zhang & Spirtes, 2003; Uhler et al., 2013), and $\delta>0$ is a step toward one.
- **O4 (large $m$).** With $m\approx10^4$ and $n\approx10^3$, the union bound needs normal-approximation errors of order $o(1/m)$.
- **O5 (carriers without N1).**
- **O6 (choosing $\delta$ automatically).**
- **Conjecture 1 (general leak bound).** §2.7.
- **O7 (leaks of impure atoms).** No bound is known when the condition of Proposition 5 fails.
- **O8 (dictionary trained on the same data).**
- **O9 (observational data).** $T$ is not a root there. With all parents of $T$ in $W$ we expect the same results, but the proof is not written.
- **O10 (effects on spread only).**
- **O11 (checking forward completeness on real dictionaries).**
- **O12 (group tests).** A vector-valued GCM for atom groups is not implemented.
- **O13 (partial overlap).** A population target for sets that reveal $T$ only on part of the units, with a matching finite-sample rule.

**The CelebA benchmark must separate:**
- the original NES design (Smiling has no atom);
- a mediator design;
- a well-aligned other cause and a descendant other cause;
- a hidden common cause (expected to break every method);
- a coarse mediator atom;
- weaker dictionaries (testing whether precision falls with $N$ at $\delta=0$, and whether $\delta>0$ fixes it);
- the null design;
- a treatment-revealing attribute (overlap);
- a clustered version.

---

## 7. References

- Aliferis, C. F., Statnikov, A., Tsamardinos, I., Mani, S., Koutsoukos, X. D. (2010). Local causal and Markov blanket induction for causal discovery and feature selection for classification, Parts I–II. *JMLR* 11.
- Cadei, R., Otchere, F., Tirivayi, N., Angeles Tagliaferro, G., Bargagli-Stoffi, F. J., Locatello, F. (2026). From Tokens to Policy: Causal and Interpretable Heterogeneous Treatment Effects Identification (NEXIS). arXiv 2606.17010.
- Cameron, A. C., Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *J. Human Resources* 50(2).
- Imai, K., Keele, L., Yamamoto, T. (2010). Identification, inference and sensitivity analysis for causal mediation effects. *Statistical Science* 25(1).
- Lauritzen, S. L., Dawid, A. P., Larsen, B. N., Leimer, H.-G. (1990). Independence properties of directed Markov fields. *Networks* 20.
- Lin, W. (2013). Agnostic notes on regression adjustments to experimental data. *Ann. Appl. Stat.* 7(1).
- Lundborg, A. R., Kim, I., Shah, R. D., Samworth, R. J. (2024). The projected covariance measure for assumption-lean variable significance testing. *Ann. Statist.*
- Mencattini, T., Cadei, R., Locatello, F. (2025). Exploratory Causal Inference (NES). arXiv 2510.14073.
- Mooij, J. M., Magliacane, S., Claassen, T. (2020). Joint causal inference from multiple contexts. *JMLR* 21.
- Pearl, J. (2001). Direct and indirect effects. *UAI*.
- Peña, J. M., Nilsson, R., Björkegren, J., Tegnér, J. (2007). Towards scalable and data efficient learning of Markov boundaries. *Int. J. Approx. Reasoning* 45(2).
- Perry, R., von Kügelgen, J., Schölkopf, B. (2022). Causal discovery in heterogeneous environments under the sparse mechanism shift hypothesis. *NeurIPS*.
- Robins, J. M., Greenland, S. (1992). Identifiability and exchangeability for direct and indirect effects. *Epidemiology* 3(2).
- Scheidegger, C., Hörrmann, J., Bühlmann, P. (2022). The weighted generalised covariance measure. *JMLR* 23.
- Schölkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., Bengio, Y. (2021). Toward causal representation learning. *Proc. IEEE* 109(5).
- Shah, R. D., Peters, J. (2020). The hardness of conditional independence testing and the generalised covariance measure. *Ann. Statist.* 48(3).
- Tsamardinos, I., Aliferis, C. F., Statnikov, A. (2003). Algorithms for large scale Markov blanket discovery (IAMB). *FLAIRS*.
- Tsiatis, A. A., Davidian, M., Zhang, M., Lu, X. (2008). Covariate adjustment for two-sample treatment comparisons in randomized clinical trials. *Stat. Med.* 27.
- Uhler, C., Raskutti, G., Bühlmann, P., Yu, B. (2013). Geometry of the faithfulness assumption in causal inference. *Ann. Statist.* 41(2).
- VanderWeele, T. J., Valeri, L., Ogburn, E. L. (2012). The role of measurement error and misclassification in mediation analysis. *Epidemiology* 23(4).
- Zhang, J., Spirtes, P. (2003). Strong faithfulness and uniform consistency in causal inference. *UAI*.

---

## Appendix A. Index of results (with draft-1 numbers)

| Label | Statement | Status | Draft 1 |
|---|---|---|---|
| Lemma 1 | forms of the null; GCM functional; $\rho$ | proved | Lemma 1 (+ item 5) |
| Lemma 2 | folding: which other causes need an atom | proved | new |
| Lemma 3 | separator lemma, ancestral form (a) and (b) | proved | Lemma 5 (smaller separator) |
| Lemma 4 | from factors to atoms under screening (CI and mean) | proved | Lemmas 2–4 (determination replaced by screening) |
| Lemma 5 | screening = determination of the relevant part | proved | new |
| Theorem 1 | identification on overlap sets; $\mathrm{Adj}=J_C$ if $B_Z\in\mathcal{O}$ | proved | Theorem 6 |
| Proposition 1 | Markov boundary keeps other causes | proved | Proposition 7 |
| Proposition 2 | direct set = minimal changed mechanisms | proved | Proposition 8 |
| Proposition 3 | mediation not identified without (a), (b) | proved | new |
| Proposition 4 | a factor copying $T$: split not identified | proved | new |
| Theorem 2 | generic selection guarantee with overlap rule; strict leaks give prob. → 1 | proved | Theorem 11 (generalized) |
| Corollary 1 | statistical target $\mathrm{Adj}$; recall always | proved | Theorem 11 |
| Corollary 2 | causal target $J_C$ | proved | Corollary 12 |
| Theorem 3 | realistic alignment with margin $\delta$ | proved | new (draft 1: heuristic §7.8) |
| Proposition 5 | exact leak identity | proved | new |
| Proposition 6 | linear-Gaussian leak | proved | Conjecture 1 (linear-Gaussian case) |
| Proposition 7 | feature splitting (groups) | proved | O6 (partly) |
| Proposition 8 | carriers identified (exact capture, N1, CF) | proved | Proposition 10 |
| Proposition 9 | full output $\ge1-\alpha-\alpha_2$ | proved | Proposition 13 |
| Proposition 10 | covariates (exact capture) | proved | Proposition 14 |
| Conjecture 1 | general leak bound | conjecture | Conjecture 1 (general case) |
| §2.3 (a)–(f) | what breaks without each assumption | proved by computation | §4.3 |
| §4.2 hypothesis | causes of NES's precision fall | hypothesis | §8.9 |

Section map from draft 1: §6.1 → §2.5; §6.2–6.3 → §2.6; §7.2 → §3.1; §7.4 → §3.2; §7.5 → §3.4; §7.8 → §2.6–2.8.

## Appendix B. Measurements

All runs used CPU only. B.1–B.3 are unchanged from draft 1; B.4 is new (closed-form population computations, each under one minute).

**B.1 Toy DINES simulation** (letters unrelated to the frog).
- *Factor model:* $T\to A\to B\to W\to C\leftarrow T$, $H\to C$, $A\to D$, $L$ isolated, $T\sim\mathrm{Bern}(1/2)$. Equations: $A=0.5T+\varepsilon$, $B=0.8A+\varepsilon$, $W=0.8B+\varepsilon$, $D=0.8A+\varepsilon$, $C=0.5T+0.8W+0.8H+\varepsilon$, with $H,L,\varepsilon\sim\mathcal{N}(0,1)$.
- *Atoms ($m=10$):* one per factor (factor plus $\sigma_\eta\mathcal{N}(0,1)$), plus $0.5A+0.5D+\varepsilon$, $0.5C+0.5H+\varepsilon$, $L+\varepsilon$.
- *Truth:* the direct set is $\{A,C\}$; the blanket is $\{A,C,W,H\}$. Every set has overlap (Gaussian atoms).
- *Test:* GCM with 2-fold cross-fitted logistic propensity and OLS; $\alpha=0.05$; forward gate $\alpha/|R|$; terminal level $\alpha/m$. The margin test is used in both phases when $\delta>0$.
- 100 seeds per cell.

| alignment | $\delta$ | $n$ | exact recovery | non-direct atoms in forward output | false atoms kept | recall |
|---|---|---|---|---|---|---|
| exact ($\sigma_\eta=0$) | 0 | 1000 / 4000 / 16000 | 99 / 99 / 99 | 1.86 / 2.10 / 2.03 | 0.00 / 0.01 / 0.01 | 0.99 / 1.00 / 1.00 |
| noisy ($\sigma_\eta=0.3$) | 0 | 1000 / 4000 / 16000 | 96 / 96 / 84 | 1.56 / 2.15 / 2.36 | 0.01 / 0.04 / 0.17 | 0.98 / 1.00 / 1.00 |
| noisy | 0.05 | 1000 / 4000 / 16000 | 70 / 100 / 100 | 0.35 / 1.64 / 2.00 | 0.01 / 0.00 / 0.00 | 0.85 / 1.00 / 1.00 |
| exact | 0.05 | 1000 / 4000 / 16000 | 70 / 100 / 100 | 0.56 / 1.94 / 2.00 | 0.00 / 0.00 / 0.00 | 0.84 / 1.00 / 1.00 |

False atoms kept at $\sigma_\eta=0.3$, $\delta=0$, $n=16000$: the mixed atom $0.5A+0.5D$ (10 runs), $D$ (4), $W$ (3).

Separator check (exact alignment, $n=20{,}000$, one run), $T$ vs $W$: $p=3.6\times10^{-70}$ given $\emptyset$; $0.14$ given $\{A\}$; $0.15$ given $\{A,H\}$; $0.37$ given $\{A,B\}$; $2.2\times10^{-56}$ given $\{A,C\}$.

**B.2 Closed-form checks** ($n=2\times10^6$): binarized-stratum retained fraction (§2.3(b)); noisy-proxy coefficient (§2.3(c)); hidden-common-cause coefficient (§2.3(a); the coefficient on $T$ is $-\beta/2$ for any $\gamma$). The Principal–Max counterexample (§4.1) used $n=20{,}000$.

**B.3 Saved NES CelebA runs** (read-only): selections from `sneurons_{420..429}.json`; round-2 statistics recomputed with the project's `_stratified_ttests_lotp`; atom–attribute F1 from `experiment_M3_ATE:0.4_size:1000_seed:420.csv`.

**B.4 Population leak computations.** For the toy of B.1, the atoms are Gaussian given $T$ with a $T$-free covariance. So $\theta_{j|A}=\tau_j-\Sigma_{jA}\Sigma_{AA}^{-1}\tau_A$ and $\sigma^2_{j|A}=\Sigma_{jj}-\Sigma_{jA}\Sigma_{AA}^{-1}\Sigma_{Aj}$. $V_A$ is computed by Gauss–Hermite quadrature over the discriminant score, which is $\mathcal{N}(\pm D^2/2,D^2)$ with $D^2=\tau_A^\top\Sigma_{AA}^{-1}\tau_A$, and $\rho$ follows from Proposition 6. Leaks are evaluated at both kinds of separator.

| quantity | $\sigma_\eta=0$ | $\sigma_\eta=0.3$ |
|---|---|---|
| leak $\lvert\rho_{j\mid A_j}\rvert$ for $B$, $D$, $W$, $0.5A{+}0.5D$, $0.5C{+}0.5H$ (full blanket; $\{A,H\}$ for $W$) | 0 (all) | 0.0033, 0.0141, 0.0097, 0.0147, 0.0023 |
| same, at the ancestral separators of Lemma 3 ($\{A\}$ for $B$, $D$, $W$, $0.5A{+}0.5D$; $\{A,W,C,H\}$ for $0.5C{+}0.5H$) | 0 (all) | 0.0150, 0.0150, 0.0097, 0.0157, 0.0023 |
| smallest direct effect over all $2^9$ sets, for $A$ and $C$ | 0.126, 0.128 | 0.120, 0.128 |
| forward completeness at margin $\delta$ holds iff | $\delta<0.099$ | $\delta<0.090$ |
| explaining-away shift $\lvert\rho_{H\mid\{A,C\}}\rvert$ | 0.082 | 0.077 |

Additional checks:
- Proposition 5 identity: both sides agree to 5 decimals at every separator. The factor $\rho_{g|A}$ is 0.011–0.070, and the square-root factor is about 0.2.
- Proposition 6 matrix formula on a random 3-factor instance: 0.143909 from the formula, 0.143909 from direct conditioning. The operator-norm bound gives 1.97, which is valid but loose.
