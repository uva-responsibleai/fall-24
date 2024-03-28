# Introduction and Motivations
With massive amounts of data being required for machine learning, data analysis, and decision-making, data privacy arises as a natural concern as such data often constitutes sensitive information collected from entities like individuals or organizations. While data privacy has been discussed and attempted to be enforced using various methods, it was not until the 2000s that a widely-accepted, mathematically rigorous definition of privacy was adopted.

Pure ($`\varepsilon`$-)differential privacy ($`\varepsilon`$-DP) ensures that the contribution of any one individual/record does not change the output of an algorithm (much). Approximate differential privacy, on the other hand, relaxes pure DP by allowing for a small failure probability $`\delta`$. This allows for more forgiving data perturbations in practice with a low failure probability.

More precisely, given any two neighboring datasets (i.e. datasets that differ by one record) $`D`$, $`D'`$ (henceforth denoted by the relation $`D\sim D'`$), an algorithm $`M:\mathcal{X}^n\to \mathcal{Y}`$ is said to be ($`\varepsilon,\delta`$)-differentially private if it satisfies for some $`\varepsilon>0`$ and $`\delta\in[0,1]`$ and $\forall\ T\in P(\mathcal{Y})$,
$$\Pr[M(X)\in T]\leq\exp(\varepsilon)\Pr[M(X')\in T] + \delta.$$

In practice, given a dataset with $`n`$ records, it is common to set $`\delta<\frac{1}{n}`$. This relaxed definition of differential privacy (DP) comes with its own canonical mechanism, the Gaussian mechanism, much like the Laplace mechanism for $`\varepsilon`$-DP. In addition, as with $`\varepsilon`$-DP, it also comes with the very convenient guarantees of sequential composition, parallel composition, post-processing invariance, and group privacy.

However, it does raise some questions. For instance, the failure probability $`\delta`$, as used in the definition of ($`\varepsilon,\delta`$)-DP seems to suggest that a privacy catastrophe of arbitrary proportions (not ruling out the entire dataset being leaked!) can occur with that probability. However, the means by which ($`\varepsilon,\delta`$)-DP is enforced in practice (thankfully) do not lead to such catastrophic outcomes. This has led to further reformulations of DP that are outside of the scope of this discussion. In addition, researchers have derived tighter composition bounds for ($`\varepsilon,\delta`$)-DP. The following discussion shall touch upon the technical/theoretical aspects of approximate DP, showing how it can be used in practice, and some interesting results. In addition, some interesting mechanisms and algorithms for different tasks like picking the best object out of a collection and applications to mechanism design will be discussed.

# Methods

- Linear/numerical/histogram queries
- Gaussian Mechanism

The Gaussian mechanism is the canonical mechanism for ($`\varepsilon,\delta`$)-DP. Like the Laplace mechanism, it is a global sensitivity method, named after the eponymous quantity that is defined as follows.

Let $`f: \mathcal{X}^n \rightarrow \mathbb{R}^k`$. The global $\ell_2$-sensitivity of $f$ is given by the maximum difference in the output of $`f`$ over the choice of all the pairs of neighboring datasets. More precisely, it is given by
$$\Delta_2^{(f)}=\max_{D\sim D'}\left\|f(D)-f\left(D'\right)\right\|_2.$$

> Include the definition of the Gaussian mechanism here.

Note that the Gaussian Mechanism uses $\ell_2$-sensitivity unlike the Laplace Mechanism, which uses $\ell_1$-sensitivity.

The $\ell_2$-sensitivity might be up to a factor $\sqrt{d}$ less than the $\ell_1$-sensitivity, which lends to the fact that Laplace Mechanism offers a stronger privacy guarantee but has a potentially worse error, compared to the Gaussian Mechanism. 

> The Gaussian Mechanism is $(\varepsilon, \delta)$-differentially private.

- Exponential Mechanism
- Private Multiplicative Weights

Private Multiplicative Weights algorithm is designed to answer a set of linear queries on a database while ensuring differential privacy. The algorithm guarantees that for a database of size $n$, it can answer a set $Q$ of linear queries to accuracy $\alpha$ under $(\epsilon, \delta)$-differential privacy, provided $n=O\left(\frac{\log |Q| \log |X| \log (1 / \delta)}{\alpha^2 \epsilon}\right)$ data points are available.

There are two steps in each iteration which depend on the dataset: 
1. selecting a query which causes the algorithm to err. It uses the exponential mechanism to select a query from the set $Q$ that the current iteration of the algorithm is most inaccurate, using score function $\left|\left\langle q^t, p^t\right\rangle-\left\langle q^t, p\right\rangle\right|$
2. checking how much error this query incurs (and the associated sign). After selecting a query, the algorithm computes $y^t=\left\langle q^t, p^t\right\rangle-\left\langle q^t, p\right\rangle+{\rm Laplace}\left(1 / \varepsilon_0 n\right)$. Based on the magnitude of $y_t$, if the error exceeds a certain threshold $2 \alpha$, the algorithm updates $p_t$ to reduce this error, using function $p_i^{t+1} \propto p_i^t\left(1-s \sqrt{\frac{\ln |\mathcal{X}|}{T}} q_i^t\right)$


- Report Noisy Max
- DP and Mechanism Design

Mechanism Design involves the problem of algorithm design when a self-interested individual controls the input to an algorithm, rather than the designer of that algorithm. This individual has some preferences of outputs which the algorithm maps its inputs to, which can lead to an incentive for the individual to mis-report data so as to get their preferred outcomes. 

An algorithm $A$ is $\epsilon$-differentially private if for every function $f$ and pair of neighboring databases $x, y$:
$exp(-\epsilon)E_{z\sim A(y)}|f(z)|\leq E_{z\sim A(x)}|f(z)|\leq exp(\epsilon)E_{z\sim A(y)}|f(z)|$

In this case, f is a function mapping outcomes to an agent's utility for them. Another way of phrasing this expression is that the mechanism is $\epsilon$-differentially private if an agent's participation in the mechanism does not affect their expected utility by a factor of utility more than $exp(\epsilon)$.

In such a mechanism, agents have private 'types' which determine their utility functions, but they can report any type to the mechanism. Thus an agent could be incentivized to misreport their type to get greater utility. If the dominant (highest incentive) strategy for every agent is to report the agent's true type, then the mechanism is *truthful*, or *dominant strategy truthful* Formally, for a mechanism $M$, truthful reporting is an $\epsilon$-approximate dominant strategy for agent $i$ if for every pair of types $t_i, t\prime_{i}$ and every vector of types $t_{i - 1}$:

$u(t_i, M(t_i, t_{i - 1}) \geq u(t_i, M(t\prime_i, t_{i - 1}))$

If $\epsilon =0$, then the mechanism $M$ is exactly truthful. 

From these definitions, it follows that:

If a mechanism $M$ is $\epsilon$-differentially private, then $M$ is also $2\epsilon$-approximately dominant strategy truthful.

The proposition above makes differential privacy robust as a solution concept. In addition, differential privacy generalizes to group privacy. 

The authors note one drawback of differential privacy: since the outcome of the mechanism is approximately independent of any single agent's report, *any* report is actually the dominant strategy for an agent, rather than truthfully reporting the agent's type. However, the authors also describe situations in which this can be alleviated. 

In a Nash Equilibrium: Suppose each player has a set of actions $\mathcal{A}$, and can choose to play any action $a_i \in \mathcal{A}$. Suppose, moreover, that outcomes are merely choices of actions that the agents might choose to play, and so agent utility functions are defined as $u: \mathcal{T} \times \mathcal{A}^n \rightarrow[0,1]$. Then:

A set of actions $a \in \mathcal{A}^n$ is an $\epsilon$-approximate Nash equilibrium if for all players $i$ and for all actions $a_i^{\prime}$ :
$$u_i(a) \geq u_i\left(a_i^{\prime}, a_{-i}\right)-\epsilon$$

Every agent is simultaneously playing an (approximate) best response to what the other agents are doing, assuming they are playing according to $a$. This work showed that if we could compute an approximate equilibrium of the game under the constraint of differential privacy, then truthful reporting, followed by taking the suggested action of the coordination device would be a Nash equilibrium. 

# Key Findings

# Critical Analysis
