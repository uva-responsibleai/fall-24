# Introduction and Motivations
With massive amounts of data being required for machine learning, data analysis, and decision-making, data privacy arises as a natural concern as such data often constitutes sensitive information collected from entities like individuals or organizations. While data privacy has been discussed and attempted to be enforced using various methods, it was not until the 2000s that a widely-accepted, mathematically rigorous definition of privacy was adopted.

Pure ($`\varepsilon`$-)differential privacy ($`\varepsilon`$-DP) ensures that the contribution of any one individual/record does not change the output of an algorithm (much). Approximate differential privacy, on the other hand, relaxes pure DP by allowing for a small failure probability $`\delta`$. This allows for more forgiving data perturbations in practice with a low failure probability.

More precisely, given any two neighboring datasets (i.e. datasets that differ by one record) $`D`$, $`D'`$ (henceforth denoted by the relation $`D\sim D'`$), an algorithm $`M:\mathcal{X}^n\to \mathcal{Y}`$ is said to be ($`\varepsilon,\delta`$)-differentially private if it satisfies for some $`\varepsilon>0`$ and $`\delta\in[0,1]`$ and $\forall\ T\in P(\mathcal{Y})$,
$$\Pr[M(X)\in T]\leq\exp(\varepsilon)\Pr[M(X')\in T] + \delta.$$

In practice, given a dataset with $`n`$ records, it is common to set $`\delta<\frac{1}{n}`$. This relaxed definition of differential privacy (DP) comes with its own canonical mechanism, the Gaussian mechanism, much like the Laplace mechanism for $`\varepsilon`$-DP. In addition, as with $`\varepsilon`$-DP, it also comes with the very convenient guarantees of sequential composition, parallel composition, post-processing invariance, and group privacy.

However, it does raise some questions. For instance, the failure probability $`\delta`$, as used in the definition of ($`\varepsilon,\delta`$)-DP seems to suggest that a privacy catastrophe of arbitrary proportions (not ruling out the entire dataset being leaked!) can occur with that probability. However, the means by which ($`\varepsilon,\delta`$)-DP is enforced in practice (thankfully) do not lead to such catastrophic outcomes. This has led to further reformulations of DP that are outside of the scope of this discussion. In addition, researchers have derived tighter composition bounds for ($`\varepsilon,\delta`$)-DP. The following discussion shall touch upon the technical/theoretical aspects of approximate DP, discuss how it can be used in practice, and some interesting results on the same.
# Methods 

# Key Findings

# Critical Analysis
