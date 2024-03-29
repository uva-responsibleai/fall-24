# Introduction and Motivations
With massive amounts of data required for machine learning, data analysis, and decision-making, data privacy arises as a natural concern as such data often constitutes sensitive information collected from entities like individuals or organizations. While data privacy has been discussed and attempted to be enforced using various methods, it was not until the 2000s that a widely accepted, mathematically rigorous definition of privacy was adopted.

Pure ($`\varepsilon`$-)differential privacy ($`\varepsilon`$-DP) ensures that the contribution of any one individual/record does not change the output of an algorithm (much). Approximate differential privacy, on the other hand, relaxes pure DP by allowing for a small failure probability $`\delta`$. This allows for more forgiving data perturbations in practice with a low failure probability.

More precisely, given any two neighboring datasets (i.e. datasets that differ by one record) $`D`$, $`D'`$ (henceforth denoted by the relation $`D\sim D'`$), an algorithm $`M:\mathcal{X}^n\to \mathcal{Y}`$ is said to be ($`\varepsilon,\delta`$)-differentially private if it satisfies for some $`\varepsilon>0`$ and $`\delta\in[0,1]`$ and $\forall\ T\in P(\mathcal{Y})$,
$$\Pr[M(D)\in T]\leq\exp(\varepsilon)\Pr[M(D')\in T] + \delta.$$

In practice, given a dataset with $`n`$ records, it is common to set $`\delta<\frac{1}{n}`$. This relaxed definition of differential privacy (DP) comes with its own canonical mechanism, the Gaussian mechanism, much like the Laplace mechanism for $`\varepsilon`$-DP. In addition, as with $`\varepsilon`$-DP, it also comes with the very convenient guarantees of sequential composition, parallel composition, post-processing invariance, and group privacy.

However, it does raise some questions. For instance, the failure probability $`\delta`$, as used in the definition of ($`\varepsilon,\delta`$)-DP seems to suggest that a privacy catastrophe of arbitrary proportions (not ruling out the entire dataset being leaked!) can occur with that probability. However, the means by which ($`\varepsilon,\delta`$)-DP is enforced in practice (thankfully) do not lead to such catastrophic outcomes. This has led to further reformulations of DP that are outside of the scope of this discussion. In addition, researchers have derived tighter composition bounds for ($`\varepsilon,\delta`$)-DP. The following discussion shall touch upon the technical/theoretical aspects of approximate DP, showing how it can be used in practice, and some interesting results. In addition, some interesting mechanisms and algorithms for different tasks like picking the best object out of a collection and applications to mechanism design will be discussed.

# Methods
## Query Types
Discussions regarding differential privacy involve the knowledge of the query involved. Such queries can come in the form of counting queries (viz. querying how many records/individuals in a dataset satisfy a particular property). Another related class of queries are histogram queries, such as taking $`k`$-marginals and releasing the counts of records pertaining to each combination of the attributes considered. It is easy to see that the sensitivity (defined below) for both these queries is 1, as neighboring datasets can only differ in one record.

These are of interest as they form some of the earliest classes of queries considered for analyzing DP guarantees. Consider a **differential attack**: let us say an adversary asks about how many people in a dataset smoke, and then asks about how many people not named Bob smoke. Assume that the name Bob here is unique (or substitute it with any specific name). This can lead to a privacy leakage about Bob's smoking just by responding to these counting queries. Differential privacy helps mitigate such attacks.

## Gaussian Mechanism

The Gaussian mechanism is the canonical mechanism for ($`\varepsilon,\delta`$)-DP. Like the Laplace mechanism, it is a global sensitivity method, named after the eponymous quantity that is defined as follows.

Let $`f: \mathcal{X}^n \rightarrow \mathbb{R}^k`$. The global $\ell_2$-sensitivity of $f$ is given by the maximum difference in the output of $`f`$ over the choice of all the pairs of neighboring datasets. More precisely, it is given by
$$\Delta_2^{(f)}=\max_{D\sim D'}\left\|f(D)-f\left(D'\right)\right\|_2.$$

The Gaussian mechanism is as follows:

Let $f : \mathcal{X}^n → \mathbb{R}^k.$ The Gaussian mechanism is defined as
$$M(X) = f(X) + (Y_1, . . . , Y_k),$$
where the $Y_i$ are independent $N(0, 2 ln(1.25/δ)∆_2^2 /ε^2$) random variables

Note that the Gaussian Mechanism uses $\ell_2$-sensitivity unlike the Laplace Mechanism, which uses $\ell_1$-sensitivity.

The $\ell_2$-sensitivity might be up to a factor $\sqrt{d}$ less than the $\ell_1$-sensitivity, which lends to the fact that Laplace Mechanism offers a stronger privacy guarantee but has a potentially worse error, compared to the Gaussian Mechanism. 

> The Gaussian Mechanism is $(\varepsilon, \delta)$-differentially private.

## Exponential Mechanism

The exponential mechanism is another fundamental technique for achieving differential privacy in the setting of selecting objects. Unlike the Laplace and Gaussian mechanisms, which focus on adding noise to query results, the exponential mechanism selects the best object from a set of objects depending on a specified notion of utility of these objects.

The exponential mechanism takes in the following input:
- A dataset $`X\in\mathcal{X}^n`$  (private)
- A set of objects $`\mathcal{H}`$   (public)
- A score function $`s : \mathcal{X}^n \times \mathcal{H} → \mathbb{R}`$ .  (public)

The only private information is the dataset X. With this in mind, we define the sensitivity(∆) of the score function with respect to the dataset only: 

$$\Delta s = \max_{h\in H} \max_{X,X'\in\mathcal{X}}  |s(X, h) − s(X', h)|$$

The Exponential mechanism is as follows:

Using the inputs defined above, $`X`$, $`\mathcal{H}`$, and $`s`$ the mechanism outputs $h\in H$ with the probability proportional to 
exp($\frac{\varepsilon \mathcal{s}(X,h)}{2\Delta}$)

> **The exponential mechanism is $`\varepsilon`$-differentially private** and that it will **select an object that is comparable in quality to the best choice of object with a small loss**, depending on the value of $`\varepsilon`$, the sensitivity, and the number of candidate objects 

> COMMENT: Add discussion on composition and the statement of the advanced composition theorem. Add report noisy max.


## An online mechanism: Private Multiplicative Weights

Private Multiplicative Weights algorithm is designed to answer a set of linear queries on a database while ensuring differential privacy. The algorithm guarantees that for a database of size $n$, it can answer a set $Q$ of linear queries to accuracy $\alpha$ under $(\epsilon, \delta)$-differential privacy, provided $n=O\left(\frac{\log |Q| \log |X| \log (1 / \delta)}{\alpha^2 \epsilon}\right)$ data points are available.

<p align="center">
  <img src="https://github.com/wenqian-ye/fall-24/assets/32115593/fec0f2d3-6185-4614-a00f-cbf66af84f69" alt="Description of the image" width="48%">
</p>

There are two steps in each iteration which depend on the dataset: 
1. selecting a query which causes the algorithm to err. It uses the exponential mechanism to select a query from the set $Q$ that the current iteration of the algorithm is most inaccurate, using score function $\left|\left\langle q^t, p^t\right\rangle-\left\langle q^t, p\right\rangle\right|$
2. checking how much error this query incurs (and the associated sign). After selecting a query, the algorithm computes $y^t=\left\langle q^t, p^t\right\rangle-\left\langle q^t, p\right\rangle+{\rm Laplace}\left(1 / \varepsilon_0 n\right)$. Based on the magnitude of $y_t$, if the error exceeds a certain threshold $2 \alpha$, the algorithm updates $p_t$ to reduce this error, using function $p_i^{t+1} \propto p_i^t\left(1-s \sqrt{\frac{\ln |\mathcal{X}|}{T}} q_i^t\right)$


## DP and Mechanism Design

Mechanism Design involves the problem of algorithm design when a self-interested individual controls the input to an algorithm, rather than the designer of that algorithm. This individual has some preferences of outputs which the algorithm maps its inputs to, which can lead to an incentive for the individual to mis-report data so as to get their preferred outcomes.  which is the science of designing incentives to get people to do what you
want them to do.

An algorithm $A$ is $\epsilon$-differentially private if for every function $f$ and pair of neighboring databases $x, y$:
$exp(-\epsilon)E_{z\sim A(y)}|f(z)|\leq E_{z\sim A(x)}|f(z)|\leq exp(\epsilon)E_{z\sim A(y)}|f(z)|$

In this case, f is a function mapping outcomes to an agent's utility for them. Another way of phrasing this expression is that the mechanism is $\epsilon$-differentially private if an agent's participation in the mechanism does not affect their expected utility by a factor of utility more than $exp(\epsilon)$.

In such a mechanism, agents have private 'types' which determine their utility functions, but they can report any type to the mechanism. Thus an agent could be incentivized to misreport their type to get greater utility. If the dominant (highest incentive) strategy for every agent is to report the agent's true type, then the mechanism is *truthful*, or *dominant strategy truthful* Formally, for a mechanism $M$, truthful reporting is an $\epsilon$-approximate dominant strategy for agent $i$ if for every pair of types $t_i, t\prime_{i}$ and every vector of types $t_{i - 1}$:

$u(t_i, M(t_i, t_{i - 1}) \geq u(t_i, M(t\prime_i, t_{i - 1}))$

If $\epsilon =0$, then the mechanism $M$ is exactly truthful. 

From these definitions, it follows that:

> If a mechanism $M$ is $\epsilon$-differentially private, then $M$ is also $2\epsilon$-approximately dominant strategy truthful.

The proposition above makes differential privacy robust as a solution concept. In addition, differential privacy generalizes to group privacy. 

The authors note one drawback of differential privacy: since the outcome of the mechanism is approximately independent of any single agent's report, *any* report is the dominant strategy for an agent, rather than truthfully reporting the agent's type. However, the authors also describe situations in which this can be alleviated. 

### Approximately Truthful Equilibrium Selection Mechanisms

In a Nash Equilibrium: Suppose each player has a set of actions $\mathcal{A}$, and can choose to play any action $a_i \in \mathcal{A}$. Suppose, moreover, that outcomes are merely choices of actions that the agents might choose to play, and so agent utility functions are defined as $u: \mathcal{T} \times \mathcal{A}^n \rightarrow[0,1]$. Then:

A set of actions $a \in \mathcal{A}^n$ is an $\epsilon$-approximate Nash equilibrium if for all players $i$ and for all actions $a_i^{\prime}$ :
$$u_i(a) \geq u_i\left(a_i^{\prime}, a_{-i}\right)-\epsilon$$

Every agent is simultaneously playing an (approximate) best response to what the other agents are doing, assuming they are playing according to $a$. This work showed that if we could compute an approximate equilibrium of the game under the constraint of differential privacy, then truthful reporting, followed by taking the suggested action of the coordination device would be a Nash equilibrium. 

To obtain exact truthfulness (i.e. mechanisms that are exactly dominant strategy truthful), the author also mentioned a framework which uses differentially private mechanisms as a building block toward designing exactly truthful mechanisms without money. The idea is to randomize between the exponential mechanism (with good social welfare properties) and a strictly truthful mechanism which punishes false reporting (but with poor social welfare properties). If we mix appropriately, then we will get an exactly truthful mechanism with reasonable social welfare guarantees. One punishing mechanism is ( simple, but not necessarily the best):

The commitment mechanism $`M^P\left(t^{\prime}\right)`$ selects $s \in \mathcal{O}$ uniformly at random and sets $`\hat{R}_i= \{r_i(t_i^{\prime}, s, R_i)\}`$, i.e., it picks a random outcome and forces everyone to react as if their reported type was their true type.
Define the gap of an environment as
$`\gamma=\min _{i, t_i \neq t_i^{\prime}, t_{-i}} \max _{s \in \mathcal{O}}(u(t_i, s, r_i(t_i, s, R_i))-u(t_i, s, r_i(t_i^{\prime}, s, R_i)))`$
i.e., $\gamma$ is a lower bound over players and types of the worst-case cost (over $s$) of mis-reporting. Note that for each player, this worst-case is realized with probability at least $1/|\mathcal{O}|$. 


# Key Findings
In examining the various facets of differential privacy, especially the $\varepsilon$-differential privacy ($\varepsilon$-DP) and the ($\varepsilon,\delta$)-differential privacy ($\varepsilon,\delta$)-DP, alongside mechanisms like the Gaussian and Laplace mechanisms designed to enforce these privacy standards that protects individual privacy in data analysis, several key findings such as theoretical baselines of differential privacy and its practical implications and potential limitations can be found.

## Efficacy of Differential Privacy Mechanisms

One of the pivotal findings is the effectiveness of these differential privacy mechanisms, particularly the Gaussian mechanism, in protecting individual data. The Gaussian mechanism operates by adding noise calibrated to the $\ell_2$ sensitivity of a function, $\Delta_2f$, which is a measure of the maximum change in the function's output that any single individual's data can cause. This is formalized as: $`\sigma\geq c\Delta_2 f/\varepsilon`$
for $\varepsilon$ within (0, 1) and $\delta$ satisfying $c^2 > 2 \ln(1.25/\delta)$. This mechanism ensures $(\varepsilon, \delta)$-differential privacy, which offers a quantifiable foundation for privacy-preserving data analysis. It ensures that the presence or absence of any single data point does not significantly alter the outcome of data analyses. This is crucial in an era where data breaches are increasingly common, and traditional data protection methods have shown limitations.

## Flexibility and Practical Application

The properties of sequential and parallel composition in differential privacy enable complex, multi-step data analysis processes to maintain privacy guarantees throughout. Composition theorems, as discussed by Dwork and Roth, elaborate on how differential privacy guarantees degrade with the sequential application of differential privacy mechanisms. An essential part of these theorems is understanding how privacy parameters ($\varepsilon$ and $\delta$) adjust in composite analyses, ensuring that privacy guarantees remain intact across multiple queries or analyses on the same dataset. The balance between privacy and utility is particularly evident in the setting of the privacy parameters, where $\delta$ represents a small probability of privacy breach, allowing for practical utility at the expense of a negligible risk. This compromise facilitates the use of differential privacy in a variety of settings, extending its applicability beyond strictly theoretical frameworks.

## Composition and Sequential Analysis

Another significant finding is the composability properties of differential privacy, including sequential and parallel composition. These properties enable the application of differential privacy to complex data analysis workflows, where multiple differential privacy mechanisms may be applied 
with adjustments to the privacy budget across multiple analyses either in sequence or in parallel. This composability is instrumental in ensuring that the overall data analysis process remains privacy-preserving, even when composed of multiple steps.

## Challenges

While differential privacy provides a well-constructed framework for protecting individual privacy, it also introduces certain challenges. The trade-off between privacy and utility is a constant balancing act; achieving higher levels of privacy often results in decreased accuracy of the data analysis results. Additionally, the setting of parameters ($\varepsilon$ and $\delta$) requires careful consideration, as it directly impacts the level of privacy and utility.

Moreover, the failure probability $\delta$ in ($\varepsilon,\delta$)-DP introduces a theoretical possibility of privacy breaches, although it is mentioned with a low probability, this aspect calls for further research into mechanisms that can minimize this risk while maintaining practical utility as mentioned by Joseph and Chiké.

Brought up by Dwork and Roth, it is important to note that as data continues to play a pivotal role in decision-making across various sectors, the demand for privacy-preserving data analysis methods will only increase. Future research directions may include the development of new differential privacy mechanisms that offer better trade-offs between privacy and utility, the refinement of parameter setting methods to ease the application of differential privacy in practice, and the exploration of differential privacy in emerging fields such as machine learning and artificial intelligence.

# Critical Analysis

### Lectures 5 - 8 by Gautam Kamath

Each of the lecture notes provided a strong overview of their respective differential privacy topic, going very detailed into how they derived and proved their theorems. We found these readings, more some than others, to be incredibly dense technically especially when only having introductory knowledge of Differential Privacy. The author gave useful practical scenarios and applications for certain mechanisms, especially for Lectures 5 and 7, but the notes could have benefited from more real-world and illustrative examples. 

### [Sections 3.3, 3.4, 10.1-10.2 of the Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)

Interesting to see how they were able to incorporate differential privacy as a tool for mechanical design. 

Incorporating different fields of study such as game theory, which is talked about more in the field of Economics, into how we view differential privacy is a compelling avenue of research to find better ways for us to quantify and control privacy loss in certain situations.

