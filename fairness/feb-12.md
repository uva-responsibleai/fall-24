
# Introduction
<p align="justify">
Algorithm has been used regularly by the judges to decide whether defendants should be detained or released while awaiting trial. Although judges are not only dependent on the decision made by the algorithm, they frequently seek help to conclude a decision. Therefore, a high-risk classification, particularly a high-risk misclassification, may have a direct adverse impact on a criminal defendant’s outcome. Therefore, fairness in such algorithms is necessary to reduce such adversity. There are different definitions of fairness in algorithms such as statistical parity, calibration, predictive parity, etc. One of the most popular algorithms that judges use is COMPAS. However, studies from [1] and [2] show that there are racial biases in this algorithm. Both studies indicate that in Broward County, black defendants are substantially more likely to be classified as high risk. [1] through their case studies, show the disparities in algorithms. [2] reformulates algorithmic fairness as constrained optimization with the objective of maximizing public safety while satisfying formal fairness constraints designed to reduce racial disparities. However, they also mentioned that there is tension between improving public safety and satisfying prevailing notions of algorithmic fairness. Therefore, maintaining algorithmic fairness, safety, and model accuracy is challenging. Studies from [3] showed that there is no algorithm or method that can satisfy three fairness conditions. The studies refer to the three conditions of fairness calibration within a group, balance for the negative class and balance for the positive class, which can be achieved in specific cases. [4] deeply analyse the impossibilities of fairness. The study showed that fairness can be guaranteed only with very strong assumptions about the world: namely, that “what you see is what you get,” i.e., that we can correctly measure individual fitness for a task regardless of issues of bias and discrimination. They complement this with an impossibility result, saying that if this strong assumption is dropped, then fairness can no longer be guaranteed.
</p>

# Motivation
<p align="justify">
The notion of fairness in the context of artificial intelligence and machine learning is a multi-faceted and complex problem that consists of several criteria and conditions. These papers discuss the need to quantify algorithmic fairness, talking about the tradeoffs that must be kept in mind, the limitations of such algorithms, and cases where achieving complete fairness can seem impossible. 
Recently, the world of computer science is engaged in a debate regarding fairness in decision-making. Two perspectives are key to this debate: individual fairness and group fairness. Assigning the right amount of weight to each group or individual within a population to ensure unbiased results is still something that demands research. These papers, in general, emphasize the need to ensure that the tools employed by such algorithms are free from quantifiable biases that could result in differential impacts in the contexts in which they are utilized. Two of the papers [3, 4] describe the superficial conditions under which complete fairness can be achieved - though these are extremely ideal cases.  One paper [1] explains through an empirical case study the limitations of existing fairness algorithms and the need to eliminate quantifiable biases. The motivation behind [2] is to present the costs associated with algorithmic decision making and how satisfying common definitions of fairness may sacrifice some degree of public safety. Friedler et al. [4] mathematically analyzed the individual and group fairness, uncovering the assumptions needed to achieve each of the fairness definitions. The trade-off between individual equity and group non-discrimination is further illustrated, as different equity mechanisms need to be grounded in conflicting worldviews.
</p>

# Methods
<p align="justify"> 

## Fairness definition

There are multiple definition of algorithmic fairness. Common definitions of fairness are statistical parity, conditional statistical parity, and predictive equality. Assume that an algorithm makes decision about the defendants should be detained or release. We denote the decision as $d(x)$, where $x$ is the defendant attributes that are considered for the decision making. We assume that $X$ is the random variable that on values $X = x$ for an individual drawn randomly from the the population of defendants for whom pretrial
decisions must be made. We can formally define the different definition of fairness as follows: 

1. **Statistical Parity**: denotes for an equal proportion of individuals from different groups to be subjected to a specific decision. For instance, equal detention rates among different racial groups. Assume that a person x belongs to the group $g(x)$. For this, statistical parity is defined as follows:
   
    $$E[d(X) | g(X)] = E[d(X)]$$
   
2. **Conditional statistical Parity**: indicates that controlling for a limited set of “legitimate” risk factors, an equal proportion of defendants are detained within each race group. Let, $l(x)$ is the legitimate risk associated with person $x$.

   $$E[d(X) | l(X), g(X)] = E[d(X)]$$

3. **Predictive equality**: means that the accuracy of decisions is equal across race groups, as measured by false positive rate (FPR). This condition means that among defendants who would not have gone on to commit a violent crime if released, detention rates are equal across race groups. Assume that $Y$ is a random variable where $y = 1$ for those defendants who would have committed a violent crime if released, and $y = 0$ otherwise. Formally, predictive equality means

   $$E[d(X) | Y=0, g(X)] = E[d(X) |Y=0]$$
</p>

## Formulation of algorithomic fairness using constrainted optimization

As there exists multiple definition of fairness, there should be some criteria to select one. In making this choice, authors [2] assume policymakers seek to maximize a specific notion of utility. In context of pretrial setting, there must be a balance between the benefit of preventing violent crime committed by released defendants and the social and economic costs of detention. To capture these costs and benefits, authros of [2] define the immediate utility of a decision rule as follows

**Immediate utility** For $c$ a constant such that $0 < c < 1$, the immediate utility of a decision rule $d$ is:
   
   $$u(d,c) = E[Y(d(X))-c d(X)]$$

This can be further written as $u(d,c)=E[d(X)(P_{Y|X}-c)]$, where $P_{Y|X}$ denotes the probability that the defendant will commit crime if released. This shows that it is
beneficial to detain an individual precisely when $P_{Y|X}>c$.

Optimal decision rules $( d^* )$ that maximizes $u(d,c)$ under various fairness condition are unique and have following form:

__Unconstrained Optimum__
 $d^*(X) = 1$ if $P_{Y|X} \geq c$ and $0$ otherwise

__Statistical Parity__
$d^*(X) = 1$ if $P_{Y|X} \geq t_{g(X)}$ and $0$ otherwise, where $t_{g(X)}$ are constants that depend only on group membership. The optimal rule satisfying predictive equality
takes the same form, though the values of the group-specific thresholds are different.

__Predictive Equality__ The optimal decision rule $d^*(X)$ satisfying predictive equality takes a similar form to the statistical parity case, with group-specific thresholds.

__Conditional Statistical Parity__ $d^*(X) = 1$ if $P_{Y|X} \geq t_{g(X),l(X)}$ and $0$ otherwise, where $t_{g(X),l(X)}$ are constants dependent on group membership and legitimate attributes.

In summary, for statistical parity and predictive equality, the optimal algorithms detain defendants when $P_{Y|X}$ exceeds a group-specific threshold. For example, black defendants might be detained if $P_{Y|X}\ge 0.2$, and white defendants detained if $P_{Y|X}\ge 0.1$. The exact thresholds for statistical parity differ from those for predictive equality. For conditional statistical parity, the thresholds in the optimal decision rule depend on both group membership and the “legitimate” factors.

## Fairness Properties for Risk Assignments
Risk assignments in alogrithm is challenging. Model assign risk based on different assumptions which are case specific. Authors [3] identifies three properties of fairness for risk assignments (s(X)) to be fair as follows:

**Calibration within groups** means that among defendants with a given risk score, the proportion who reoffend is the same across race groups. Formally, calibration means,
  
  $p(Y=1|s(X),g(X))=p(Y=1|s(X))$
  
**Balance for the negative class** requires that the average score assigned to individuals of one group who belong to the negative class is the same as the average score assigned to individuals of another group who belong to the negative class.
  
  $E[s(X) | Y=0, g(X)] = E[s(X)|Y=0] $
  
**Balance for the positive class** symmetrically requires that the average score assigned to individuals of one group who belong to the positive class is the same as the average score assigned to individuals of another group who belong to the positive class.
  
  $E[s(X) | Y=1, g(X)] = E[s(X)|Y=1] $


## Mathematical Framework of Fairness
The authors in [4] introduced a mathematical framework for the interpretation of fairness, which includes three major metric spaces:
- Construct space (CS): a metric space consisting of individuals and a distance between them, which captures unobservable but meaningful variables for the prediction 
- Observed space (OS): in reality, this is what we can measure or observe
- Decision space (DS): the space of outcomes

Here is an example of the three spaces:
<p align="center">
  <img src="img/impossible_space_example.png" alt="Description of the image">
</p>

The authors also formally defined fairness and non-discrimation:
- The mathematical definition of fairness is
<p align="center">
  <img src="img/impossible_fairness_def.png" alt="Description of the image">
</p>

- The mathematical definition of non-discrimination is
<p align="center">
  <img src="img/impossible_non_dis_def.png" alt="Description of the image">
</p>

- - where the group skew measures the way in which group (geometric) structure might be distorted between
spaces.

While the above definitions conceptually describe the requirements of fairness and non-discrimation in the mapping from CS to DS, the real-world fairness algorithms are typically deployed on the tranformation from OS to DS. The authors further defined mathematically two major fairness mechnisms in real-world applications:
- The mathematical definition of individual fairness mechanism (IFM) is
<p align="center">
  <img src="img/impossible_IFM_def.png" alt="Description of the image">
</p>

- The mathematical definition of group fairness mechanism (GFM) is
<p align="center">
  <img src="img/impossible_GFM_def.png" alt="Description of the image">
</p>
The individual fairness mechanism asserts that the mechanism for decision making treats people similarly if they are close, and can treat them differently if they are far, in the observed space. The group fairness mechanism asserts that the decision mechanism should treat all groups the same way. 

Two frequently used but incompatible worldviews are formally introduced by the authors, including:
- what you see is what you get (WYSIWYG): There exists a mapping $f : CS \rightarrow OS$ such that the distortion ρf
is at most $\epsilon$ for some small $\epsilon > 0$. Or equivalently, the distortion $\rho$ between $CS$ and $OS$ is at most $\epsilon$.
- structural bias: The metric spaces $CS = (X, dX)$ and $OS = (Y, dY)$ admit $t$-structural bias if the
group skew $\sigma(X , Y) > t$.

# Key Findings

### Unfairness in existing COMPAS algorithm


 
***Uncovering Inequity: Assessing Disparate Impact in Decision-Making***


To assess fairness, we can use the following equation which relates false negative rate and false positive rate. Given that S={HR,LR} is random variable indicates the risk factor. There are two possible values of S that is HR (high risk) and LR (low risk). FNR and FOR are defined as follows:

   $FNR = p(S = LR | Y = 1)$
   
   $FPR = P(S = HR | Y = 0)$

<p align="center">
  <img src="img/img1.png" alt="Description of the image">
</p>

This straightforward equation implies that a test-fair score S cannot have identical false positive and negative rates across groups when the recidivism incidence varies.
This finding helps us to comprehend why the ProPublica writers saw significant differences in FPR and FNR between defendants who were Black and those who were White.

<p align="center">
  <img src="img/img2.png" alt="Description of the image">
</p>

Disparate impact under penalty policies where high-risk assessments result in stricter penalties is the consequence of differences in false positive and false negative rates between groups.
The higher recidivism prevalence group will often have a higher FPR and a lower FNR when employing a test-fair RPI in populations where recidivism prevalence varies across groups. Furthermore, it's evident that this would lead to harsher punishments for defendants in the higher prevalence group, both for repeat offenders and non-repeaters.


<p align="center">
  <img src="img/img3.png" alt="Description of the first image" width="48%">
  <img src="img/img4.png" alt="Description of the second image" width="48%">
</p>



The figure above (left) displays a plot of false positive rates for defendants charged with misdemeanor offenses, the least serious category of criminal offenses, over various ranges of prior counts. It is evident that disparities in false positive rates between White and Black offenders continue throughout previous record segments.

In the above figure (right), COMPAS decile score distribution isn't normal, a better approach is to equate % non-overlap with total variation distance. This involves comparing score distributions across races and recidivism outcomes, allowing for a precise boundary on disparate impact. Essentially, it shifts from a standard normal assumption to a more robust method reflecting the actual score distribution characteristics.

***Fairness at the Cost of Safety: Striking a Delicate Balance***


Author [2] show that even though fairness can be achieved, the risk of crime in society can increase. Using the constraints optimization of algorithmic fairness, the effect can be mitigated. However, there exists a trade-off between risk and fairness. 

<p align="center">
  <img src="img/alg-decision-2.png" alt="Description of the first image" width="48%">
</p>


For each fairness constraint, Figure above shows that violent recidivism increases while low risk defendants are detained. For example, for statistical parity, 17% of detained defendants are relatively low risk. Therefore, to enforece fairness, an equal number of high-risk defendants are should be released. This will lead to an estimated 9% increase in violent recidivism among released defendants. There are thus tangible costs to satisfying popular notions of algorithmic fairness. Similarly, the risk for predictive equality is 7% whereas the number is 4% for conditional statistical parity.


<p align="center">
  <img src="img/alg-decision-1.png" alt="Description of the first image" width="48%">
</p>
A decision rule constrained to satisfy statistical parity, conditional statistical parity, or predictive equality reduces public safety. However, a single-threshold rule that maximizes public safety generally violates all of these fairness definitions. This happens because white and black defendants in Broward County have different distributions of risk, $P_{Y|X}$, as shown in above figure. Specifically, a larger portion of black defendants have relatively high risk scores, in part because black defendants are more likely to have prior arrests, which is a strong indicator of reoffending. Importantly, while an algorithm designer can choose different decision rules based on these risk scores, the algorithm cannot alter the risk scores themselves, which reflect underlying features of the population of Broward County.







***Fairness: Yet Elusive in Practice***


Study indicate that it is challenging to achieve fairness in risk assignments to individuals according to three fundamental conditions simultaneously, except in highly constrained scenarios as follows:

* __Perfect Prediction__: This is a special case where there exists a definite, known risk probability that is either 0 or 1. In other words, we don’t have a continuous range of values (no uncertainty). This case satisfies all the above 3 conditions.
* __Equal Base Rates__: If the proportion of members that belong to a particular class (or condition) of each group is equal (for e.g., if 20% of people in Group A belong to a positive class and 20% of people in Group B also belong to the same positive class), all the above 3 conditions will be satisfied.

This difficulty persists regardless of the method used for risk calculation. The authors illustrate this challenge with an example involving gender-based differences in disease carrier rates. They suggest further research directions to explore computational feasibility and various cost considerations for false positives and false negatives. They also provide mathematical proofs demonstrating the existence of fair risk assignments in specific cases


### Trade-off between Individual Fairness and Group Non-discrimination

***Individual fairness mechanisms guarantee fairness under WYSIWYG.***


Fairness can be guaranteed only with very strong assumptions about the world: namely, that “what you see is what you get,” i.e., that we can correctly measure individual fitness for a task regardless of issues of bias and discrimination.
- Under the WYSIWYG with error parameter $\delta$, an $\text{IFM}_{\delta'}$ will guarantee $(\epsilon, \epsilon')$-fairness for some function $f$ such that $\epsilon'=f(\delta,\delta')$.

***Group fairness mechanisms guarantee non-discrimination under WAE.***


Building non-discriminatory decision algorithms is shown to require a different worldview, namely that “we’re all equal,” i.e., that all groups are assumed to have similar abilities with respect to the task in the construct space.
- Under the WAW, a GFM with parameter $\epsilon'$ guarantees $(\max(\epsilon,\epsilon')/\delta)$-nondiscrimination.


***Conflicting worldviews necessitate different mechanisms.***

<p align="justify">
However, the authors found the two worldviews conflict with each other and the two fairness mechanisms cannot be achieved simultaneously.
- Failure of fairness: If there is structural bias in the decision pipeline, no mechanism can guarantee fairness. Fairness can only be achieved under the WYSIWYG worldview using an individual fairness mechanism, and using a group fairness mechanism will be unfair within this worldview.
- Failure of non-discrimination: While group fairness mechanisms were shown to achieve nondiscrimination under a structural bias worldview and the we’re all equal axiom, if structural bias is assumed, applying an individual fairness mechanism will cause discrimination in the decision space whether the we’re all equal axiom is assumed or not.
</p>

# Critical Analysis

**Thoughts for Inherent Tradeoffs in Risk Assignment**

<p align="justify">
The study suggests the necessity of equalizing error rates across distinct groups in certain risk assessment scenarios, despite the potential introduction of predictive bias. Merely aligning overall error rates may not suffice, as disparities can persist within specific categories, such as prior record scores. It becomes crucial to determine the granularity at which balancing error rates is essential for fairness.
</p>

<p align="justify">
One aspect of the study that could be subject to further scrutiny is the generalizability of the results. While the example provided regarding gender-based differences is illustrative, it's important to consider the broader applicability of the findings across different contexts and demographic characteristics. Additionally, the study acknowledges the need for further research to explore computational feasibility and various cost considerations for false positives and false negatives. This recognition underscores the importance of ongoing inquiry and refinement in this area.
</p>

<p align="justify">
Moreover, the mathematical proof provided by the paper authors adds rigor to their conclusions and contributes to the academic discourse on fairness in risk assignments. However, mathematical models, while powerful tools, can have limitations in capturing the full complexity of real-world scenarios. Therefore, it's crucial for future research to complement mathematical analyses with empirical studies (such as in paper [1]) to gain a comprehensive understanding of fairness in risk assessment.
</p>

**Challenges of algorithomic fairness**

<p align="justify">
Author in [2] proposed constrainted optimization for algorithmic fairness. However, during the problem formulation, there are several assumption that differs from practical worlds. For instance, 

- All violent crime is assumed to be equally costly. However, in practice magnitude of all offense are not similar. For instance, bank robbery and over-speeding are not equal costly in society.
- The cost of detaining every individual is assumed to be constant, without regard to personal characteristics. This assumption often does not hold. In practice, however, it is often difficult to approximate individualized costs and benefits of detention. This can significantly effect the fairness of the algorithm.
- Risk scores might also fail to accurately capture costs in specific, idiosyncratic cases. For instance, detaining a defendant who is the sole caretaker of her children arguably incurs higher social costs than detaining a defendant without children. However, adding such dicretion potentially introduce bias. Therefore, this should be handled more carefully. 
- The immediate utility of a decision rule might not an effective measurement for long-term costs and benefits. For instance, in the context of credit extensions, offering
loans preferentially to minorities might ultimately lead to a more productive distribution of wealth, combating harms from historical under-investment in minority communities.
- There are some decisions that generally are thought of as group rather than individual choices. For example, when universities admit students, they often aim to select the best group, not simply the best individual candidates, and may thus decide to deviate from a single-threshold rule in order to create diverse communities with varied perspectives and background.
</p>


**On the (im)possibility of fairness**

<p align="justify"> 
The paper [4] offers a deep dive into the puzzle of achieving fairness in algorithmic decision-making. It thoroughly outlines the conflict between different fairness criteria, demonstrating how attempts to satisfy one criterion can inadvertently violate another. This conflict is framed within the broader context of societal structures and the inherent biases embedded within them. By examining the mathematical foundations of these issues, the authors present a compelling argument that fairness cannot be a one-size-fits-all concept but is instead a multifaceted goal that requires careful consideration of the specific context and values at stake.
</p>

<p align="justify">
In their exploration, the authors introduce a framework that distinguishes between construct space (the ideal measurements of attributes), observed space (the imperfect real-world measurements), and decision space (the outcomes of algorithmic decision-making). This framework is crucial for understanding how discrepancies and biases can arise at each stage of the decision-making process. The paper highlights the importance of transparency about the assumptions and values that underlie different fairness definitions, arguing that without explicit recognition of these assumptions, discussions about fairness are likely to be unproductive. This nuanced approach emphasizes the role of societal norms and values in shaping what is considered fair, suggesting that any attempt to address fairness must engage with these broader societal factors.
</p>

<p align="justify">
Lastly, the paper critically examines the implications of its findings for the development and deployment of algorithmic systems. It suggests that the quest for fairness is not merely a technical challenge but also a deeply ethical one that requires ongoing critical reflection and dialogue among developers, policymakers, and affected communities. The authors advocate for a more collaborative approach to algorithmic fairness, one that involves diverse stakeholders in defining and pursuing fairness goals. This call to action underscores the need for interdisciplinary efforts to tackle the complexities of fairness, highlighting the limitations of purely technical solutions and the importance of integrating ethical considerations into the design and implementation of algorithmic systems.
</p>

## Reference
[1]. Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. A. Chouldechova, 2016

[2]. Algorithmic decision making and the cost of fairness. S. Corbett-Davies, E. Pierson, A. Feller, S. Goel, A. Huq, 2017

[3]. Inherent Trade-Offs in the Fair Determination of Risk Scores. J. Kleinberg, S. Mullainathan, M. Raghavan, 2017

[4]. On the (im)possibility of fairness. S.A. Friedler, C. Scheidegger, S. Venkatasubramanian, 2017
