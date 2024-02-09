# Introduction
The main skeleton of this summary is based on reading [5], while we also discuss reading [6], [7], [8] as different cases of statistical non-discrimination criteria.
## Statistical Decision Theory and Terminologies
Statistical decision theory classification is the foundation of supervised machine learning. 
The goal of classiﬁcation is to determine a plausible value for an unknown target $Y$ given observed covariates $X$. The function $f$ that maps our covariates into our guess $\widehat{Y}$ is called a classiﬁer, or predictor.
Under the standard supervised learning setting there are a couple of statistical terms and measures that we want to cover: 

- **Accuracy:** the probability of correctly predicting the target variable $P(Y = \widehat{Y})$
- **Optimal Classfier\:** any classiﬁer that minimizes the expected loss $E(\ell (Y, \widehat{Y}))$ where $\ell (Y, \widehat{Y})$ is the loss function. This objective is called classiﬁcation risk and risk minimization refers to the optimization problem of ﬁnding a classiﬁer that minimizes risk.
- **Risk score:** the risk score is sometimes called Bayes optimal. It minimizes the squared loss  $E (Y − r ( X ))^2$ 
among all possible real-valued risk scores $r(X)$.

Here are the common classification criteria based on the conditional probability $P(\text{event} | \text{condition})$.

| Event | Condition | Resulting notion $P(\text{event} \mid \text{condition})$ |
| :--- | :---: | :--- |
| $\widehat{Y}=1$ | $Y=1$ | True positive rate, recall |
| $\widehat{Y}=0$ | $Y=1$ | False negative rate |
| $\widehat{Y}=1$ | $Y=0$ | False positive rate |
| $\widehat{Y}=0$ | $Y=0$ | True negative rate |
| $Y=1$ | $\widehat{Y}=1$ | Positive predictive value, precision |
| $Y=0$ | $\widehat{Y}=0$ | Negative predictive value |

**ROC Curve:** a curve in a two-dimensional space where the axes correspond to true positive rate and false positive rate. The ROC curve is often used to see how predictive our score is of the target variable. A common measure of predictiveness is the area under the curve (AUROC), which equals the probability that a random positive instance gets a score higher than a random negative instance.

![ROC Curve](https://fairmlbook.org/assets/3-3-roc_curve_1.svg)

**Group Membership:** Group membership refers to the assignment of an individual into a particular group based on characteristics that are specific to that group, in accordance with widely held intersubjective definitions.

## Discrimination
In order to achieve fairness, we must first look at discrimination on the basis of membership in speciﬁc groups of the population. 
It is concerned with certain protected categories including *race, sex (which extends to sexual orientation), religion, disability status, and place of birth*. 
These categories are subject to change over time as we progress with our definition of what is ethical in society. 
The methods that will be discussed aim to minimize discrimination, however do not address intersectionality or the disadvantage that members of multiple protected categories face.

## No Fairness through Unawareness
Removing or ignoring sensitive attributes or protected categories is ineffective and in some cases harmful, due to redundant encodings in the feature space which is especially true in the case of large feature sets. 
These redundant encoding allow ways to predict the protected attributes from other features even when they have been removed. 
Redundant encodings can be neutralized by implementing statistical parity but this is proven later on to be insufficient as statistical parity introduces other problems and does not fully solve fairness. 
Knowing that there is “no fairness through unawareness,” we can also believe the inverse that there is “fairness through awareness.” (Reading [6])

## Motivations

The overall goal for the collection of papers we reviewed was to address the concept of fairness in classification, with an emphasis on preventing discrimination (however it is defined) while maintaining utility for the classifier. 
This is the motivating force behind creating a well-defined statistical non-discrimination criteria. 
One that will help mitigate existing biases and discrimination when making decisions and help promote fairness, equity, and accountability in machine learning models.

### Statistical non-discrimination criteria
Statistical non-discrimination criteria aim to deﬁne the absence of discrimination in terms of statistical expressions involving random variables describing a classiﬁcation or decision making scenario.

Formally, statistical non-discrimination criteria are properties of the joint distribution of the sensitive attribute $A$, the target variable $Y$, the classiﬁer $\widehat{Y}$ or score $R$, and in some cases also features $X$. This means that we can unambiguously decide whether or not a criterion is satisﬁed by looking at the joint distribution of these random variables.

Researchers have proposed dozens of different criteria, each trying to capture different intuitions about what is fair. 
Broadly speaking, different statistical fairness criteria all equalize some group dependent statistical quantity across groups deﬁned by the different settings of A. Simplifying the landscape of fairness criteria, we can say that there are essentially three fundamentally different ones. Each of these equalizes one of the following three statistics across all groups:


- **Independence ($R \perp A$):** sensitive attribute is statistically independent to the score
requires the acceptance rate  to be the same in all groups
- **Separation ($R \perp A \mid Y$):** sensitive attribute is statistically independent to the score given the target variable
requires that all groups experience the same false negative rate and same false positive rate (error rate parity)
- **Sufficiency ($Y \perp A \mid R$):** sensitive attribute is statistically independent to the target variable given the score
requires a parity of positive/negative predictive values across all groups of for all scores of R


# Methods
In this section, we dive into these basic fairness criteria and discuss the limitations of each method.

## Common Fairness Criteria
### Demographic Parity / Statistical Parity

The condition for demographic parity is, for all $a, b \in A$
$$P(\widehat{Y}=1 \mid A=a)=P(\widehat{Y}=1 \mid A=b)$$
- Demographic parity requires equal proportion of positive predictions in each group ("No Disparate Impact").
- The evaluation metric requiring parity in this case is the prevalence.

#### Limiations of Statistical Parity
***glass cliff***: women and people of color are more likely to be appointed CEO when a ﬁrm is struggling. 
When the ﬁrm performs poorly during their tenure, they are likely to be replaced by White men.

### Accuracy Parity
The condition for accuracy parity is, for all $a, b \in A$
$$P(\widehat{Y} =Y \mid A=a)=P(\widehat{Y}=Y \mid A=b) .$$
- Accuracy parity requires equal accuracy across groups.
- The evaluation metric require parity in this case is the accuracy.

### Equalized Odds

Equality of Odds is satisfied when both True Positive Parity and False Positive Parity are satisfied.

The condition for True Positive parity is, for all $a, b \in A$
$$P(\widehat{Y}=1 \mid Y=1, A=a)=P(\widehat{Y}=1 \mid Y=1, A=b) .$$

Equalized odds enforces that the accuracy is equally high in all demographics, punishing models that perform well only on the majority. 
Reading [8] provides a possible relaxation of equalized odds, which require non-discrimination only within the “advantaged” outcome group. 
That is, to require that people who pay back their loan, have an equal opportunity of getting the loan in the ﬁrst place (without specifying any requirement for those that will ultimately default).

**Definition (Equal oppotunity).** We say that a binary predictor $\widehat{Y}$ satisfies equal opportunity with respect to $A$ and $Y$ if $P(\widehat{Y}=1 \mid A=0, Y=1)= P(\widehat{Y}=1 \mid A=1, Y=1)$.

### Predictive Parity

Predictive value parity is satisfied when both PPV-parity and NPV-parity are satisfied.

The condition for PPV-parity is, for all $a, b \in A$
$$P(Y=1 \mid \widehat{Y}=1, A=a)=P(Y=1 \mid \widehat{Y}=1, A=b)$$

### Calibration
Sufficiency is closely related to an important notion called calibration. In some applications it is desirable to be able to interpret the values of the score functions as if they were probabilities. The notion of calibration allows us to move in this direction. Restricting our attention to binary outcome variables, we say that a score $R$ is calibrated with respect to an outcome variable $Y$ if for all score values $r \in[0,1]$, we have
$$(Y=1 \mid R=r)=r$$
Calibration by group implies sufficiency. The converse is necessarily not true.

### These metrics cannot hold simultaneously!
If the prevalence of $Y$ across $A$ is not equal, then:

1. If $A$ and $Y$ are not independent, then Demographic Parity and Predictive Parity cannot simultaneously hold.
2. If $A$ and $\widehat{Y}$ are not independent of $Y$, then Demographic Parity and Equalized Odds Parity cannot simultaneously hold.
3. If $A$ and $Y$ are not independent, then Equalized Odds and Predictive Parity cannot simultaneously hold.


## Individual Fairness and Group Fairness

**Individual Fairness:** concept that requires that similar individuals should be treated similarly.
Mathematically, this concept is formalized as a Lipschitz condition on the classifier [6]. A mapping $M: V \rightarrow \delta(A)$ satisfies the $(D, d)$-Lipschitz property if for every $x, y \in V$ we have
 $$D(Mx, My) \leq d(x,y)$$
This condition will ensure that individuals who are similar according to the metric are treated similarly by the classifier. 
Fairness through Awareness [6] is a method that uses this concept to achieve fairness. 
It captures fairness by the principle that any two individuals who are similar with respect to a particular task should be classiﬁed similarly with a distance metric that deﬁnes the similarity between the individuals..

**Group Fairness:** Statistical parity is a good measure of group fairness, meaning the proportion of positive and negative classifications are identical to the demographics of the whole population. 
It is inadequate to use as a notion of fairness as it does not guarantee individual fairness
Lipschitz condition implies statistical parity between two groups only if the Earthmover distance between them is small.

 Key Findings

- We can only apply one non-discrimination criteria to maintain a valid model. Applying more than one criteria will result in degenerate solutions and these criteria conflicts with each other as we claimed above.
- Independence is inadequate to capture fairness, as it does not guarantee that the score is independent of the sensitive attribute given the target variable.
- Observational criteria can not give satisfactory answers to the root cause of discrimination; whether or not discrimination is due to decision maker action or inherent inequality in society is unclear.
- There is concern that higher error rates coincide with historically marginalized and disadvantaged groups, thus inflicting additional harm on these groups.
- The classifier must be able to detect group membership. If detecting group membership is impossible, then group calibration generally fails.
- LFR (Learned Fair Representations) [7] seems to be the closest to achieving a well-defined statistical non-discrimination criteria.

# Critical Analysis

- Solutions to minimize discrimination does not cover intersectionality. there are disadvantages that members of multiple protected categories face. In most cases, if one criterion holds, another is violated. Indeed, this poses questions about being able to come up with a gold standard for fairness, much like differential privacy for privacy, that satisfies all key desirable properties. This remains a hard problem.
- Lipschitzness might not be enough to ensure fairness. When two groups are significantly different, the Lipschitz condition does not imply statistical parity between them. From [6], Lipschitzness implies statistical parity for two distributions $S$ and $T$ if $S$ and $T$ are close enough/exhibit low bias. Bias is the maximum difference in probabilities over all $(D,d)$-Lipschitz mappings $M$ for any individual over distributions $S$ and $T$.
- In reading [5], we think Lipschitzness might not be enough to ensure fairness. When two groups are significantly different, the Lipschitz condition does not imply statistical parity between them. From [6], Lipschitzness implies statistical parity for two distributions $S$ and $T$ if $S$ and $T$ are close enough/exhibit low bias. Bias is the maximum difference in probabilities over all $(D,d)$-Lipschitz mappings $M$ for any individual over distributions $S$ and $T$.  Lipschitzness is an analytic definition of continuity, and while it may make some sense for fairness intuitively, it feels a bit out of place: why use Lipschitz continuity right off-the-shelf? Why not with a multiplier on the Lipschitz upper bound? There is a chance we can come up with a better variant of this Lipschitz condition that is tailored better to parity condition.
- The namings of criteria in [5] are at times unintuitive and confusing. We might need to have more intuitive names for the criteria based on how it was constructed. 



## References
- [5]. Fairness and Machine Learning, Ch 3. S. Barocas, M. Hardt, A. Narayanan, 2023
- [6]. Fairness Through Awareness. C. Dwork, M. Hardt, T. Pitassi, O. Reingold, R. Zemel, 2011
- [7]. Learning Fair Representations. R. Zemel, Y Wu, K. Swersky, T. Pitassi, C Dwork, 2013
- [8]. Equality of Opportunity in Supervised Learning. M. Hardt, E. Price, N. Srebro, 2016