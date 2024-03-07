
# Introduction and Motivations

Machine learning is often done with the naive assumption that the data used to train a model is well-behaved. However, a class of attacks known as poisoning attacks have been developed in the last decade or so that involve adding carefully maliciously perturbed points to the training set to make models misbehave. Even the addition of one or a few points can impact the behavior of a model, leading it to exhibit poor performance or misclassify certain points to the adversary's benefit.

Consider the case when a company selling antivirus software relies on the use of ML to train malware classifiers. The data used to train such classifiers is often publicly sourced and given the large amounts of data that training such classifiers may require, it may not be humanly possible to audit each and every data point in the training set. Adversaries can then sneak in carefully perturbed points, given some knowledge about the model's architecture, training data distribution, parameters, etc. (white-box setting) or even just query access to it and some limited information about the features of the training data and the learning algorithm (black-box setting), and make the trained malware classifier classify some malware as benign software. Issues related to data poisoning have been investigated in the contexts of self-driving cars, medical AI, etc. as well.

Adversarial attacks manipulate machine learning models by exploiting vulnerabilities in their decision-making processes. They can be split into two categories: causative adversarial attacks and exploratory adversarial attacks. Poisoning attacks fall under the causative adversarial attack category because it involves either injecting new malicious points or manipulating existing ones in the training data to degrade the model's performance generally or for specific targets. Subsequently, poisoning attacks can be categorized into either poisoning availability attacks or poisoning integrity attacks. Poisoning availability attacks aim to disrupt the availability of the model's prediction results indiscriminately, often leading to denial-of-service (DoS) scenarios. Conversely, poisoning integrity attacks target specific mispredictions at test time, therefore undermining the integrity of the model's output for malicious purposes while also in most cases keeping their attack undected.
    
In the context of online training, where machine learning models are continuously updated based on incoming data, poisoning attacks can be particularly potent. Depending on the training system, control over a small number of devices that submit fake information adversarially-poison the dataset.

For ML and ML security/safety practitioners, it thus becomes key to ensure that their models are *adversarially robust*. To this end, studying possible vulnerabilities (here to data poisoning) and attacks is an important area of security research. ML security research over the last few years has resulted in attacks that utilize human-indistinguishable adversarial points. This document is meant to introduce readers to the rationale behind poisoning attacks, explain their workings, and how they might be prevented.


# Methods
In this section, we discuss the methodology of some poisoning attacks and some defenses against them, in that order. For the attacks, we describe the respective threat models and the procedure involved in mounting them. Results for each attack and defense are deferred to the next section (Key Findings).
## Attacks

### Attack 1. Poisoning Attack against SVMs
Support Vector Machines, or SVMs, are prime targets for poisoning attacks. In certain cases, even moving one point (a support vector) can entirely alter the behavior of the SVM. In this attack, an adversary can "construct malicious data" based on predictions of how an SVM's decision function will change in reaction to malicious input.
#### Threat Model
This is a white-box attack where it is assumed that the adversary has knowledge about the training data, the validation data, and the algorithm used to train the model (the incremental SVM training algorithm).
#### Procedure
1. Given the training data $`\mathcal{D}_\text{tr}`$ and the validation data $`\mathcal{D}_\text{val}`$, first train an SVM using $`\mathcal{D}_\text{tr}`$.
2. Then pick any member of the attacked class (preferably sufficiently deep within its margin) and flip its label. Denote this point and its flipped label as $`x_c^(0), y_c`$. This is the initial attack point.
3. Using the previously trained SVM, update it on $`\mathcal{D}_\text{tr}\cup x_c^{(k)}`$ for timestep $k$ using the incremental SVM training method.
4. Perform gradient ascent using the the hinge loss incurred on $`\mathcal{D}_\text{val}`$ by this updated SVM and finding the direction in which it is maximized. Let $`u`$ be the corresponding unit directional vector.
5. Update $`x_c^{(k+1)}\gets x_c^{(k)} + tu`$ where $`t`$ is a small step size.
6. Iterate until termination (when the increase in loss is below a particular threshold).

This produces an attack point that maximizes the (validation) loss for an SVM and can thus force it to misclassify. As we shall observe in the following section, this attack performs quite effectively. In addition, this work provides some salient points that lend to the attack's potency against SVMs.

- The "gradient ascent" attack strategy against SVMs can be kernelized, allowing for attacks to be constructed in the input space.
    - The attack method can be kernelized because it only depends on "the gradients of the dot products in the input space".
    - This is a big advantage for the attacker when compared to a method which can't be kernelized; an attacker using a non-kernelizable method "has no practical means to access the feature space".
 
In other words, the authors incorporate arguments on how to account for kernelization (for the linear, polynomial, and RBF kernels) and include it within the attack's considerations. 

The authors note that their findings emphasize that "resistance against adversarial data" should be considered in the design of learning algorithms.

### Poison Attack on Linear Regression Models:
#### Optimization-Based Attack:
This attack optimizes the variables selected to make the largest impact on that specific model and dataset by optimizing the difference in predictions from the original unpoisoned model as opposed to the loss.

(Comment: mention combinations of attack loss, regression type, optimization dataset).

#### Statistical-Based Attack (StatP):
This attack is a fast statistical attack that produces poisoned points similar to that of the training data using a multivariate normal distribution. Then, knowing that the most effective poisoning points are near corners, it rounds the feature values of the generated points towards the corners. 

### Clean-label Poison Attack on Neural Networks:
A clean-label attack that does not require control over the labeling function. This attack produced poisoned points that appear to be correctly labeled by the expert observer but are able to control the behavior of the classifier.


## Defenses
### TRIM

# Key Findings

[Manipulating Machine Learning](https://arxiv.org/abs/1804.00308) proposes optimized poisoning attacks on the linear regression models and a novel way to defend against them.
- Statistical-based attacks that are derived from insights gained from optimized research conclusions can act as very low maintenance way to manipulate the training data effectively
- Neat mathematical tricks can be applied to these machine learning problems such as replacing the inner learning problem of the bilevel optimization problem with KKT equilibrium conditions allows us to efficiently approximate our desired learning
- TRIM utilizes a trimmed loss function to iteratively estimate regression parameters, effectively isolating poisoned points and performing better than other robust regression algorithms designed for adversarial settings 


The [Certified Defenses for Data Poisoning Attacks](https://arxiv.org/abs/1706.03691) paper discusses how certain defenses against data poisoning can be tested against "the entire space of attacks".
- Specifically, the defense must be one that removes "outliers residing outside a feasible set," and minimizes "a margin-based loss on the remaining data"
- The authors' approach can find the upper bound of efficacy of any data poisoning attack on such a classifier
- The authors analyze the case where a model is poisoned and where both the model and its poisoning detector are given poisoned data

The [Poison frogs!](https://arxiv.org/abs/1804.00792) paper discuss poisoning on neural networks that manipulate the behavior of machine learning models at test time without degrading overall performance

- Clean-labels are poisoned points that don’t require the attacker  to have any control over the labeling of training data. Only having knowledge of the model and its parameters allows us to create points that are undetected as poison and yet still able to influence the classifier.
- There is an optimized method in making a single poisoned image influence the model especially for transfer learning scenarios. The addition of a low-opacity watermark of the target instance increases effectiveness for multiple poisoned instances
- Transfer Learning reacts to poison points by rotating the decision boundary to encompass the target, while end-to-end training reacts by pulling the target into the feature space with the decision remaining stationary.


# Critical Analysis
The [Poisoning Attacks against SVMs](https://arxiv.org/abs/1206.6389) paper describes a kernelizable method of attacks against SVM models. The authors note that this work "breaks new ground" in the research of data-driven attacks against such models because their attack strategy is kernelizable. Their findings and conclusions are intuitive and easy to follow from the reader's point of view. Their attack strategy description is accompanied by a clear example: an attack on a digit classifier. They clearly show how the attacker is able to use their strategy to modify training input data to produce classification error. These attack modifications are understandable to the user as well: they clearly show how the attacker modifies the data so that the "7 straightens to resemble a 1, the lower segment of the 9 becomes more round thus mimicking an 8," and so on. Overall, the authors of this paper adequately communicate the technical aspects of their attack strategy in a way that even non-technical readers can easily understand, and the conclusions that follow are well-supported by the evidence they gathered in their attack.

However, the results in this paper are rudimentary as compared to those in recent ML papers. They provide results on synthetic data and subsets of MNIST (by taking two digits and training a binary classifier SVM to distinguish between them). While they illustrated a novel concept well and in depth back then, a more thorough investigation may have been desirable. In addition, the attack on SVM is entirely white box, and involves strong assumptions about the knowledge the adversary has that may be too strong in practice: in real life, adversaries may not have access to information such as the training/validation data.

In [27], while the work on producing (then) state-of-the-art attacks is impressive, the rationale for basing the success of their results for OptP was not convincing and might lead to a slight inflation of claimed success. In particular, an adversary would have to mount multiple attacks (combinatorial in the number of choices of loss, label initialization strategy, argument, and attack objective/loss) and then report the best. This is not fairly comparable against baselines that only offer one candidate (apparently). However, discussion on the feasibility of each method (attack or defense) as compared to their baselines was very interesting and provided a practical insight into these methods' feasibility in practice. Also a more tangible quantification of the damage the proposed poisoning attack can do in terms of, say, prescribing medicinal dosage helped underline the fact that these attacks can lead to serious real-life consequences.

# References
[26]. Poisoning attacks against support vector machines. Biggio et al. 2012

[27]. Manipulating machine learning: Poisoning attacks and countermeasures for regression learning. Jagielski et al. 2018

[28]. Certified defenses for data poisoning attacks. Steinhardt et al. 2017

[29]. Poison frogs! Targeted clean-label poisoning attacks on neural networks. Shafahi et al. 2018

    
