# Auditing and Membership Inference

## Introduction

Before dealing with inference attacks, we need to define what privacy means in the context of machine learning:

 Inference about members of the population:
A privacy breach occurs if an adversary can use the model’s output to infer the values of unintended (sensitive) attributes used as input to the model. However, it may not be possible to prevent this “breach” if the model is based on statistical facts about the population. For example, suppose that training the model has uncovered a high correlation between a person’s externally observable phenotype features and their genetic predisposition to a certain disease. This correlation is now a publicly known scientific fact that allows anyone to infer information about the person’s genome after observing that person.

This means that the creator of a generalizable model cannot do anything to protect “privacy” as defined above because the correlations on which the model is based -and the inferences that these correlations enable- hold for the entire population, regardless of how the training sample was chosen or how the model was created from this sample. 

Inference about members of the training dataset:
To bypass the difficulties inherent in defining and protecting privacy of the entire population, we focus on protecting privacy of the individuals whose data was used to train the model. Our ultimate goal is to measure the membership risk that a person incurs if they allow their data to be used to train a model.

The basic attack in this setting is membership inference, i.e., determining whether a given data record was part of the model’s training dataset or not.

## Motivations

Neural networks are now trained on increasingly sensitive datasets, and so it is necessary to ensure that trained models are privacy-preserving. As an example of breach of privacy in sensitive settings, knowing that a certain patient’s clinical record was used to train a model associated with a disease (e.g, to determine the appropriate medicine dosage or to discover the genetic basis of the disease) can reveal that the patient has this disease.

## Methods

### Shadow Training Technique

This membership inference attack exploits the observation that machine learning models often behave differently on the data that they were trained on versus the data that they “see” for the first time. The attack model is a collection of models, one for each output class of the target model.

To train the attack model, we build multiple “shadow” models intended to behave similarly to the target model. In contrast to the target model, we know the ground truth for each shadow model, i.e., whether a given record was in its training dataset or not. Therefore, we can use supervised training on the inputs and the corresponding outputs (each labeled “in” or “out”) of the shadow models to teach the attack model how to distinguish the shadow models’ outputs on members of their training datasets from their outputs on non-members. Figure 1 illustrates our end-to-end attack process.
<p align="center">
  <img src="https://https://github.com/elleryyu/fall-24-RAI/blob/main/privacy/img_0408/img_0408_1.png" alt="Description of the image">
</p>


The standard metrics for attack accuracy are precision (what fraction of records inferred as members are indeed members of the training dataset) and recall (what fraction of the training dataset’s members are correctly inferred as members by the attacker). 

### The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks (Efficient Extraction Algorithm)
The Efficient Extraction Algorithm introduced by Carlini et al. in "The Secret Sharer" paper is designed to determine if a neural network has unintentionally memorized specific sequences (or "canaries") from its training data. This algorithm is an innovative approach that adapts Dijkstra's shortest-path algorithm to navigate the vast space of possible sequences that a model could generate, focusing on efficiently uncovering those sequences the model predicts with unusually low perplexity—indicative of memorization.

At its core, the algorithm constructs a weighted tree where nodes represent partial sequences generated so far, and edges represent the model's predicted probability (or its negative log) of the next token given the current sequence. Starting from an empty sequence (the root node), the algorithm explores outward, prioritizing paths that lead to lower perplexity sequences—those the model finds less surprising and thus more likely has memorized. 

To efficiently manage this exploration, the algorithm employs a priority queue to keep track of which sequences to expand next, based on their current perplexity. In each step, it expands the most promising sequences (those with the lowest accumulated perplexity) by considering all possible next tokens and adding these extended sequences back into the queue. The search continues until a full-length sequence is found that meets the criteria of being a memorized sequence, indicated by its low perplexity relative to others.

A key optimization utilized by the algorithm involves batch processing: instead of evaluating the model's predictions for one sequence extension at a time, it leverages modern GPUs' ability to perform parallel computations. This allows the algorithm to evaluate many possible next tokens simultaneously, drastically reducing the computational overhead and making the search process more feasible in practice.

This Efficient Extraction Algorithm serves a dual purpose. It not only validates the exposure metric proposed in the paper as a reliable measure of unintended memorization but also provides a practical tool for identifying specific instances of memorization. By efficiently pinpointing sequences that a model has memorized to an extent that they can be extracted through this method, the algorithm highlights potential privacy risks in models trained on sensitive data, offering a pathway towards more secure and privacy-preserving machine learning practices.

<p align="center">
  <img src="https://https://github.com/elleryyu/fall-24-RAI/blob/main/privacy/img_0408/img_0408_2.png" alt="Description of the image">
</p>

### The Likelihood Radio Attack(LiRA):This method is based on the following game:
(Membership inference security game). The game proceeds between a challenger C and an adversary A: 
1) The challenger samples a training dataset D ← D and trains a model fθ ← T (D) on the dataset D. 
2) The challenger flips a bit b, and if b = 0, samples a fresh challenge point from the distribution (x,y) ← D (such that (x,y) / ∈ D). Otherwise, the challenger selects a point from the training set (x,y) ←$ D. 
3) The challenger sends (x,y) to the adversary. 
4) The adversary gets query access to the distribution D, and to the model fθ, and outputs a bit ˆ b ← AD,f(x,y). 5) Output 1 if ˆ b = b, and 0 otherwise.

Based on the above game, a hypothesis test is needed in which we will inference whether or not f was trained on (x,y). The researcher consider two distribution over models here: Qin(x,y)={fT(D{(x,y)})|DD}, and Qin(x,y)={fT(D\{(x,y)})|DD}. Given a model f and a target example (x,y), the adversary’s task is to perform a hypothesis test that predicts if f was sampled either from Qin or if it was sample from Qout. Researchers will conduct a hypothesis test based on the Neyman-Pearson lemma. It involves the use of a likelihood-ratio test to compare two hypotheses. However, the process can be complicated due to the unknown analytical forms of some distributions. To simplify this, the distributions are defined as losses for models that are either trained or untrained on a given example. Following this intuition, researchers train several “shadow models” in order to directly estimate the distribution Qin/out. 

The researcher train model on a random samples of data from the distribution D and obtain the empirical estimates of the distribution Qin and Qout for any example (x,y). To enhance performance at very low false-positive rates, instead of directly modeling the distributions ˜ Qin/outfrom the data, a decision has been made to opt for Gaussian distributions for ˜Qin/out, employing a parametric approach. This choice offers several notable advantages over nonparametric modeling. Parametric modeling necessitates training fewer shadow models to achieve comparable generalization compared to nonparametric methods. Additionally, the approach allows for the extension of attacks to multivariate parametric models, which can further enhance attack success rates by querying the model multiple times. However, caution is warranted in this process. As illustrated in Figure 3, the model's cross-entropy loss doesn't lend itself well to approximation by a normal distribution. Firstly, the cross-entropy loss operates on a logarithmic scale. Taking the negative exponent yields the model's "confidence," which is bounded within the interval [0,1] and therefore not normally distributed either. Specifically, the confidences for outliers and inliers concentrate around 0 and 1 respectively. Hence, a logit scaling is applied to the model's confidence.
The overall training algorithm is display below

<p align="center">
  <img src="https://https://github.com/elleryyu/fall-24-RAI/blob/main/privacy/img_0408/img_0408_3.png" alt="Description of the image">
</p>

<p align="center">
  <img src="https://https://github.com/elleryyu/fall-24-RAI/blob/main/privacy/img_0408/img_0408_4.png" alt="Description of the image">
</p>

The researchers begin by training N shadow models on random samples from the data distribution D. Half of these models are trained on the target point (x,y), while the other half are not. They are respectively referred to as IN and OUT models for (x,y). Two Gaussians are then fitted to the confidences of the IN and OUT models on (x,y) in logit scale. Subsequently, the confidence of the target model f on (x,y) is queried, and a parametric Likelihood-ratio test is conducted to generate the output.

 ### Auditing Differentially Private Machine Learning 
The paper investigates the practical privacy levels of Differentially Private Stochastic Gradient Descent (DP-SGD) through novel data poisoning attacks, assessing if DP-SGD offers better privacy than its theoretical analysis suggests. And we are going to introduce these concepts as follows.
#### Measuring Differential Privacy:
DP-SGD modifies the traditional Stochastic Gradient Descent (SGD) by adding noise to the gradients during training, which protects the privacy of individual data points. It involves two main steps: first, it clips the gradient of each data point in a batch to limit its influence, and then adds random noise to the sum of these clipped gradients. This ensures that the final model cannot be exploited to infer specific details about the training data, providing a balance between utility and privacy.	
Another component they introduced is how to measure lower bound, which is equivalent to estimate p0 and p1 properly. And the estimation method rely on Monte Carlo estimation. What is more, to ensure statistical certainty, they used the Clopper Pearson confidence interval as correction. The Algorithm detail is shown below

<p align="center">
  <img src="https://https://github.com/elleryyu/fall-24-RAI/blob/main/privacy/img_0408/img_0408_7.png" alt="Description of the image">
</p>


#### Poisoning Attacks
In a backdoored model, the performance on natural data is maintained, but, by adding a small perturbation to a data point x into Pert(x), the adversary changes the predicted class of the perturbed data. These attacks have been developed for image datasets. The original backdoor poisoning perform unsatisfied because of clipping, so they use Clipping-Aware Backdoor Poisoning Attack and Test Statistic primal poisoning attack method even though they also have experiments on original poisoning attack.				

<p align="center">
  <img src="https://https://github.com/elleryyu/fall-24-RAI/blob/main/privacy/img_0408/img_0408_8.png" alt="Description of the image">
</p>

		
				

 		

## Key Findings

### Shadow Training Technique

The first takeaway from evaluating the attack model is that the number of output classes of the target model contributes to how much the model leaks. The more classes, the more signals about the internal state of the model are available to the attacker. We can see in the figure below that the results for CIFAR-100 are better than for CIFAR-10. The CIFAR-100 model is also more overfitted to its training dataset. For the same number of training records per class, the attack performs better against CIFAR-100 than against CIFAR-10. Informally, models with more output classes need to remember more about their training data, thus they leak more information.

<p align="center">
  <img src="https://https://github.com/elleryyu/fall-24-RAI/blob/main/privacy/img_0408/img_0408_9.png" alt="Description of the image">
</p>


The figure below illustrates how the target models’ outputs distinguish members of their training datasets from the non-members. This is the information that the attack exploits. The plots show that there is an observable difference between the output of the model on the member inputs versus the non-member inputs in the cases where the attack is successful.

<p align="center">
  <img src="https://https://github.com/elleryyu/fall-24-RAI/blob/main/privacy/img_0408/img_0408_10.png" alt="Description of the image">
</p>


In conclusion, success of membership inference is directly related to the generalizability of the target model and diversity of its training data. If the model overfits and does not generalize well to inputs beyond its training data, or if the training data is not representative, the model leaks information about its training inputs.

It is also observed that overfitting is not the only factor that causes a model to be vulnerable to membership inference. The more overfitted a model, the more it leaks—but only for models of the same type. The structure and type of the model also contribute to the problem. Different machine learning models, due to their different structures, “remember” different amounts of information about their training datasets. This leads to different amounts of information leakage even if the models are overfitted to the same degree.

### The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks
One of the key findings is the introduction and validation of the "exposure" metric, which quantitatively measures the extent of a model's memorization of specific sequences (canaries). The authors show that as the exposure of a canary increases, so does the likelihood that the canary has been memorized by the model, to the point where it can be efficiently extracted using the Efficient Extraction Algorithm they developed. This correlation between exposure and extractability validates the metric's usefulness in assessing privacy risks associated with training machine learning models on sensitive data. Furthermore, the results reveal that unintended memorization is not merely a byproduct of overtraining (training a model for too long) but occurs even without it, indicating a more fundamental challenge within the training process of neural networks. The paper also explores several strategies for mitigating unintended memorization, including regularization techniques like weight decay, dropout, and quantization. However, these methods were found to be largely ineffective at reducing memorization without compromising model performance. The most promising solution identified by the authors is the application of differential privacy techniques during training. By adding carefully calibrated noise to the training process, they were able to significantly reduce the exposure of canaries without severely impacting the utility of the models. This suggests that differential privacy could be a viable approach to training models on sensitive data while minimizing the risk of leaking private information through memorization. In conclusion, "The Secret Sharer" sheds light on the critical issue of unintended memorization in neural networks, offering both a methodology for evaluating this risk and exploring potential mitigations. The findings emphasize the need for more privacy-aware training processes in machine learning, particularly when dealing with sensitive information, and pave the way for future research in the area of privacy-preserving artificial intelligence.

### The Likelihood Radio Attack(LiRA)：

 The researchers utilize both conventional datasets typically employed for membership inference attack assessments, as well as less commonly utilized datasets. In addition to the previously introduced CIFAR-10 dataset, they incorporate three other datasets: CIFAR-100 (another standard image classification task), ImageNet (a challenging image classification task), and WikiText-103 (a natural language processing text dataset). For CIFAR-100, the same procedure as for CIFAR-10 is followed, involving training a wide ResNet to 60% accuracy on half of the dataset (25,000 examples). For ImageNet, a ResNet-50 is trained on 50% of the dataset, which comprises roughly half a million examples. Regarding WikiText-103, the researchers employ the GPT-2 tokenizer to partition the dataset into a million sentences and train a small GPT-2 model on 50% of the dataset for 20 epochs to minimize the cross-entropy loss. It's noted that prior studies have also conducted experiments on two toy datasets, namely Purchase and Texas, which the researchers consider as not meaningful benchmarks for privacy due to their simplicity. 

Online Attack Evaluation: 
Figure 5 presents the primary results of the researchers' online attack when evaluated on the four more complex datasets mentioned above (CIFAR-10, CIFAR-100, ImageNet, and WikiText-103). The attack demonstrates true-positive rates ranging from 0.1% to 10% at a false-positive rate of 0.001%. When comparing the three image datasets, consistent with prior works, the researchers find that the attack's average success rate (i.e., the AUC) is directly correlated with the generalization gap of the trained model. All three models have perfect 100% training accuracy, but the test accuracy of the CIFAR-10 model is 90%, the ImageNet model is 65%, and the CIFAR-100 model is 60%. Yet, at low false positives, the CIFAR-10 models are easier to attack than the ImageNet models, despite their better generalization.

<p align="center">
  <img src="https://https://github.com/elleryyu/fall-24-RAI/blob/main/privacy/img_0408/img_0408_11.png" alt="Description of the image">
</p>


Offline Attack Evaluation: 


### Auditing Differentially Private Machine Learning:

From the experiment, we can conclude that ClipBKD returns epsilon lower bounds that are closer to the theoretical bound than other models.

<p align="center">
  <img src="https://https://github.com/elleryyu/fall-24-RAI/blob/main/privacy/img_0408/img_0408_12.png" alt="Description of the image">
</p>


The privacy level of DP-SGD is highly dependent on the choice of hyperparameters like the noise multiplier and the clipping norm. It was found that when the model is initialized in a fixed manner, it achieves better privacy bounds as compared to random initializations. The empirical privacy loss measured is smaller with fixed initializations, suggesting that initial conditions play a significant role in privacy protection. Despite variations in these parameters, the model's utility remains high, indicating a disparity between theoretical privacy guarantees and empirical privacy measurements. The role of clipping norm was highlighted as more critical than the amount of noise added. This suggests a complex interaction between hyperparameters in determining the privacy-utility trade-off in DP-SGD.




## Critical Analysis

### Shadow Training Technique

Several defenses against membership inference are quantitatively evaluated.

#### Restrict the prediction vector to top k classes: 
When the number of classes is large, many classes may have very small probabilities in the model’s prediction vector. The model will still be useful if it only outputs the probabilities of the most likely k classes. The smaller k is, the less information the model leaks. In the extreme case, the model returns only the label of the most likely class without reporting its probability. The smaller k is, the less information the model leaks. In the extreme case, the model returns only the label of the most likely class without reporting its probability. 

#### Coarsen precision of the prediction vector:
To implement this, we round the classification probabilities in the prediction vector down to d floating point digits. The smaller d is, the less information the model leaks. 

Use regularization. Regularization techniques are used to overcome overfitting in machine learning. We use L2-norm standard regularization that penalizes large parameters. The larger the regularization factor is, the stronger the effect of regularization during the training. 

After evaluating the effectiveness of different mitigation strategies, we observe that overall, the attack is robust against these mitigation strategies. Filtering out low-probability classes from the prediction vector and limiting the vector to the top 1 or 3 most likely classes does not foil the attack. Even restricting the prediction vector to a single label (most likely class), which is the absolute minimum a model must output to remain useful, is not enough to fully prevent membership inference. The attack can still exploit the mislabeling behavior of the target model because members and non-members of the training dataset are mislabeled differently (assigned to different wrong classes). If the prediction vector contains probabilities in addition to the labels, the model leaks even more information that can be used for membership inference. 

Regularization, however, appears to be necessary and useful. It generalizes the model and improves its predictive power and also decreases the model’s information leakage about its training dataset. However, regularization needs to be deployed carefully to avoid damaging the model’s performance on the test datasets.

### The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks
One of the primary shortcomings is the focus on specific types of neural network models and datasets, which may limit the generalizability of the findings across the diverse landscape of machine learning applications. The experiments are conducted primarily on generative sequence models, such as language models, leaving open questions about how these findings apply to other types of models like convolutional neural networks used in image processing or other architectures employed in different domains. Another limitation is the reliance on the "exposure" metric and the Efficient Extraction Algorithm, which, while innovative, might not capture all nuances of memorization or be applicable in all scenarios. For instance, the effectiveness of these tools in detecting memorization in models trained on highly dimensional or non-sequential data remains to be fully understood. Additionally, while the paper suggests differential privacy as a promising solution for mitigating unintended memorization, implementing differential privacy comes with its own set of challenges, including the potential degradation of model utility and the complexity of selecting appropriate privacy parameters. The trade-offs between privacy and utility, particularly in highly sensitive applications, require further investigation to develop practical, privacy-preserving machine learning frameworks that do not compromise on performance. Lastly, the paper opens the door to broader discussions about the ethical implications of training models on large datasets containing potentially sensitive information. It underscores the need for comprehensive strategies that go beyond technical solutions, including legal, regulatory, and policy frameworks to protect individuals' privacy in the age of AI. Further research is necessary to address these limitations and explore the multifaceted aspects of privacy and security in machine learning, ensuring that advancements in AI are both powerful and privacy-preserving.


### The Likelihood Radio Attack(LiRA)：  

One critical analysis stemming from the study lies in the implications of the attack's success rates across different datasets and models. The observed disparity in attack success rates between CIFAR-10 and ImageNet models despite their varying generalization gaps prompts a deeper examination into the factors influencing model vulnerability to membership inference attacks. While prior works have often correlated attack success with poorer generalization, this study suggests a more nuanced relationship, indicating that factors beyond generalization accuracy might play pivotal roles. This raises questions regarding the transferability of privacy vulnerabilities across datasets and the robustness of different model architectures against such attacks. Additionally, the choice of datasets for evaluation warrants scrutiny; while the inclusion of conventional datasets like CIFAR-10 and ImageNet provides valuable insights into attack performance on standard benchmarks, the absence of analysis on more diverse data types or domains limits the generalizability of findings.

### Auditing Differentially Private Machine Learning:

This paper is mainly about empirical work. At the end of the paper, the researchers raise key questions about the quantitative push of attacks, the gap between theoretical and practical privacy boundaries, the integration of additional features such as gradient shear norms, and the realism of attack examples. The challenges of fully determining precise privacy levels through empirical methods are critically noted, but the value of empirical methods in complementing research on privacy analysis in machine learning is emphasized. In addition, the researchers call for more research to understand and bridge the gap between theoretical and empirical privacy measures and to make it easier for laypeople to understand the consequences of privacy breaches. 
Another problem is related to the experiment set up, where the researchers only use LR and FNN. I think it would be more convincing if they could use some cutting edge models trained with DP-SGD. 


