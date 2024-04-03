# Introduction
When applying machine learning algorithms to datasets that contain high volumes of sensitive personal data, privacy is a top priority. Technology breakthroughs that make it possible to gather and analyze enormous volumes of data also increase the risk of privacy violations and improper use of personal data. People have a right to anticipate that their data will be treated carefully and in accordance with their right to privacy. Even though machine learning algorithms are effective tools for deriving insights and forecasts from data, if they are not used appropriately, they may unintentionally reveal private information about specific people. As a result, it is crucial to protect personal data privacy at every stage of the data processing pipeline, from data collection and storage to analysis and result distribution. <br /><br />
Monteleoni et. al. [59] discuss general techniques to create privacy-preserving approximations of classifiers learned through empirical risk minimization (ERM), including regularized ERM. The algorithms proposed in this paper are private under the epsilon-differential privacy definition introduced by Dwork et al. (2006). Theoretical results are provided to show that the proposed algorithms preserve privacy and offer generalization bounds for linear and nonlinear kernels, given certain convexity and differentiability criteria. The results are applied to produce privacy-preserving versions of regularized logistic regression and support vector machines (SVMs). Finally, encouraging results from evaluating the performance of the proposed methods on real demographic and benchmark datasets are presented.

Abadi et al. [60] focus on more general non-convex objectives and introduce the application of differential privacy within the framework of stochastic gradient descent, presenting an algorithm that ensures privacy without significantly compromising the efficiency or quality of the learning model. The introduction of the Moments Accountant  enables the execution of deep learning algorithms within a strictly controlled privacy budget. The practical efficacy of these methods is demonstrated through experiments on benchmark datasets, showcasing their ability to train deep neural networks with modest privacy expenditure. 

[61] presents a semi-supervised learning approach called Private Aggregation of Teacher Ensembles (PATE). It protects the privacy of sensitive training data by training an ensemble of teacher models on disjoint datasets.
These teacher models are then used to train a student model without direct access to the sensitive data, ensuring privacy even against adversaries with access to the student model’s parameters. The approach is validated with strong privacy guarantees and high utility on benchmark datasets like MNIST and SVHN.
Further authors demonstrate its applicability to deep learning methods without making assumptions about the learning algorithm.


<!-- # paper 61: SEMI-SUPERVISED KNOWLEDGE TRANSFER FOR DEEP LEARNING FROM PRIVATE TRAINING DATA -->


## Motivation
Paper [59] talks about the exponential increase in the amount of personal data kept in electronic databases across areas such as financial transactions, medical records, search history on the internet, and social network activities which has sparked serious privacy concerns. Although machine learning presents a significant opportunity to extract useful population-wide insights from these datasets, it also poses a key challenge: the unintentional exposure of people's private information. While the anonymization of personal data may appear to be a simple solution at first, the success of this approach is frequently compromised by the survival of distinct "signatures" in the remaining data fields, which allow persons to be re-identified. Thus, adopting more sophisticated techniques for machine learning to preserve sensitive data is required.

With the advent of deep learning, which requires extensive data for training accurate and robust models, the risk of inadvertently exposing sensitive information increases significantly. Paper [60] addresses the crucial challenge of developing a methodology that allows for the training of deep neural networks while rigorously preserving the privacy of the individuals represented in the training data. By introducing a differentially private approach to stochastic gradient descent, the authors aim to strike a balance between the utility of machine learning models and the imperative need to protect personal privacy. Their work is pioneering in its application of differential privacy to deep learning, providing a framework for privacy-preserving machine learning that does not significantly compromise the performance of the models. 

Paper [61] addresses the challenge of training machine learning models on sensitive data, such as medical records or personal photographs. ML models often inadvertently memorize and expose private information. The authors aim to develop a method that provides strong privacy guarantees for training data to prevent such unintended disclosures. Key contributions of this work are as follows:

- Introduction of Private Aggregation of Teacher Ensembles (PATE) approach that ensures privacy by aggregating the knowledge of multiple models trained on disjoint datasets.
- Demonstrate the applicability of PATE methods in machine learning algorithms and achieve state-of-the-art privacy and utility trade-offs.
- The use of semi-supervised learning to enhance the student model’s performance without compromising privacy.
- Provision of formal privacy guarantees in terms of differential privacy, ensuring that the student model’s training is not influenced by any single sensitive data point.
- Empirical validation of the approach on benchmark datasets like MNIST and SVHN, demonstrating competitive accuracy with meaningful privacy bounds.

## Methodology

### Modeling Empirical Risk Minimization (ERM)
Let ${  D \in \{ \{(x_{i}, y_{i}) \in X \times Y : i = 1, 2, \ldots, n\} \} }$ denote the training data that a machine learning algorithm will be learning from, where ${X = \rm I\!R^{d}}$ and ${ Y = (-1, +1)  }$

We would like to produce a predictor ${f:X\to Y}$ and the measure the quality of this predictor on the training data using a non-negative loss function ${l: Y \times Y\to \rm I\!R}$. <br>
In regulared empirical risk minimization (ERM), we choose a predictor ${f}$ that minimizes the regularized empirical loss defined below:
$${J(f,d) = \frac{1}{n}\sum_{i=1}^{n}l(f(x_{i}),y_{i}) + \Lambda N(f)}$$
There are some assumptions to keep in mind when constructing privacy-preserving ERM algorithms: <br>
1. The loss function ${J(f,d)}$ and the regularizer ${N(.)}$ must be convex. Convex functions are those that satisfy the following property: ${H(\alpha f + (1-\alpha)g) < \alpha H(f) + (1-\alpha)H(g)}$. Strong
convexity plays a role in guaranteeing privacy and generalization requirements
2. Algorithms must satisfy the ${\epsilon_{p}}$ differential privacy model. This model provides a quantifiable framework for measuring the privacy guarantees of algorithms that process sensitive data. It aims to ensure that the presence or absence of any individual's data in a dataset doesn't significantly affect the outcome of the computation, thus preserving the privacy of individuals whose data is included in the dataset.
   
#### Approaches for Privacy Preserving ERM
[59] describes two approaches for achieving privacy preserving ERM.
- **Output Perturbation: The Sensitivity Method**
<br>Inputs: Data ${D}$ with parameters ${\epsilon_{p}, \Lambda}$
<br>Output: Approximate minimizer ${f_{priv}}$
<br>Draw a vector ${b}$ with ${\beta= \frac{n\Lambda \epsilon_{p}}{2}}$
<br>Compute ${f_{priv} = \text{argmin} J(f,D) + b}$, where ${b}$ is random noise with density

- **Objective Perturbation**
<br>Inputs: Data ${D}$ with parameters ${\epsilon_{p}, \Lambda}$
<br>Output: Approximate minimizer ${f_{priv}}$
<br>Similar to the first method, but instead of perturbing the output using sensitivity, we perturb the objective function

<br><p align="center">
  <img src="img/objective_perturb.png" alt="Description of the image">
</p>

<<<<<<< Updated upstream
## Kernal methods
paper[59] also introduces a "kernel trick" that would allow efficient construction of predictor f that lies in a reproducing kernel Hilbert space $\mathcal{h}$ associated with a positive definite kernel function. This theorem demonstrated that regularized ER is minimized by function f(x) which is given by linear combination of kernel functions centers at data points. 

<br><p align="center">
  <img src="img/Kernal1.png" alt="Description of the image">
</p>

Although this result is important for computational and theoretical reasons, there are privacy concerns as the classifier requires revealing the data. the paper suggests a new algorithm to mitigate this challenge

<br><p align="center">
  <img src="img/Kernel2.png" alt="Description of the image">
</p>

This algorithm is designed to ensure privacy-preserving empirical risk minimization. 
=======
- **Parameter Tuning**
The paper introduces a privacy-preserving parameter tuning method for machine learning algorithms, extending privacy guarantees to the tuning phase beyond just model training. Utilizing differential privacy principles and a randomized selection process based on disjoint data subsets, the technique ensures that privacy is maintained for all data used in learning, including parameter selection. This approach addresses the critical balance between privacy protection and model performance, acknowledging the inherent trade-offs in privacy-enhancing technologies.

<br><p align="center">
  <img src="img/privacy_59_tuning.png" alt="parameter tuning">
</p>

>>>>>>> Stashed changes
### Deep Learning with Differential Privacy

For the differentially private training of neural networks, Abadi et al. [60] introduced two different components in their paper, including a *differentially private stochastic gradient descent (SGD) algorithm* and a *moments accountant*.

#### Differentially Private SGD Algorithm

The algorithm below outlines their basic method for training a model with with parameters θ by minimizing the empirical loss function $L(θ)$. At each step of the SGD, it computes the gradient $∇_θL(θ, x_i)$ for a random subset of examples, clip the 2-norm of each gradient, compute the average, add noise in order to protect privacy, and take a step in the opposite direction of this average noisy gradient. At the end, in addition to outputting the model, it will also compute the privacy loss of the mechanism based on the information maintained by the privacy accountant.

<p align="center">
  <img src="img/privacy_60_algorithm.png" alt="Description of the image" width="50%">
</p>

At the core of the differentially private SGD algorithm is the mechanism for clipping and adding noise to the gradients. Each gradient computed from a batch of data points is clipped to have a bounded L2 norm, ensuring that no individual data point can have an outsized influence on the gradient. This step is crucial for limiting the sensitivity of the optimization process to the presence or absence of any single record in the dataset. Subsequently, Gaussian noise is added to the averaged, clipped gradients, which introduces the necessary randomness to achieve differential privacy. The magnitude of this noise is a function of the predefined privacy budget, with the parameters ε (epsilon) and δ (delta) dictating the trade-off between privacy and the accuracy of the model.

#### The Moments Accountant

To accurately manage and track the privacy expenditure associated with the repeated application of noisy updates, the authors introduce a analytical tool known as the "moments accountant." This tool provides a sophisticated method for quantifying the cumulative privacy loss over multiple iterations of the training process. Unlike traditional approaches to differential privacy that might overly constrain model performance or provide loose privacy guarantees, the Moments Accountant allows for a more nuanced and tight control of privacy loss. It enables the algorithm to make full use of its privacy budget, optimizing the training process within strict privacy constraints.

The moments accountant involves evaluating the impact of the noise added to the gradients to mask individual contributions. The Moments Accountant calculates the cumulative privacy cost using a detailed analysis of the noise distribution, encapsulated in the formula:

$$α(λ) = \max_{aux,d,d'} (\log E[exp(λ c(o; \mathcal{M}, aux, d, d'))] )$$

Here, $α(λ)$ represents the moment generating function of the privacy loss random variable, $λ$ is the order of the moment, $\mathcal{M}$ denotes the mechanism (e.g., the addition of noise in SGD), $aux$ represents auxiliary input, and $d, d'$ are adjacent datasets. The privacy loss $c$ at output $o$ is calculated as the logarithm of the ratio of the probabilities of $o$ given $d$ versus $d'$:

$$c(o; M, aux, d, d') = \log \frac{\Pr[M(aux, d) = o]}{\Pr[M(aux, d') = o]}$$

By integrating over all possible outputs and taking the maximum across all auxiliary inputs and pairs of adjacent datasets, the Moments Accountant provides a tight bound on the total privacy loss. This allows the algorithm to make full use of its privacy budget, optimizing the model's utility while strictly controlling privacy risks.

Furthermore, the cumulative privacy budget is carefully managed to ensure it does not exceed the predefined thresholds, ε (epsilon) for privacy loss and δ (delta) for the probability of exceeding this loss. This management is crucial for balancing the trade-off between model accuracy and privacy protection, enabling the training of complex deep learning models on sensitive datasets without compromising individual privacy.

### Private Aggregation of Teacher Ensembles (PATE)

PATE is an innovative approach that ensures privacy during machine learning model training by aggregating knowledge from multiple teacher models trained on disjoint datasets. It ultimately enables the creation of a student model without direct access to sensitive data. The model has five major components: $(i)$ sensitive data, $(ii)$ teacher models, $(iii)$ student model, $(iv)$ aggregate teacher, and $(v)$ privacy protection. A short description of these components is given below:

+ __Sensitive Data__: This represents the private training data that is divided into multiple disjoint datasets.
+ __Teacher Models__: Each dataset is used to train a separate “teacher” model. These models are knowledgeable about their respective datasets but do not share information with each other to maintain data privacy.
+ __Student Model__: A “student” model is trained using public, non-sensitive data that is labeled based on the aggregated output of the teacher models. The student model learns to mimic the ensemble of teachers.
+ __Aggregate Teacher__: The predictions from all teacher models are aggregated to form a consensus on the output for new data points.
+ __Privacy Protection__: The approach ensures privacy by limiting the student’s exposure to the teachers’ knowledge, which is quantified and bounded.

First, the sensitive data are partitioned into $n$ disjoint datasets. Each dataset is used to train a teacher model and $n$ teacher model is aggregated into a new teacher model. The aggregate teacher model provides the predicted labels. This label is open to use but the aggregate model and data are not accessible by the adversary. On the other hand, a student model is trained on public data using the labels provided by the teacher model. This process ensures that the student model learns to accurately mimic the ensemble without having direct access to the sensitive training data. The student’s training is influenced by noisy voting among the teachers, which is a mechanism to protect the privacy of the training data. The student model can then be deployed for predictions without compromising the privacy of the sensitive data it was trained on. An overview of this approach is shown in the figure below.

<p align="center">
  <img src="img/overview_pate.png" alt="Description of the image">
</p>

***Private Learning with ensembles of teachers***


We can divide this into the following 
+ __Training the ensemble of teachers__: Assume (X,Y) is the dataset, where X denotes inputs and Y is the outputs. Let m be the number of the classes in the task. The dataset is partitioned into n disjoint datasets $(X_i, Y_i)$. Further, n classifiers $f_i$ called teacher is trained using $(X_i,Y_i)$. For an input $\tilde{x}$, each teacher provides a prediction $f_i(\tilde{x})$. The label counts $n_j(\tilde{x})$ for a given class $j \in m$ is the number of teachers that assigned class $j$ to input $\tilde{x}$. To ensure privacy, random noise is added to the vote counts using the Laplace mechanism. This process ensures that the student model’s training does not depend on the details of any single sensitive training data point.
+ 
  <p align="center">
  $n_j​(\tilde{x})=∣{i:i\in [n],f_i​(\tilde{x})=j}∣$
</p>
<p align="center">
    $f(\tilde{x})=\text{arg max}_j​(n_j​(\tilde{x})+\text{Lap}(\frac{1}{\gamma}​))$ ,
</p>
<p align="center">
  where γ is a privacy parameter and Lap(b) is the Laplacian distribution with location 0 and scale b.

</p>
     
+ __Semi-supervised transfer of the knowledge from an ensemble to a student__: The student model is trained using non-sensitive and unlabeled data. Some of this data is labeled using the aggregation mechanism from the teacher ensemble. The student model replaces the teacher ensemble for deployment. Here privacy loss is fixed and does not increase with user queries to the student model. Even if the student model’s architecture and parameters are public or reverse-engineered, the privacy of original training dataset contributors is maintained. Furthermore, the privacy cost is determined by queries made during teacher ensemble training. In addition, various techniques were considered (distillation, active learning, etc.), but the most successful one is semi-supervised learning with GANs (PATE-G).

***Privacy analysis of the approach***

__Privacy Loss__: Assume a randomized mechanism $M$ with domain $D$ and range $R$ satisfies ($\epsilon, \delta$)-differntial privacy if any two adjacent inputs d, $d' \in D$ and for any subset of outputs $S \subseteq R$ it holds that:
<p align="center">$Pr[M(d)\in S]\le e^\epsilon Pr[M(d')\in S]+\delta$</p>

Now let, $M:D \to R$ be a randomized mechanism and $d, d'$ a pair of adjacent databases. Let aux denote an auxiliary input. For an outcome $o \in R$, the privacy loss at o is defined as:
<p align="center"> $c(o; M, aux, d, d') = log \frac{Pr[M(aux, d)=o]}{Pr[M(aux, d')=o}]$ </p>

The privacy loss random variable $C(M, aux, d, d')$ is defined as $C(M(d);M,aux, d,d')$ i.e., random variable defined by evaluating the privacy loss at an outcome sampled from $M(d)$.

__Moments accountant__: $\alpha_M(\lambda) = max_{aux, d,d'}$ $\alpha_M (\lambda; aux, d , d')$, where $\alpha_M (\lambda; aux, d , d') = log~E[exp(\lambda C(M, aux, d, d'))]$ is the moment generating function of privacy loss random variable.

Based on this definition, authors provide three theorm as follows:

<p align="center">
  <img src="img/privacy_61_theorem-1.png" alt="Description of the image">
</p>
<p align="center">
  <img src="img/privacy_61_theorem-2.png" alt="Description of the image">
</p>

From this theorem, we have following properties:

+ Privacy Bounds and Moments:
  + Theorems 2 and 3 provide bounds for specific moments related to privacy.
  + The smaller of these bounds is used, and Theorem 1 allows combining them to derive an (ε, δ) guarantee.
  + Privacy moments are data-dependent, so the final ε value should not be revealed.
  + To estimate privacy cost differentially, smooth sensitivity is bounded, and noise is added to moments.
+ Effect of Number of Teachers on Privacy Cost:
  + Student uses a noisy label (parameter γ) computed in (1).
  + Smaller γ leads to lower privacy cost.
  + A larger gap between the two largest values of nj results in a smaller privacy cost.
  + More teachers can lower privacy cost, but there’s a limit: With n teachers, each trains on a 1/n fraction of data, potentially reducing accuracy.

## Evaluation and Result analysis 

### Evaluation of Differentially Private Empirical Risk Minimization

#### Privacy-Accuracy Tradeoff
The results shown in Figures 2 and 3, along with Tables 1 and 2 of the paper, collectively illustrate the privacy-accuracy trade-offs and the performance of different regularization parameters across two datasets: the Adult data set and the KDDCup99 data set.

**Figure 2** demonstrates the privacy-accuracy trade-off for the Adult data set. It highlights the performance comparison between sensitivity-based and objective perturbation methods for logistic regression and Support Vector Machines (SVMs), showing how error rates change with varying levels of privacy parameter (εp). The graph indicates that as privacy increases (εp decreases), misclassification error rates generally increase for both logistic regression and SVMs, reflecting the trade-off between privacy and accuracy.
<p align="center">
  <img src="img/privacy_59_fig2.png" alt="Description of the image">
</p>

**Figure 3** explores similar dynamics for the KDDCup99 data set, detailing how varying the privacy parameter impacts the misclassification error for both logistic regression and SVMs. This comparison underscores the consistent trend where enhanced privacy (lower εp) correlates with higher error rates, underscoring the inherent balance between preserving data privacy and maintaining model accuracy.
<p align="center">
  <img src="img/privacy_59_fig3.png" alt="Description of the image">
</p>

**Table 1** presents the error rates for different regularization parameters on the Adult dataset with a fixed privacy parameter (εp = 0.1), showcasing the best error per algorithm in bold. This table allows for an understanding of how varying regularization strengths impact the model's error rate, with the optimal values for minimizing the error highlighted for both logistic regression and SVMs under privacy constraints.
<p align="center">
  <img src="img/privacy_59_tab1.png" alt="Description of the image">
</p>

**Table 2**, which was expected to show similar data for another dataset or parameter configuration based on the query, is not directly shown in the provided excerpts. However, it would presumably follow the structure of Table 1, presenting error rates across different configurations to further dissect the relationship between regularization, privacy, and model performance.
<p align="center">
  <img src="img/privacy_59_tab2.png" alt="Description of the image">
</p>
Together, these results paint a comprehensive picture of the trade-offs involved in privacy-preserving machine learning, demonstrating how adjustments to privacy parameters and regularization strength influence model accuracy across different datasets.

#### Accuracy vs. Training Data Size Tradeoffs

Accuracy versus training data size tradeoffs, particularly through Figures 4 and 5, reveals the impact of increasing training set size on classification accuracy for the KDDCup99 data set. These figures, alongside experimental data, illustrate how classification error trends vary between non-private Empirical Risk Minimization (ERM), objective perturbation, and the sensitivity method under different privacy budgets (εp = 0.01 and εp = 0.05).

**Figure 4** outlines the learning curves for logistic regression. It shows that for both privacy levels (εp = 0.01 and εp = 0.05), the classification error decreases as the size of the training set increases. This trend is consistent across the objective perturbation and sensitivity methods, with the non-private ERM method serving as a baseline. Notably, objective perturbation consistently outperforms the sensitivity method, suggesting that it's more effective at maintaining classification accuracy under privacy constraints.

<p align="center">
  <img src="img/privacy_59_fig4.png" alt="Description of the image">
</p>

**Figure 5** provides a parallel analysis for Support Vector Machines (SVM). Similar to the logistic regression results, the error rates decrease with larger training sets. Again, objective perturbation demonstrates superior performance to the sensitivity method, aligning with the trends observed in logistic regression.
<p align="center">
  <img src="img/privacy_59_fig5.png" alt="Description of the image">
</p>
Overall, the results underscore the effectiveness of objective perturbation over the sensitivity method across both machine learning models, highlighting the potential of privacy-preserving techniques to achieve lower classification errors with increasing training data sizes, within the bounds of specified privacy budgets (εp values).

### Evaluation of Differentially Private Deep Learning

For the MNIST dataset, known for its collection of handwritten digits, the authors apply their differentially private SGD algorithm to train a deep neural network. The evaluation focuses on the algorithm's ability to maintain high accuracy while adhering to stringent privacy guarantees. By adjusting the noise scale and the privacy budget, the paper illustrates how the algorithm achieves commendable accuracy levels, thereby indicating the practicality of deploying differential privacy in environments where data sensitivity is a concern. This part of the evaluation underscores the nuanced balance between privacy protection and model accuracy, highlighting the effectiveness of the proposed privacy accountant in managing the cumulative privacy budget.

<p align="center">
  <img src="img/privacy_60_mnist.png" alt="Description of the image">
</p>

Moving to the more challenging CIFAR-10 dataset, which comprises color images across ten different classes, the evaluation showcases the algorithm's scalability and adaptability to more complex data representations. The CIFAR-10 results further emphasize the algorithm's robustness, where despite the increased data complexity and the intrinsic challenges of maintaining privacy in datasets with richer information content, the differentially private SGD algorithm still manages to produce models with respectable accuracy. This is particularly noteworthy, considering the dataset's diversity and the higher dimensionality of the data points.

<p align="center">
  <img src="img/privacy_60_cifar.png" alt="Description of the image">
</p>

The evaluation also delves into the impact of various hyperparameters, such as the noise scale, the clipping threshold, and the lot size, on the model's performance. Through a detailed analysis, the paper demonstrates how careful tuning of these parameters can significantly enhance the trade-off between privacy and accuracy. It's shown that by optimizing these hyperparameters, the algorithm can be fine-tuned to achieve better performance, thereby providing valuable insights into the practical considerations required when implementing differential privacy in deep learning models.

<p align="center">
  <img src="img/privacy_60_ablation.png" alt="Description of the image">
</p>

Through experimentation on MNIST and CIFAR-10 datasets, Abadi et al. [60] not only validates the proposed algorithm's effectiveness in preserving privacy but also its ability to retain high levels of model accuracy. These findings are instrumental in advancing the field of privacy-preserving machine learning, offering a promising avenue for researchers and practitioners aiming to leverage the power of deep learning in sensitive data environments.

### Evaluation of Private Aggregation of Teacher Ensembles (PATE)
For the evaluation of the PATE and its generative variant PATE-G, the authors first train a teacher ensemble for each dataset. The trade-off between label accuracy and privacy depends on the number of teachers in the ensemble. Further, they minimize the privacy budget spent on training the student by using as few queries to the ensemble as possible. The experiments focus on MNIST and SVHN datasets, comparing the private student’s accuracy with that of a non-private model trained on the entire dataset under different privacy guarantees.

***TRAINING AN ENSEMBLE OF TEACHERS PRODUCING PRIVATE LABELS***

The study evaluates how well the MNIST and SVHN datasets can be partitioned without significantly impacting individual teacher performance. Despite injecting substantial random noise for privacy, ensembles of 250 teachers achieve accurate aggregated predictions: 93.18% accuracy for MNIST and 87.79% for SVHN, with a low privacy budget of ε = 0.05 per query. Key findings of this experiments are as follows:
+ __Prediction Accuracy__: For MNIST and SVHN datasets, even with n = 250 teachers, individual teacher test accuracy remains reasonably high. For MNIST, average test accuracy is 83.86% and for SVHN, average test accuracy is 83.18%. The larger size of SVHN compensates for its increased task complexity. These findings highlight the delicate balance between privacy, ensemble size, and teacher accuracy in the context of privacy-preserving machine learning.
+ __Prediction confidence__: Privacy of predictions by the teacher ensemble requires agreement among a quorum of teachers. Data-dependent privacy analysis provides stricter bounds when the quorum is strong. Authors find the gap between the most popular and second most popular label. Even as the ensemble size (n) increases, the gap remains larger than 60% of the teachers, allowing accurate label output despite noise during aggregation (Follow figure 3).
<p align="center">
  <img src="img/results_ensemble_teacher.png" alt="Description of the image">
</p>

+ __Noisy Aggregation__: Authors consider three ensembles with varying numbers of teachers: $n \in$ {10, 100, 250}. Larger ensemble sizes are necessary to mitigate the impact of noise injection on accuracy. The accuracy of test set labels inferred by the noisy aggregation mechanism is reported for different values of ε.
Notably, the number of teachers must be substantial to maintain accuracy despite noise injection. (Follow Figure 2)

***Semi-Supervised Training of the Student with Privacy***

To reduce the privacy budget spent on student training, authors are interested in making as few label queries to the teachers as possible. We therefore use the semi-supervised training approach. For MNIST dataset, the student uses 9,000 samples, with subsets of 100, 500, or 1,000 labeled using noisy aggregation and for SVHN, the student has 10,000 training inputs, labeling 500 or 1,000 samples. Student performance is evaluated on remaining test samples.  Leveraging semi-supervised training with GANs, the MNIST and SVHN students achieve impressive accuracies.
Specifically, the MNIST student achieves 98.00% accuracy with strict differential privacy bound of ε = 2.04 (at $10^-5$ failure probability), and the SVHN student achieves 90.66% accuracy with privacy bound of ε = 8.19 (likely due to more queries to the aggregation mechanism). These results surpass the differential privacy state-of-the-art for these datasets. (Follow Figure 4)
<p align="center">
  <img src="img/privacy_61_table.png" alt="Description of the image">
</p>

## Discussion and Conclusion
The conclusion of the paper by Chaudhuri et al [59] reflects on the significant stride in privacy-preserving machine learning through the introduction of two algorithms: sensitivity-based output perturbation and the novel objective perturbation method, within the εp-differential privacy framework. Objective perturbation, in particular, is highlighted for its effectiveness in balancing privacy and performance, showing near-equivalent accuracy to non-private algorithms under certain conditions. This work not only provides the first computationally efficient, differentially private classifiers validated on standard datasets but also sets a precedent in the empirical validation of differential privacy's impact on learning. The paper outlines the potential for future research, including extending these methods to broader convex optimization problems and refining privacy-accuracy trade-offs, underscoring the foundational contribution of this research to the field of privacy-preserving data analysis.

The work by Abadi et al. [60] marks a crucial advancement in blending differential privacy with deep learning, specifically through the use of a differentially private SGD algorithm. Their rigorous evaluation on benchmarks like MNIST and CIFAR-10 demonstrates the practicality of applying differential privacy without significantly sacrificing model performance. The introduction of the Moments Accountant for precise privacy loss tracking is a significant contribution, offering a pathway to more effective privacy budget management.

The paper by Papernot et al. [61] introduces the PATE approach, which aggregates knowledge from “teacher” models trained on separate data.
The “student” model, whose attributes may be public, benefits from this transfer.
Demonstrably achieves excellent utility on MNIST and SVHN tasks while providing a formal bound on user's privacy loss.
The approach requires disjoint training data for a large number of teachers.
Encouragingly, combining semi-supervised learning with precise, data-dependent privacy analysis shows promise.
Future work could explore whether this approach reduces teacher queries for tasks beyond MNIST and SVHN.

## References
[59] Differentially Private Empirical Risk Minimization. Chaudhuri et al 2011.

[60] Deep Learning with Differential Privacy. Abadi et al, 2016

[61] Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data. Papernot et al, 2016

