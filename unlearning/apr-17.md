
# Machine Unlearning

Machine unlearning is a concept that mirrors the human ability to forget, granting Artificial Intelligence (AI) systems the power to discard specific information. It is the converse of machine learning that allows the models to unlearn or forget certain aspects of their training data. It holds promise not only for complying with regulations but also for rectifying factually incorrect information within a model.
Authors in paper [2] addresses the difficulty of removing a user’s data from machine learning models once it has been included. It introduces a framework called SISA training to expedite the unlearning process by strategically limiting the influence of a data point during training. The paper evaluates SISA training across various datasets and demonstrates its effectiveness in reducing computational overhead associated with unlearning, while maintaining practical data governance standards.

## Motivation

These days deep learning models are trained on a large dataset. It’s difficult for users to revoke access and request deletion of their data once shared online. ML models potentially memorizes the data that raises privacy concerns. Consider another example where an employee leaves the company and the employee wants the company removed all his information from the trained model. 
However, the challenge is we can not train the model from scrath as it is time consuming and cost inefficient. Moreover, some of the used data may be changed later. For instance, consider the example "Man City is the current winner of uefa champion league." However, this information will change every year. So, model should update this information by forgetting the previous information. Therefore, unlearning data from ML models is notoriously challenging, yet crucial for privacy protection. In this context, the paper introduces SISA (Sharded, Isolated, Sliced, and Aggregated) training, a framework designed to expedite the unlearning process and reduce computational overhead.


## Current Landscape and Data Protection Laws
[1] talks about the current landscape of the machine learning and data privacy domains while discussing some of the laws in the GDPR, and the implications of common model attacks. 
### The GDPR
The General Data Protection Regulation (GDPR) is a comprehensive data protection law enacted by the European Union (EU) in May 2018. It aimed to harmonize data privacy laws across Europe, enhance the protection of individuals' personal data, and reshape the way organizations approach data privacy. <br />
The GDPR primarily focuses on the processing of personal data. This means that if a machine learning model is trained on personal data, individuals have certain rights regarding the use and processing of their data. However, once data has been anonymized or aggregated to the point where individuals cannot be identified, it may fall outside the scope of these data protection laws. This creates a challenge because while individuals may have control over their personal data, they may not have control over the insights or models derived from it. <br/>
However, the governance of machine learning models is often seen through the lens of intellectual property rights, such as trade secrets. Companies invest significant resources into developing and training machine learning models, and they often seek to protect their models as valuable assets. This means that while individuals may have rights over their personal data, companies may retain control over the models themselves as proprietary information.

### Existing Laws in the GDPR
The term data subjects refers to individuals to which said data relates to. Data controllers refers to organizations or entities who control the purposes and means of processing the data. <br />
The GDPR is a lengthy collection of complex laws that applies whenever personal data is accessed (either collected, transformed, consulted or erased). Some of these are: 
- Models cannot be trained from personal data without a specific lawful ground such as consent, contract or legitimate interest
- Data subjects should be informed of the intention to train a model and
- Usually maintain a right to object or withdraw consent; and 
- In situations where models inform a significant, solely automated decision, individuals can appeal to the data controller for meaningful information about the logic of processing, or to have the decision taken manually reconsidered.
<br/>
There are no specific data protection rights or obligations concerning machine learning models in the period after they have been built but before any decisions have been made about their use. This means that once a machine learning model has been developed, it exists in a regulatory grey area where it is not explicitly covered by data protection laws.
<br/>
Furthermore, in today's landscape, companies are cautious about trading raw data due to strict data protection laws. Instead, they are shifting their focus towards trading or renting out trained machine learning models, which allows them to share value with fewer privacy and regulatory concerns. Many prominent firms are already offering trained models for various tasks such as facial recognition, emotion classification, nudity detection, and identifying offensive text.

### Types of Attacks
<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/ml_attacks.png" alt="Description of the image"/>
</p>
1. <b>Model Inversion:</b>  These attacks essentially turn the training-to-model process a two-way process, which basically allows an unauthorized entity to predict the training data used for building the model. 
In a model inversion attack scenario, a data controller lacking direct access to a dataset B but possessing dataset A and a model M(B) trained on B, seeks to infer variables from B for individuals shared between A and B (set Z). Leveraging the predictive capabilities of M(B) and information in A, the attacker attempts to reconstruct certain attributes from B for set Z individuals (individuals present in both A and B)
<br/>
2. <b>Membership Inference Attacks:</b> They don’t recover training data, but rather, ascertain whether a given individual’s data was a part of the training set or not. The holder of A and M(A) does not recover any columns in B, but can add an additional column to A representing whether or not an individual in A is also in B.

These attacks, where possible, risk models being considered as personal data. The question that remains is, what are the practical consequences of this, and are they of interest to those hoping to better control these systems and hold them to account?<br/>

### Implications of Models as Personal Data
- <b>Implications for Data Subjects</b><br><br>
  o	While individuals possess rights to access and portability of their personal data, the GDPR also mandates data controllers to disclose information to individuals without solicitation, such as the recipients of personal data and its sources. However, providing access to entire models trained on multiple individuals' data could compromise privacy and breach security principles. Instead, these information rights offer potential for better tracking of data provenance, enabling transparency and accountability in the data economy. Despite exemptions like disproportionate effort, controllers might be obligated to take measures to protect data subjects' rights, potentially including making information publicly available, thereby enhancing understanding of data flows and algorithmic accountability.<br><br>
- <b>Implications for Data Controllers</b><br><br>
   <b>Security Principle and Data Protection by Design:</b> Data controllers must prevent leakage of personal data through their systems to avoid breaches and violations of data protection principles, especially when transferring models outside the EU. Differential privacy is a common defence against model inversion attacks, adding random noise to queries to protect individual data. However, deployment challenges and limitations exist, particularly with outliers in datasets. Another defence is to prevent models from overfitting to data, although this alone may not fully protect against model inversion.
   <br><br>
   <b>Storage Limitation Principle:</b> The principle of storage limitation requires discarding data and potentially models when no longer needed. Techniques from concept drift adaptation, like managing data and gradually forgetting old correlations within models, can help address this. These methods ensure machine learning systems adapt to changing phenomena, which is crucial for mitigating vulnerabilities like model inversion.

## Formalizing Machine Unlearning
Unlearning is difficult due to the stochastic and complex nature of ML training methods. There’s no clear method to measure the impact of a single data point on the model’s parameters. Moreover, randomness in training, such as batch sampling and non-deterministic parallelization, complicates unlearning. Model updates reflect all prior updates, making it hard to isolate the influence of a single data point. Authors of paper [2] formalize machine unlearning:

**Machine Unlearning**: Let $\mathcal{D} = {d_i:i \in U}$ denote the training set collected from population U. Let $\mathcal{D'}=\mathcal{D} \cup d_u$. Let $D_M$ denote the distribution of models learned using mechanism $M$ on $\mathcal{D'}$ and then unlearning $d_u$. Let $D_{real}$ be the distribution of models learned using $M$ on $\mathcal{D}$. the mechanism $M$ facilitates unlearning when these two distribution are identical.

There are two aspects of this definition: $(i)$ the definition captures inherent stochasticity in learning: it is possible for multiple hypotheses to minimize empirical risk over a training set and ($ii$) the definition does not necessarily require that the owner retrain the model $M'$ from scratch on $\mathcal{D'} - d_u$, as long as they are able to provide evidence that model $M'$ could have been trained from scratch on $\mathcal{D'} - d_u$. 
For illustration, consider the following figure. Suppose we have a given dataset $\mathcal{D}$ and we want to train two DNN models $M_A$ and $M_C$. Now if we add a new data point $d_u$ to the training data $\mathcal{D}$, we get new dataset $\mathcal{D'}=\mathcal{D} \cup d_u$. Now using this dataset $\mathcal{D'}$, we can train a model $M_B$ in various way. First, we can use the parameter of $M_A$ that is trained on $\mathcal{D}$ and continue the training on $d_u$ to obtain $M_B$. However, it is difficult to understand the influence of $d_u$ in the parameter of $M_B$ and invert the procedure to get back $M_A$ unless we save a copy of $M_A$. One convincing way to obtain plausible deniability and ensure the removal of $d_u$ is to retrain the model from scratch without $d_u$ (keeping all hyperparameter same).  It is conceivable that the parameters of $M_A$ and $M_C$ are similar (despite stochasticity in learning) and it is
desired for their performance (in terms of test accuracy) to be comparable. 

<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_1.png" alt="Description of the image">
</p>

Goals of Unlearning are as follows:

- Intelligibility: The unlearning strategy should be conceptually easy to understand and implement.
- Comparable Accuracy: Even if the baseline accuracy degrades due to unlearning (e.g., removing a fraction of training points or prototypical points), any new strategy should aim to introduce only a small accuracy gap compared to the baseline.
- Reduced Unlearning Time: The strategy should be more efficient in terms of time and computational resources than the baseline approach.
- Provable Guarantees: Similar to the baseline, any new strategy must provide provable guarantees that unlearned points do not influence model parameters.
- Model Agnostic: The new unlearning strategy should work across various types of models, regardless of their complexity or nature.
- Limited Overhead: The new strategy should not introduce additional computational overhead beyond what is already required for training procedures.

## SISA Training Approach for Machine Unlearning

Figure below shows the SISA training approach. SISA training replicates the model being learned multiple times. Each replica receiving a disjoint subset of the dataset (similar to distributed training strategies). Each replica is referred to as a “constituent model.” Unlike traditional strategies, SISA training does not allow information flow between constituent models. Gradients computed on each constituent are not shared between different constituents. This isolation ensures that the influence of a specific shard (and its data points) is restricted to the model trained using it. Each shard is further partitioned into slices. Constituent models are trained incrementally (in a stateful manner) with an increasing number of slices. At inference, the test point is fed to each constituent, and their responses are aggregated (similar to ML ensembles). When a data point needs to be unlearned (e.g., due to privacy concerns or model updates), only the constituent model containing that data point is affected. Retraining can start from the last parameter state saved before including the slice with the unlearned data point. Only the models trained using the slice containing the unlearned point need to be retrained. We can divide the procedure in 4 parts: sharding, isolation, slicing, and aggregation.

**Sharding**: The dataset $\mathcal{D}$ is uniformly partitioned into $S$ shards ($D_k$) such that each shard contains a portion of the dataset. No data point belongs to more than one shard. The training cost can be distributed by dividing the dataset into smaller parts and training models on each part separately. This allows for parallelism across shards. If a user requests to unlearn a specific data point $d_u$, the service provider needs to locate the shard ($D_u$) that contains that data point. Then it retrains the model on the remaining data in that shard ($D_u - d_u$) to produce a new model ($M_{u}$). This process is faster than retraining the model from scratch on the entire dataset ($D - d_u$). The time required for retraining the model from scratch on the baseline dataset ($D - d_u$) is far greater than the time required for retraining on the smaller shard ($D_u - d_u$). This results in an expected speed-up of $S×$.

**Isolation**: Isolation could potentially degrade the generalization ability of the overall model beacuse it might not capture the full complexity of the data. However, through empirical demonstration, it is shown that this degradation doesn't happen in practice for certain learning tasks.

**Slicing**: Each shard's data $D_k$ is divided into $R$ disjoint slices, ensuring that each slice contains a portion of the shard's data. Autors follow incremental training:
   - At each step, the model is trained with an increasing number of slices, starting from one slice up to all slices.
   - The parameter state of the model is saved after each step.
   - Training for each step $i$:
     1. Train the model with random initialization using only slice $D_{k,i}$, for $e_i$ epochs, resulting in model $M_{k,i}$.
     2. For step $i > 1$, train the model $M_{k,i-1}$ using slices $D_{k,1} ∪ D_{k,2} ∪ ... ∪ D_{k,i}$, for $e_i$ epochs, resulting in model $M_{k,i}$.
     3. Repeat this process until all slices are included, obtaining the final model $M_{k,R} = M_{k}$.

Now, if a user requests to unlearn a specific data point $d_u$ from shard $D_k$, it locates the slice containing $d_u$, referred to as $D_{k,u}$. Then it performs training from that slice onwards, excluding $d_u$. That results in a new model $M_{k,u}$. For a single unlearning request, this method provides a best-case speed-up of up to $\frac{2R}{R+1}$ times compared to using the strategy without slicing.

**Aggregation**: The aggregation strategy is closely tied to how data is partitioned to form shards. It targets to maximize the joint predictive performance of constituent models.
This strategy should not rely on the training data to ensure that it does not need to be unlearned in some cases. In the absence of knowledge about which points will be the subject of unlearning requests, a uniform data partitioning and a simple label-based majority vote strategy are considered the best approach. This strategy ensures that each constituent model contributes equally to the final outcome. It also satisfies both requirements mentioned above. However, in scenarios where constituent models assign high scores to multiple classes, the majority vote aggregation might lose information about runner-up classes. To address this, a refinement strategy is evaluated where the entire prediction vectors (post-softmax) from constituent models are averaged. Then the label with the highest value is selected.

<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_sisa.png" alt="Description of the image">
</p>

## Measuring Time Analysis of SISA

**Measuring time for sharding**: For sequential setting, authors calculate the expectation of the number of points needed to be used for retraining. If the sharding is uniform, then each model has (roughly) the same number of initial training data points $\frac{N}{S}$. It is obvious that the first unlearning request will result in retraining of $\frac{N}{S}-1$ points for the one shard that is affected. For the second unlearning request, there will be two cases: the shard affected in the first unlearning request is affected again, which will result in retraining $\frac{N}{S}-1$ data points with a probability $\frac{1}{S}$, or any other shard is impacted resulting in retraining $\frac{N}{S}-1$ data points with probability $1-\frac{1}{S}$. Thus, inductively, we can see that for the $i^{th}$ unlearning request, the probability that $\frac{N}{S}-1-j$ points (for $0 \le j \le i − 1$) are retrained is:
<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_sharding_prop.png" alt="Description of the image">
</p>
Using binomial theorem, expected number of points to be retrained is:
<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_sharding_seq.png" alt="Description of the image">
</p>
Alternatively for batch setting, service provider S could aggregate unlearning requests into a batch, and service the batch. In this case the expected cost is:
<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_sharding_batch.png" alt="Description of the image">
</p>

**Measuring time for slicing**: For sequential setting, we need to find the expectation of the number of samples that will need to be retrained for a single unlearning request. The expected number of samples that need to retrain is:
<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_slicing_seq.png" alt="Description of the image">
</p>
which is an upper bound on the expected number of points to be retrained for a single unlearning request.

Similarly, for batch setting, we need to find the expected minimum value over multiple draws of a random variable to compute the index of the slice from which we will have to restart training. The minimum slice index (among all slices affected by $K$ unlearning requests) is computed under the assumption that multiple unlearning requests are sampled from a uniform distribution with replacement. The expected cost of this setting is:
<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_slicing_batch.png" alt="Description of the image">
</p>




## Evaluation of SISA
Model is evaluated on MNIST, Purchase, SVHN, CIFAR-100, ImageNET, and mini-imagenet dataset. Results are as follows:

**Impact of sharding**: Impact of sharding can be found in figure below. Despite providing similar benefits to batch $K$ and $\frac{1}{S}$ fraction baselines, SISA training shows more accuracy degradation for complex tasks like ImageNet. While consistently outperforming the $\frac{1}{S}$ fraction baseline, SISA training still faces challenges. With label aggregation results in an average top-5 accuracy degradation of 16.14 PPs. Varying the aggregation strategy mitigates this gap (Average improvements of 1.68 PPs in top-1 accuracy and 4.37 PPs in top-5 accuracy). It emphasizes the importance of ensuring each shard contains a sufficient number of data points to maintain high accuracy in constituent models.
<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_sharding.png" alt="Description of the image">
</p>

**Impact of slicing**: From figure below, we observe that slicing does not have detrimental impact on model accuracy in comparison to the approach without slicing if the training
time is the same for both approaches. It is clear that slicing reduces the retraining time so long as the storage overhead for storing the model state after adding a new slice is acceptable.
<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_slicing.png" alt="Description of the image">
</p>

**Combination of sharding and slicing**: From figure below, it can be shown that a combination of sharding and slicing induces the desired speed-up for a fixed number of unlearning requests
(0.003% the size of the corresponding datasets). Note that, the speed-up grows rapidly with an increase in S, but increasing S provides marginal gains in this regime.
<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_sharding_slicing.png" alt="Description of the image">
</p>

**Bridging the Accuracy Gap**: In a realistic deployment scenario, transfer learning from a base model (trained on ImageNet using ResNet-50) to the CIFAR-100 dataset, followed by SISA training, significantly reduces the accuracy gap between single-shard (S=1) and multi-shard (S>1) cases. We can show this from figure below. At S=10, the top-1 accuracy gap is reduced to around 4 PPs, and the top-5 accuracy gap to less than 1 PP. This approach allows for decreasing accuracy degradation induced by SISA training on complex tasks without varying constituent model hyperparameters while maintaining model homogeneity.
<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_accuracy_gap.png" alt="Description of the image">
</p>



## Distributional Knowledge aware SISA

Now, consider we relax assumptions and explore how knowledge of the distribution of unlearning requests can benefit service providers. Retraining time and accuracy degradation can be minimized by understanding which data points are more likely to be unlearned based on auxiliary information. For instance, grouping users likely to request data erasure into shards can reduce retraining time. For instance, show the figure below. Moreover, adapting sharding strategies based on the distribution of unlearning requests, such as concentrating points from high-risk groups into fewer partitions, can further reduce the number of shards needing retraining.
<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_distribution.png" alt="Description of the image">
</p>

**Distribution Aware Sharding**: Assume the service provider can create shards in a way so as to minimize the time required for retraining. One approach is shown in the algoritm below. Assume: $(i)$ the distribution of unlearning requests is known precisely, and $(ii)$ this distribution is relatively constant over a time interval. Now, each data point $d_u \in \mathcal{D}$ has an probability $p(u)$ to be erased. First the algorithm sorts the data points in the order of their erasure probability, and points to a shard $D_i$ till the desired value of $E(D_i)$ is reached. Once this value is exceeded, it creates a new shard $D_{i+1}$ and restart the procedure with the residual data $\mathcal{D} - \mathcal{D_i}$. By enforcing a uniform cumulative probability of unlearning a across shards, it naturally aggregates the training points that are likely to require unlearning into a fewer shards that are also smaller in size.

<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_distribution_aware.png" alt="Description of the image">
</p>

From above figure, we show the number of points to be retrained relative to the number of unlearning requests for both uniform and distribution-aware sharding strategies. The distribution-aware strategy decreases the expected number of points to be retrained and creates shards of unequal size. With 19 shards generated, this approach achieves approximately 94.4% prediction accuracy, which is one percent point lower than uniform sharding at 95.7%. This trade-off between accuracy and decreased unlearning overhead highlights the need for future exploration of alternative aggregation methods to address imbalanced shard sizes.

## Certified Data Removal from Machine Learning Models
The paper [3] explores the concept of certified data removal from machine learning models to ensure that no residual information from removed data can be extracted by adversaries. It introduces a framework for certified removal, providing a strong theoretical guarantee that models from which data has been removed cannot be distinguished from models that never contained the data. The study develops a practical removal mechanism for L2-regularized linear models using a differentiable convex loss function and investigates the application of this mechanism in different settings. Through experiments, it demonstrates that the removal mechanism can effectively eliminate the influence of deleted data points, with the residual error decreasing as the size of the training set increases. The paper emphasizes the balance between certified removal and model accuracy, highlighting the trade-offs involved in implementing such mechanisms in practical applications.

### Removal Mechanism
The authors delve into the specific application of their certified removal mechanism within the context of linear models. After defining a linear classifier, where the training set  $D$ comprises pairs $(x_i, y_i)$ with $x_i$ representing a feature vector and $y_i$ the target. The learning algorithm $A$ is tasked with minimizing the regularized empirical risk using a convex loss function, which is typical in logistic and linear regression models. The certified removal mechanism aims to delete a specific data point $(x, y)$
 such that the adjusted model parameters reflect a model that was never trained with that data point. This is accomplished by applying a Newton step to update the parameters based on the gradient and Hessian matrix of the loss function, calculated at the current model parameters.The Newton update mechanism is mentioned as follows: 
<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/cert1.png" alt="Description of the image">
</p>


Implementation details further elaborate on the mechanism's operational aspects. The Newton update formula provided calculates the gradient and Hessian for the model parameters, excluding the removed data point, aiming to minimize the loss function $L(w; D')$, where $D'$ represents the dataset sans the removed data point.
Post-update, residual information from the removed data point might still influence the model. To mitigate this, the authors suggest adding random perturbation during training, which helps obscure any residual data that the gradient could disclose about the removed point. The section concludes with theoretical guarantees, providing bounds on the norm of the gradient residual. These mathematical boundaries are instrumental in measuring how effectively the updated model parameters mimic those of a model trained without the removed data point, thus preserving the model's integrity while minimizing the influence of the deleted data. The algorithms is as follows:


<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/cert2.png" alt="Description of the image">
</p>

### Experiments and Results
We test our certified removal mechanism in three settings: 
**(1) removal from a standard linear logistic regressor**

The experiments with the certified removal mechanism on the MNIST dataset, specifically focusing on the binary classification of digits 3 and 8. They use a regularized logistic regression model to explore the impact of varying the L2-regularization parameter ($\lambda$) and the standard deviation ($\sigma$) of the objective perturbation on test accuracy and the number of data removals the model can handle before needing retraining.

The experiments reveal key trade-offs:
- Adjusting $\lambda$ and $\sigma$ impacts both test accuracy and the capability for data removals. Higher $\lambda$ values increase the number of possible removals by reducing the gradient residual norm but can decrease accuracy if too large.
- A balance between model accuracy and data removal robustness is necessary, with higher removal capacity generally leading to slightly lower accuracy.

These results underscore the effectiveness of the certified removal mechanism in managing data deletions within model parameters, highlighting the necessity of fine-tuning the parameters to balance privacy concerns with model performance.

 **(2) removal from a linear logistic regressor that uses a feature extractor pre-trained on public data**

The authors evaluate their certified removal mechanism in scenarios where a feature extractor is pre-trained on public data. The experiments focus on two tasks: scene classification on the LSUN dataset and sentiment classification on the Stanford Sentiment Treebank (SST) dataset. The feature extractors used include a ResNeXt-101 model trained on Instagram images for LSUN, and a RoBERTa language model for SST.

Key findings from the experiments include:
- **LSUN Results**: By converting the 10-way classification task into 10 binary classifications, and balancing the classes, they demonstrate that the model supports over 10,000 removals before requiring re-training, with only a minor drop in accuracy (from 88.6% to 83.3%). This is achieved while reducing the computational cost of removal by more than 250 times compared to re-training.
- **SST Results**: For sentiment analysis, the regular model matches competitive benchmarks with 89.0% accuracy, and a slight reduction in accuracy allows for a significant increase in the number of supported removals. The cost for removal is 870 times lower than re-training.

These experiments highlight the utility of the certified removal mechanism in applications involving complex datasets and models, showing that it effectively maintains model performance while enabling robust data removal capabilities. The figure below demonstrate the results:

<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/cert3.png" alt="Description of the image">
</p>


**(3) removal from a non-linear logistic regressor by using a differentially private feature extractor**

 The authors explore the use of a differentially private feature extractor for training on private data, applying their certified removal mechanism to the logistic regression models that use these features. This approach is tested for scenarios where public data is unavailable for feature training, emphasizing the practical benefits of integrating differential privacy at the feature extraction stage. 

Key takeaways include:
- **Enhanced Model Accuracy**: Using a differentially private feature extractor allows for more precise noise application, improving the final model's accuracy compared to models trained entirely under differential privacy constraints.
- **Effective Data Removal**: The approach demonstrates that robust data deletion capabilities can be maintained without sacrificing model utility, even with stringent privacy measures from the start of training.

This section highlights the practicality and effectiveness of combining certified data removal with differential privacy in deep learning models, showing significant advantages in terms of both privacy protection and model performance. The same is demonstrated the figure below:

<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/cert4.png" alt="Description of the image">
</p>


## A Survey of Machine Learning

The authors of Paper [4] aim to capture the key concepts of unlearning techniques. In their survey, the existing solutions are classified and summarized based on their characteristics within an up-to-date and comprehensive review of each category’s advantages and limitations.

The figure below shows the overview of machine unlearning and its ecosystem, which presents the typical concept, unlearning targets, and desiderata associated with machine unlearning.

<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_86_ecosystem.png" alt="Description of the image">
</p>

### Targets of machine unlearning
The ultimate target of machine unlearning is to reproduce a model that (1) behaves as if trained without seeing the unlearned data and (2) consumes as less time as possible. The performance baseline of an unlearned model is that of the model retrained from scratch (a.k.a., native retraining). Specfically, the authors categorized the three following targets in machine unlearning and listed their advantages and limitations:

| Targets          | Aims                                                                 | Advantages                                                       | Limitations                                           |
| ---------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------- |
| Exact Unlearning | To make the distributions of a natively retrained model and an unlearned model indistinguishable | Ensures that attackers cannot recover any information from the unlearned model | Difficult to implement                               |
| Strong Unlearning | To ensure that the distributions of two models are approximately indistinguishable | Easier to implement than exact unlearning                        | Attackers can still recover some information from the unlearned model |
| Weak Unlearning  | To only ensure that the distributions of two final activations are indistinguishable | The easiest target for machine unlearning                        | Cannot guarantee whether the internal parameters of the model are successfully unlearned |

They also illustrate the targets of machine unlearning and their relationship with a trained model using the figure below. The different targets, in essence, correspond to the requirement of unlearning results.
<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_86_targets.png" alt="Description of the image">
</p>

### Desiderata of machine unlearning
The desiderata of machine unlearning can be summarized as the following three:
- Consistency: how similar the behavior of a retrained model and an unlearned model is. 
- Accuracy: if the unlearned model to predict samples correctly. 
- Verifiability: whether a model provider has successfully unlearned the requested unlearning dataset.

### Taxonomy of unlearning methods and verification mechanisms

The following figure summarizes the general taxonomy of machine unlearning and its verification used in this survey. The taxonomy is inspired by the design details of the unlearning strategy. Unlearning approaches that concentrate on modifying the training data are classified in data reorganization, while methods that directly manipulate the weights of a trained model are denoted as model manipulation. As for verification methods, initially, the authors categorize those schemes as either experimental or theoretical; subsequently, they summarize these methods based on the metrics they use.

<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_86_taxonomy.png" alt="Description of the image">
</p>

The following table summarizes and compares each verification method’s advantages and limitations.

| Methods            | Basic Ideas                                                             | Advantages                                  | Limitations                                         |
|--------------------|-------------------------------------------------------------------------|---------------------------------------------|-----------------------------------------------------|
| Retraining-based   | Removes unlearned samples and retrains models                           | Intuitive and easy to understand            | Only applicable to special unlearning schemes       |
| Attack-based       | Based on membership inference attacks or model inversion attacks        | Intuitively measures the defense effect against some attacks | Inadequate verification capability          |
| Relearn time-based | Measures the time when the unlearned model regains performance on unlearned samples | Easy to understand and easy to implement   | Inadequate verification capability                  |
| Accuracy-based     | Same as a model trained without unlearned samples                       | Easy to understand and easy to implement    | Inadequate verification capability                  |
| Theory-based       | Ensures similarity between the unlearned model and the retrained model. | Comprehensive and has theoretical support   | Implementation is complex and only applies to some specified models |
| Information bound-based | Measures the upper-bound of the residual information about the unlearned samples | Comprehensive and has theoretical support | Hard to implement and only applicable to some specified models |

### Data reorganization in machine unlearning

#### Data Obfuscation

The figure below shows, in data obfuscation, when receiving an unlearning request, the model continues to train $w$ based on the constructed obfuscation data $D_{obf}$ giving rise to an updated $w_u$.

<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_86_data_obfuscation.png" alt="Description of the image">
</p>

Verifiability of Schemes Based on Data Obfuscation:
- Graves et al.: 
  - model inversion attack
  - membership inference attack
- Tarrun et al.:
  - assessment of relearning time by measuring the number of epochs for the unlearned model to reach the same accuracy as the originally trained model. 
  - assessment of the distance between the original model, the model after the unlearning process, and the retrained model


#### Data Pruning

The following figure displays the data pruning unlearning methods, which shows that, unlearning schemes based on data pruning are usually based on ensemble learning techniques.

<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_86_data_pruning.png" alt="Description of the image">
</p>

Verifiability of Schemes Based on Data Pruning:
- Retraining-based verification
  - retrain the model from scratch after removing the samples that need to be unlearned from the training dataset
- Backdoor verification
  - design a specially crafted trigger and implant this “backdoor data” in the samples that need to be unlearned, with little effect on the model’s accuracy
  - verify the validity of the unlearning process based on whether the backdoor data can be used to attack the unlearned model with a high success rate

#### Data Replacement

As shown in the below figure, when training a model in a data replacement scheme, the first step is usually to transform the training dataset into an easily unlearned type, named transformation $T$. Those transformations are then used to separately train models. When an unlearning request arrives, only a portion of the transformations $t_i$—the ones that contain the unlearned samples—need to be updated and used to retrain each submodel to complete the machine unlearning.

<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_86_data_replacement.png" alt="Description of the image">
</p>

Verifiability of Schemes Based on Data Replacement
- Accuracy-based verification
  - perform data pollution attacks to influence the accuracy of those models
  - analyze whether the model’s performance after the unlearning process was restored to the same state as before the pollution attacks
  - If the unlearned model was actually restored to its pre-pollution value, then the unlearning operation was considered to be successful.


### Model manipulation in machine unlearning

#### Model Shifting

Model-shifting methods usually eliminate the influence of unlearning data by directly updating the model parameters. These methods mainly fall into one of two types—influence unlearning and Fisher unlearning.

<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_86_model_shifting.png" alt="Description of the image">
</p>

Unlearning Schemes Based on Model Shifting
- Influence unlearning methods
  - Certified removal: only applicable to simple ML models, e.g., linear models
  - Projection residual update: focus on linear regression
  - Certified unlearning: only suitable for tabular data
- Fisher unlearning methods
  - Kullback-Leibler (KL) based: limited applicability with various assumptions
  - Neural tangent kernel (NTK) based: information can be inferred from middle layers
  - Fisher Information Matrix (FIM) based: significant reduction in accuracy


Verifiability of Schemes Based on Parameter Shifting
- Model confidence and information bound
  - measure the distribution of the entropy of the output predictions on the remaining dataset, the unlearning dataset, and the test dataset
  - evaluate the similarity of those distributions against the confidence of a trained model that has never seen the unlearning dataset
  - use KL-divergence to measure the information remaining about the unlearning dataset within the model after the unlearning process

#### Model Pruning

Methods based on model pruning usually prune a trained model to produce a model that can meet the requests of unlearning. It is usually applied in the scenario of federated learning, where a model provider can modify the model’s historical parameters as an update. 
<!-- Federated learning is a distributed machine learning framework that can train a unified deep learning model across multiple decentralized nodes, where each node holds its own local data samples for training, and those samples never need to be exchanged with any other nodes -->

<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_86_model_pruning.png" alt="Description of the image">
</p>

Verifiability of Schemes Based on Model Pruning
- Membership inference attack
  - Attack precision & attack recall
  - Prediction difference:  the difference in prediction probabilities between the original global model and the unlearned model
- Bayes error rate-based divergence measurement [56]
  - Evaluate the similarity of the resulting distributions for the pre-softmax outputs of the unlearned model and a retrained model


#### Model Replacement

Model replacement-based methods usually calculate almost all possible sub-models in advance during the training process and store them together with the deployed model. Then, when an unlearning request arrives, only the sub-models affected by the unlearning operation need to be replaced with the pre-stored sub-models. This type of solution is usually suitable for some machine learning models, such as tree-based models. 

<p align="center">
  <img src="https://github.com/Nibir088/Responsible-AI-Group-3-/blob/main/img/unlearning_86_model_replacement.png" alt="Description of the image">
</p>

Verifiability of Schemes Based on Model Replacement
- Membership inference attack & technique based on false negative rates
  - If the target model successfully unlearns the samples, then the member inference attack will treat the training dataset as non-training data. 
  - Thus, FN will be large, while TP will be small, and the corresponding FNR will be large.

### Summary of unlearning schemes

Here the authors provide a summary and comparison of differences between different unlearning schemes.

| Schemes           | Basic Ideas                                           | Advantages                                  | Limitations                                                                 |
|-------------------|-------------------------------------------------------|---------------------------------------------|-----------------------------------------------------------------------------|
| Data Obfuscation  | Intentionally adds some choreographed dataset to the training dataset and retrains the model | Can be applied to almost all types of models; not too much intermediate redundant data need to be retained | Not easy to completely unlearn information from models |
| Data Pruning      | Deletes the unlearned samples from sub-datasets that contain those unlearned samples. Then only retrains the sub-models that are affected by those samples | Easy to implement and understand; completes the unlearning process at a faster speed | Additional storage space is required; accuracy can be decreased with an increase in the number of sub-datasets |
| Data Replacement  | Deliberately replaces the training dataset with some new transformed dataset | Supports completely unlearn information from models; easy to implement | Hard to retain all the information about the original dataset through replacement |
| Model Shifting    | Directly updates model parameters to offset the impact of unlearned samples on the model | Does not require too much intermediate parameter storage; can provide theoretical verification | Not easy to find an appropriate offset value for complex models; calculating offset value is usually complex |
| Model Pruning     | Replaces partial parameters with pre-calculated parameters | Reduces the cost caused by intermediate storage; the unlearning process can be completed at a faster speed | Only applicable to partial models; not easy to implement and understand |
| Model Replacement | Prunes some parameters from already-trained models     | Easy to completely unlearn information from models | Only applicable to partial machine learning models; original model structure is usually changed |

### Open questions and future questions
Some questions are raised in the survey which are open challenges in the area of machine unlearning.
- The Universality of Unlearning Solutions
  - Most of the current unlearning schemes are limited to a specific scenario. 
- The Security of Machine Unlearning
  - The unlearning operation not only does not reduce the risk of user privacy leakage but actually increases this risk
- The Verification of Machine Unlearning
  - Simple verification schemes, such as those based on attacks, relearning time, and accuracy seldom provide strong verification of the unlearning process’s effectiveness
  - Unlearning methods with a theoretical guarantee are usually based on rich assumptions and can rarely be applied to complex models
- The Applications of Machine Unlearning
  - In addition to strengthening data protection, machine unlearning has enormous potential in other areas

The authors also provide some directions for future research:
- Information synchronization
- Federated unlearning
- Disturbance techniques
- Feature-based unlearning methods
- Game-theory-based balance

## Conclusion & Discussion

In conclusion, the paper [2] introduces a framework to expedite the unlearning process by strategically limiting the influence of a data point in the training procedure. Moreover, it is applicable to any learning algorithm but particularly beneficial for stateful algorithms like stochastic gradient descent for deep neural networks.

**Computational Overhead Reduction**: Demonstrates how SISA training significantly reduces the computational overhead associated with unlearning. Even it shows improvement in scenarios where unlearning requests are uniformly distributed across the training set.

**Practical Data Governance**: Contributes to practical data governance by enabling machine learning models to unlearn data efficiently, thereby supporting the right to be forgotten as mandated by privacy regulations like GDPR.

**Evaluation Across Datasets**: Provides an extensive evaluation of the SISA training approach across several datasets from different domains. This shows its effectiveness in handling streams of unlearning requests with minimal impact on model accuracy.

Paper [4] provdes us a comphrensive survey of machine learning, organizing current research with its proposed taxonomy, and sharing open questions and future directions for follow-up studies.

## References

[1] Algorithms that remember: model inversion attacks and data protection law Veale et al. 2018

[2] Machine Unlearning Bourtoule et al. 2019

[3] Certified Data Removal from Machine Learning Models Guo et al. 2019

[4] Machine Unlearning: A Survey Xu et al. 2023.

