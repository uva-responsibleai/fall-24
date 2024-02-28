# Introduction
Software systems based on deep neural networks and other machine learning algorithms are becoming more and more common across all facets of the industry, such as in social media content recommendation and autonomous driving. They are used to make consequential decisions with serious impacts that can be as seemingly naive as choosing which picture to recommend first on a social media timeline, or much more imposing as when used for scenarios involving self-driving vehicles or healthcare when lives may be at stake. Software bugs or underlying problems in these models are practically unavoidable in the ever-lasting pursuit of perfect performance, but many of these bugs are never caught and lead to incorrect behavior without anyone knowing.\
In this light, software based on machine learning is notoriously difficult to test and maintain. A potential factor that contributes to diminishing performance for ML models is the distribution shift when the underlying patterns of data change over time. Seemingly subtle changes in the data distribution can destroy a model’s performance without proper care. Typical model development and maintenance pipelines rarely inspect incoming data for signs of distribution shift, so much often goes unnoticed. Given that these shifts are somewhat inevitable, it is important to determine when they are serious enough to have an effect that should be worried about.\
The growing complexity of neural networks raises the importance of reliability and trust in the model as the transparency of deep neural networks diminishes. As human intuition wanes in its ability to grasp the boundaries of model capabilities it remains vital that predictive uncertainty is accurate and well communicated. Machine learning will expand into wider domains and accurate calibration is a crucial way to ensure the trust and safety of users in new applications. 

## Motivations
Machine learning (ML) systems are very brittle to change in their inputs which has been shown through adversarial examples. ML systems rely on data for their optimal functioning and are sensitive to changes in the distribution of the data being fed into them. Big data systems have enabled the collection of enormous amounts of data on a daily basis which has further perpetuated the continuous distribution shift in these data systems. Distribution shifts can be envisioned as imbalanced class data, the addition of adversarial samples, or simply the addition of noise to the test data. Therefore, there is a need for monitoring the input data being fed into ML systems for signs of distribution shift and to prevent failure of these systems. The works discussed in this blog focus on various challenges related to distribution shifts ranging from the detection of distribution shifts in real-life ML systems to reporting uncertaining outputs due to unfamiliar data distribution.

The motivation behind the development of the “Deep Gamblers” model stems from the challenge of assessing the confidence level of predictions made by deep learning models, particularly in critical fields such as physics, biology, chemistry, and healthcare. The authors identify a gap in existing methods for effectively and efficiently assessing prediction uncertainty in deep learning models. 

# Methods
In real-time systems, high-dimensional data makes it difficult to process this data and monitor the shift in its distribution. Given labeled data and unlabeled data, shift detection attempts to determine whether or not the distribution of the knowns and unknowns follow suit.\ 
Therefore, for investigating data shifts the primary steps involve **dimensionality reduction** which can be employed with several different techniques, as seen below, for better understanding and applicability. 
- **No reduction** is used as a default baseline for comparison.
- **Principal Components Analysis (PCA)** involves a transformation that is meant to capture as much of the variability in the dataset as possible using as few components as possible.
- **Sparse Random Projection (SRP)** involves random projections entailing compromised accuracy in favor of heightened performance through faster processing times.
- **Autoencoders** use an encoder function and a decoder function to reduce the latent space to a lower dimensionality than the input space. Both of these functions are learned jointly to reduce the reconstruction loss.
- **Label Classifiers** is a strategy that uses the outputs of a (deep network) label classifier trained on source data as the dimensionality-reduced representation.
- **Domain Classifier** attempts to detect shifts by explicitly training a domain classifier to discriminate between data from source and target domains.\

Distribution shifts encourage research in the field of calibrating real-time systems. A few important terminology and observations to understand model calibration are as follows:

- **Model Calibration** A model f is well-calibrated if its output truthfully quantifies the predictive uncertainty.
- $\forall_p \in \Delta: P(Y = y | f(X) = p) = p_y.$ 
- **Expected Calibration Error(ECE)** A widely used metric for assessing miscalibration evaluates the anticipated discrepancy between the model's predicted accuracy and the actual observed probability.
- $E[|p\ast - E| Y \in argmax f(X) | maxf(X) = p\ast|]$

- **Estimation of ECE** Since for the event with probability = 0 or close to 0, there will not be any examples for us to test, normally people will first bucket the prediction into m bins based on their top predicted probability, then take the expectation over these buckets.
- $ECE_d = \sum_{i = 1}^{m}\frac{B_i}{n}|accuracy(B_i) - confidence(B_i)|$

Two biases impact calibration measurements: binning-induced bias, which is negative, and sample size bias, which is positive. These biases mean the ECE estimator can both under- and overestimate true calibration, affecting model comparisons.
Positive bias from sample size is affected by model accuracy, with more bins increasing bias for models of mid-range accuracy. High or low-accuracy models are less impacted due to lower variance and better prediction-outcome correlation.
Section 5 delves into how model accuracy influences ECE bias, focusing on a squared difference ECE metric. It highlights the complex relationship between model performance and calibration accuracy estimation.

- **l2ECE Sampling Bias** takes the squared differences instead of the absolute differences for tractable calculations. The bias for this estimator depends on the model accuracy and the confidence difference. For the l2ECE increasing accuracy reduces bias if they have a top-1 accuracy above 50%.

Experiments Setup of Model Calibration
 Methodology:
In-distribution Calibration & Distribution shift
with temperature scaling & without temperature scaling
Compare the changes in prediction error and expected calibration error of each model on the image dataset with and without using temperature scaling, respectively
		
affecting factors:
model size
Previous research has shown that larger neural networks have more calibration issues.

Pretraining
Because it is a popular trend today to pre-train image models with a large amount of data and then fine-tune them with a small amount of final task data for transfer learning, compare the impact of the duration of model pre-training and the amount of data used on the ECE and accuracy of the model.
By separately controlling one of the two variables, the training step and the amount of training data, compare the impact of changes in a single variable on ECE and accuracy.

Once the dimensionality reduction process has been completed successfully, an appropriate hypothesis testing technique must be chosen. Such techniques include the following:
- **Multivariate Kernel Two-Sample Tests: Maximum Mean Discrepancy (MMD)**
- **Multiple Univariate Testing: Kolmogorov-Smirnov (KS) Test + Bonferroni Correction**
- **Categorical Testing: Chi-Squared Test**
- **Binomial Testing**

The “Deep Gamblers” proposes a novel method for implementing selective classification in deep learning models, inspired by portfolio theory, to allow models to abstain from making predictions when uncertain. The key aspects of the method include:
- **Modification of the Loss Function:** The traditional cross-entropy loss for a m+1-class classification problem is modified to incorporate an additional class, effectively making it a m+1-class problem where the extra class represents the model's decision to abstain. This modification encourages the model to learn when to abstain from making a prediction due to uncertainty.
- **Gambler's Loss Implementation:** The loss function, referred to as the gambler's loss, is designed to optimize the balance between betting on confident predictions and abstaining when the model is uncertain. This is achieved by changing the output dimension from m to m+1 and adjusting the training loss to include the abstention option.
- **Hyperparameter o:** A hyperparameter o is introduced, which needs to be tuned. It controls the trade-off between making predictions and abstaining, ensuring that the model only abstains when it is significantly uncertain. The value of o is critical for the model's performance, with specific ranges suggested for different datasets to avoid trivial solutions where the model might always choose to abstain.

# Key Findings
The [Failing Loudly](https://arxiv.org/abs/1810.11953) paper concludes by contributing the following findings:
- Black-box shift detection with soft predictions works well across a wide variety of shifts, even when some of the underlying assumptions do not hold
- Aggregated univariate tests performed separately on each latent dimension offer comparable shaft detection performance to multivariate two-sample tests
- Harnessing predictions from domain-discriminating classifiers enables characterization of a shift’s type and its malignancy
- Model size can not fully explain the intrinsic calibration differences between the models
- More pretraining data consistently increases accuracy but has no consistent effect on calibration
- Larger and more accurate models can maintain their good in-distribution calibration even under severe distribution shifts.
- The most accurate available but less-calibrated model should be chosen. 

The key findings from Deep Gambler are as follows:
- Model Performance on Uncertainty Identification: The proposed method can effectively identify uncertainty in data points. It excels in differentiating between confident predictions and those where abstention due to uncertainty is preferable. This capability is demonstrated through experiments on synthetic Gaussian datasets, MNIST, and other real-world datasets like SVHN and CIFAR-10.
- Outperforming Baselines: The method competes strongly against state-of-the-art (SOTA) models in selective classification, particularly in cases requiring high confidence levels. For instance, it outperforms entropy-based selection methods by correctly identifying disconfident images that do not resemble numbers, indicating a more nuanced understanding of uncertainty.
- Flexibility and Simplicity: The approach is highlighted for its simplicity and flexibility. It requires minimal modifications to the model inference algorithm or the model architecture, making it an attractive option for enhancing existing deep learning models with an abstention capability.


# Critical Analysis
The [Failing Loudly](https://arxiv.org/abs/1810.11953) paper provides an excellent evaluation and comparison of statistical measures used to investigate the severity of distribution shifts. The analysis is very detailed and technical with myriad figures and specific experimental diagrams from their testing. This is potentially both a strength and a weakness, in that they provide the reader with a lot of detailed figures through an appendix that far surpasses the length of the main portion of their work. However, this could be seen as overwhelming by some. It seems like the highly technical aspect of the piece could be hard to follow for readers who are not as familiar with a broad range of machine learning methodologies, but the inclusion of this detail caters well to the target audience of those truly interested in preventing distribution shifts.

The [Calibration](https://arxiv.org/pdf/2106.07998.pdf) paper evaluates the calibration properties of modern architectures with great clarity and compares them to the previous trends of declining calibration. The finding that non-convolution models are breaking from the trend to actually have better calibration is a novel result that probes the importance of architecture for eventual calibration quality. This paper deftly handles the specified image classification architectures but could benefit from a greater diversity in datasets and problem types. This lack of diversity hampers the generalizability of the paper’s findings to other domains. The main ethical implications of the paper are on the critical applications of machine learning algorithms that could have drastic implications if the models are unreasonably overconfident such as in healthcare or self-driving applications. This is where understanding the boundaries of the calibration accuracy are necessary to understand when a model is trustworthy in real-world applications. The ability to robustly handle distribution shift away from the training set is a vital criteria for implementation in safety-critical domains. While the biases of the authors may lead them to solve machine learning problems with machine learning solutions, they generally are open to the downsides of these models and are careful in their analysis throughout. 

The [Deep Gambler](https://arxiv.org/pdf/1907.00208.pdf)paper underscores the potential of the proposed method to enhance the flexibility and effectiveness of deep learning models in selective classification tasks. It highlights the method's ability to learn more nuanced data representations and suggests significant implications for both practical applications and theoretical advancements in the field.


