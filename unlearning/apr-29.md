# LLMs: Targeted Unlearning

## Introduction

Pretrained Language Models memorize a vast amount of knowledge during initial pretraining, including information that may violate the privacy of personal lives and identities. Recent work has shown that an adversary can extract training data from Pretrained Language Models including Personally Identifiable Information (PII) such as names, phone numbers, and email addresses. In 2021, an AI chatbot Iruda became the first AI system to be sued for violating the Personal Information Protection Act after generating the exact home addresses and bank account numbers of actual individuals unintentionally.

Practitioners are required to delete personal information from the LMs by individuals’ request because each individual has the “Right To Be Forgotten (RTBF)" and can limit the direct and indirect commercial use of their personal information. To this end, knowledge unlearning is proposed as an efficient solution.

In this presentation, we will introduce different methods used for targeted unlearning, as well as benchmarks used for assessing how well these methods perform in real-world unlearning tasks.

## Motivation
Previous methods addressing privacy risks for language models attempt to remove all private information from the training data (data preprocessing) or attempt to design algorithms that ensure differential privacy (DP). Both approaches require retraining the underlying LM every time individuals want to practice their “Right To Be Forgotten (RTBF)", which makes them inadequate for large LMs that are extremely costly to retrain. 

Furthermore, data preprocessing methods assume private information to be easily identifiable, specified, and removed and DP algorithms can only guarantee protection for information that has clear privacy borders, which makes them inadequate in the real-world scenarios where the standard of privacy might differ by each individual.

Some prior work focuses on classification models, but with recent advancement in chatbots and instruction-tuend LLMs, we need to shift our attention to question and answer tasks that reflect the way most people interact with LLMs. 

## Methods
 # Knowledge Unlearning by Gradient Ascent [87]
  Unlearning Method:

We propose simply negating the original training objective of minimizing the negative log-likelihood of the token sequences as our main method of knowledge unlearning in LMs. Specifically, given a sequence of tokens x = (x1,...,xT), our unlearning training objective is simply maximizing the following loss function:

Quantifying Privacy Risks of Language Models:

Extraction Likelihood (EL)

Extraction Likelihood can be seen as estimating the general extraction likelihood, since it is essentially measuring the average success rate of varying extraction attacks quantified via getting the n-gram overlap of generated and target token sequences. 

While previous metrics quantifying the privacy risks of LMs are dependent on specific adversarial attacks, this characteristic of EL allows it to quantify the general likelihood of extraction without any dependency on specific extraction attacks.

Memorization Accuracy (MA)

Memorization Accuracy quantifies how much the language model has memorized the given token sequences and can be used to analyze the training dynamics of large LMs.

	
We empirically define a specific token sequence x to be forgotten and no longer susceptible to extraction attacks when the EL(x) and MA(x) reach a value that is lower than the average EL and MA on token sequences that were not seen during training.

 # TOFU [89]
 TOFU Dataset: This dataset is created by instructing GPT-4 to produce information about each author using predetermined characteristics like birthplace, gender, birth year, literary genre, accolades, and parental occupations. With these attributes as the foundation, the model is assigned the task of generating 20 pairs of questions and answers for each fictional author. 

For the evaluation metric, the author considers two properties: Model Utility and Forget Quality. In order to facilitate these two properties, the author introduces four evaluation datasets: Forget Set, Retain Set,Real Authors,and World Facts.

The Forget Set contains questions and answers about the works of a specific number of fictitious authors. The model needs to forget or unlearn this information.
The Retain Set ensures that even after unlearning the Forget Set, the model performs well on questions about other fake authors included in the fine-tuning data.
Real Authors dataset evaluates the model's ability on questions about real-world authors, gradually moving away from the Forget Set's concepts.
World Facts dataset tests the model's general world knowledge to confirm that unlearning is targeted and doesn't degrade the model's broader factual accuracy.

The three levels of distance from the dataset being unlearned—Retain Set, Real Authors, and World Facts—offer a spectrum of relevance, aiding in gauging the accuracy of the unlearning process. The goal is to refine the model's forgetting mechanism to selectively discard specific undesired information while preserving the remainder.


Unlearning Algorithm:
Gradient Ascent: The Gradient Ascent approach is fundamentally straightforward. It entails reducing the likelihood of correct predictions on the forget set. Specifically, for each instance in SF, the goal is to maximize the standard training loss in order to make the model deviate from its initial prediction. As in the finetuning stage, the loss on a given sample x ∈ SF is denoted by ℓ(x,w); and the loss we aim to maximize is the average over the forget set,


Model Utility: For the Forget Set and Retain Set, we calculate the conditional probability P(a|q) based on the model's output and raise it to the power of 1/|a| to normalize for answer length, following a common practice.

On the other hand, for Real Authors and World Facts, we treat each question q as a multiple-choice question with choices {a1, ..., an}. Without loss of generality, let's assume that a1 is the correct answer. Then, the probability is computed as:

The author also utilizes ROUGE scores to compare model-generated answers (using greedy sampling) with the ground truth. In particular, they calculate the ROUGE-L recall score (Lin, 2004), which serves as a proxy for accuracy in the question answering task. This metric considers that the phrasing of the output may vary slightly from the ground truth.
	
	For a given question, the author computes a ratio that approximately compares how likely its correct answer is to an incorrect answer. Let ˜ a represent a paraphrased version of the answer, and thus, ˜ x = [q, ˜ a]. We generate paraphrased strings by instructing GPT-4 to paraphrase the answer. Additionally, we create a set of five perturbations Apert with GPT-4 by requesting modifications of the answer that maintain the general template of the text but introduce factual inaccuracies.

The truth ratio, denoted as Rtruth, can be expressed as follows.

Model Finetuning:The loss on a sample x ∈ S is expressed as a function of model weights w, given by


where NLLw is the negative log likelihood according to the outputs of a model parameterized by w. Then, researchers aim to find w∗ that minimizes the average loss over the dataset denoted by L. 


Unlearning Algorithm:
Gradient Ascent: The Gradient Ascent approach is fundamentally straightforward. It entails reducing the likelihood of correct predictions on the forget set. Specifically, for each instance in SF, the goal is to maximize the standard training loss in order to make the model deviate from its initial prediction. As in the finetuning stage, the loss on a given sample x ∈ SF is denoted by ℓ(x,w); and the loss we aim to maximize is the average over the forget set,

Gradient Difference: The second method, called Gradient Difference, builds on the concept of gradient ascent. It not only aims to increase the loss on the forget set SF, but also strives to maintain performance on the retain set SR. The revised loss function we aim to minimize can be represented as

KL Divergence: In the KL Minimization approach, the objective is to minimize the Kullback-Leibler (KL) divergence between the predictions on SR of the original (finetuned on TOFU) and the newly trained models (as it undergoes unlearning) while maximizing the conventional loss on SF. Let M denote a model and let M(·) output a probability distribution over the vocabulary corresponding to the likelihood of the next token according to the model. The formal objective can be written as

Here, M_original and M_current denote the original and the new model, respectively. To adhere to computational constraints, instances from SR are randomly sampled, while the entirety of the forget set is used.

 # WMDP Benchmark

 The paper describes a novel method called Representation Misdirection for Unlearning (RMU), aimed at removing hazardous knowledge from language models while retaining their ability to process non-hazardous information. This method is specifically applied to areas like biosecurity and cybersecurity but is not used for chemical security due to potential trade-offs in model performance.
The RMU method utilizes an autoregressive language model that generates responses to prompts. The primary goal is to diminish the model's capability to respond accurately to queries about hazardous information, such as synthesizing anthrax, while ensuring it remains adept at answering benign queries like culturing yeast. This dual objective is assessed by reducing the model's accuracy on a specific dataset designed for hazardous knowledge (WMDP) without compromising its performance on broader capability benchmarks such as MMLU and MT-Bench.
RMU employs a two-part loss function to achieve these goals. The Forget Loss increases the model’s activations on hazardous data in the initial layers, which complicates the processing of this data in subsequent layers, effectively 'forgetting' the risky information. This loss is calculated by altering the distance between the model's activations and a predefined random unit vector, scaled appropriately. The Retain Loss, on the other hand, preserves the model's functionality on benign data by minimizing the differences between the current model's activations and those of a 'frozen' baseline model. The overall loss function combines these two aspects with a balancing parameter, guiding the model updates to forget hazardous knowledge while retaining essential general knowledge.
For practical application, RMU uses two datasets: a Forget Dataset (Dforget) that includes data resembling the hazardous knowledge to be unlearned, and a Retain Dataset (Dretain) that comprises general knowledge to ensure the model's functionality in safe domains is maintained. Initially, specific domain-related datasets were considered for retention, but the broader and more generic Wikitext dataset was chosen to prevent the relearning of hazardous knowledge.
RMU represents a strategic approach to enhancing the safety of AI applications by meticulously reducing their ability to misuse hazardous knowledge while maintaining their overall utility. This method provides a blueprint for future advancements in machine learning safety and the responsible deployment of AI technologies.


 # Rethinking Machine Unlearning [88]
How can we efficiently and effectively eliminate the influence of specific ‘unlearning targets’ and remove associated model capabilities while preserving model performance for non-targets? 
And we can dissect the statement from unlearning targets, influence erasure, unlearning effectiveness, and efficiency perspectives. 

For a LLM unlearning model design, it alway based on the following mathematical modeling:

Forget Set (D_f): This set contains data that the LLM should forget, such as harmful or toxic prompt-response pairs, or a subset of training data related to what is intended to be unlearned. The forget set can be created from the original training data, synthesized data, or data reverse-engineered from the LLM. It's designed to help the LLM unlearn not just specific instances but broader concepts, affecting related samples with common characteristics.
Retain Set (D_r): This set is comprised of data that the LLM should retain to preserve its functionality. It includes samples that are not meant to be unlearned and serves to keep the useful knowledge intact within the model, outside the scope of unlearning.

There are two different types of unlearning methods: model based and input based. And an summary table as is shown below:


Model-based methods: These methods involve modifying the weights or architecture components of large language models (LLMs) to facilitate unlearning.
Input-based methods: This approach entails designing input instructions, such as in-context examples or prompts, to guide the original LLM (without updating its parameters) toward the unlearning goal. 

## Key Findings

 # Knowledge Unlearning by Gradient Ascent [87]

For the experiments, we use the GPT-Neo (125M, 1.3B, 2.7B) LMs initially pretrained on all of the Pile corpora (825GB), and the OPT (125M, 1.3B, 2.7B) LMs, pretrained on a subset of the deduplicated version of the Pile as well as other corpora from different domains. We perform unlearning the GPT-Neo LMs and quantify the privacy risks of the target data compared to the OPT LMs to measure how effective our proposed approach is in contrast to deduplicating the training corpora before pretraining the underlying LM. We also consider the Differential Privacy Decoding (DPD) as one of the baselines.

The table below shows the main results of performing unlearning on LMs of varying sizes and the baselines. 
We highlight five main observations regarding the results.

OPT LMs show a much lower EL and MA than GPT-Neo LMs, confirming that deduplicating the pretraining corpora is indeed helpful for mitigating privacy risks.

Differential Privacy Decoding (DPD+) enables effective protection against extraction attacks demonstrated via the lowest EL and MA score; however, it brings severe degradation of generation capabilities.

Performing unlearning on GPT-Neo until target sequences match the forgetting criteria (UL+) results in severe degradation of both classification and dialogue tasks for the 125M, only severe degradation of dialogue tasks for 1.3B LM while for the 2.7B LMs, it enables retaining most of its previous capabilities.

While the LMs scale to larger sizes, it takes fewer epochs for the target sequences to be forgotten. Together with result (3), this implies that larger LMs are strong unlearners.

While UL+ provides stronger privacy protection than OPT without sacrificing its performance from NEO for the 2.7B LM, it is much more computationally efficient (3,500,000x) than re-training the underlying LM, which is required for all data preprocessing approaches.

Overall, results show unlearning to be an effective approach to providing strong privacy protection while retaining and sometimes even improving general LM capabilities.

# Tofu[89]
Crucially, in each of the leftmost plots of Figures 5 and 6, where the forget set comprises 1% of the data, we observe nearly vertical trajectories. Put differently, unlearning from very small segments of the training data may be more manageable. The forget quality metrics indicate that overall, the unlearned model's performance is lacking—the model that has undergone unlearning is still readily discernible from a model solely trained on the retain set. Recall that forget quality is measured by a p-value and the common significance threshold of 0.05 is higher than almost every model we test. With larger forget sets, models that attain high forget quality often become impractical due to the inherent privacy-utility trade-off. Even with continued unlearning for additional epochs, the issue persists, rendering these models unusable.

# WMDP Benchmark
The paper introduces the Weapons of Mass Destruction Proxy (WMDP) Benchmark, a dataset comprised of 3,668 multiple-choice questions designed to evaluate the potential misuse of large language models (LLMs) in fields such as biosecurity, cybersecurity, and chemical security. This benchmark aims to fill the gap in public evaluations for hazardous capabilities of LLMs, which are currently limited by the private and restricted nature of existing assessments.

The WMDP Benchmark serves dual purposes: firstly, as a tool to measure hazardous knowledge in LLMs, and secondly, as a standard for testing "unlearning" methods—techniques developed to eliminate dangerous knowledge from models while retaining their utility in benign applications. A notable contribution of the benchmark is the introduction of the Representation Misdirection for Unlearning (RMU) method. This approach significantly reduces the models' ability to recall hazardous information by manipulating their activations, thereby mitigating potential misuse while preserving their effectiveness in non-hazardous contexts.

By making the WMDP Benchmark publicly accessible, the developers encourage broader participation in refining these methods and advancing AI safety. This open approach aims to facilitate a more comprehensive understanding of the risks associated with AI technologies and to develop more effective safeguards against their malicious use, particularly in areas that could impact national and global security.

# Rethinking Machine Unlearning [88]
Due to this is a survey paper, It might only focus on gathering som findings of LLM unlearning, and here are the LLM unlearning applications.


Legal and Privacy Concerns: LLM Unlearning is positioned as a solution to legal issues like algorithmic disgorgement, where models trained on data without consent must be destroyed. It offers an alternative to complete destruction by removing only the illegally obtained data.


Sociotechnical Harm Reduction: LLM Unlearning is proposed for aligning LLMs with human values, reducing the generation of harmful or biased content.


Mitigating Misinformation: It can help in reducing the production of false or inaccurate content (hallucinations) by targeting the unlearning of incorrect data points.


Bias and Fairness: LLM Unlearning is also shown to mitigate biases and promote fairness in decision-making, with potential for broader applications.


Security Applications: LLM Unlearning as a defense, it can against security threats such as jailbreaking and poisoning attacks, building on its success in other domains.


## Critical Analysis

# Knowledge Unlearning by Gradient Ascent [87]
This method, however, does have its limitations:

The Forgetting Threshold is dependent on which data samples are chosen for computing the threshold.

We could not directly compare our approach with a Differential Privacy (DP) approach because there are no open-source LMs pretrained with a DP algorithm. We could not replicate the pretrainig phase because of the heavy computational resources needed to pretrain an LM with DP which is estimated to require thousands of GPU hours.

A recent work has suggested that machine unlearning (for the vision domain) can bring negative effects harming the privacy of other users. Future work should explore this phenomenon in the setting of performing unlearning on large LMs as well.

Gradient Ascent alone can be sensitive to the choice of hyperparameters during optimization, such as the number of ascent steps and the learning rate.

# TOFU
Dataset Construction: While TOFU claims to offer a dataset of diverse synthetic author profiles, it's essential to scrutinize the methodology behind generating these profiles. Ensuring true diversity and representativeness is crucial for the effectiveness and fairness of the benchmark.
Unlearning Efficacy Metrics: The suite of metrics compiled by TOFU for evaluating unlearning efficacy is commendable. However, it's essential to assess the robustness and sensitivity of these metrics to different unlearning algorithms and scenarios.
Baseline Results: The fact that none of the existing unlearning algorithms show effective unlearning according to the baseline results is noteworthy. This highlights the need for further research and development in this area. However, it's crucial to consider the choice of algorithms, hyperparameters, and other experimental settings in interpreting these results accurately.
Real-world Applicability: Understanding the potential real-world applications and implications of fictitious unlearning is crucial. While TOFU aims to address legal and ethical concerns related to sensitive data memorization by large language models, it's essential to examine how these techniques can be practically implemented and deployed in various contexts.

# WMDP Benchmark [90]
The paper introduces the Weapons of Mass Destruction Proxy (WMDP) Benchmark, which aims to measure and potentially reduce hazardous knowledge in large language models (LLMs) to mitigate risks associated with their malicious use. This benchmark is designed to fill a gap in current evaluations by offering a public, structured dataset of multiple-choice questions that cover various domains such as biosecurity, cybersecurity, and chemical security. While the benchmark provides a novel approach to "unlearning" hazardous content without undermining the general utility of LLMs, it also presents several limitations.
One significant limitation is the narrow range of hazardous scenarios it covers. The benchmark may not capture all potential malicious uses of LLMs, as it is constrained to the scenarios envisioned and encoded by its creators. This could leave out emerging threats that are not yet recognized or understood. Additionally, there is a challenge in ensuring that unlearning specific hazardous knowledge does not degrade the model’s performance on benign but related content. The paper proposes a method to unlearn targeted information while retaining general capabilities, but achieving this balance can be complex and is not guaranteed.
The effectiveness of the benchmark also heavily relies on the representativeness and quality of the questions in the WMDP dataset. If the dataset does not adequately cover the knowledge areas it aims to measure, the benchmark might not effectively reduce risk. Moreover, there is a risk that the dataset could be exploited if malicious actors find ways to reverse-engineer or bypass the unlearning methods, despite efforts to prevent misuse by filtering sensitive and controlled information.
Finally, the decision to make the benchmark publicly available is a double-edged sword. It allows for broader scrutiny and improvement by the research community, but it also makes it accessible to potential adversaries who might use it to find vulnerabilities in existing LLMs. These limitations highlight the challenges in developing and deploying benchmarks for unlearning hazardous knowledge in AI systems and underscore the need for ongoing research and updates to the benchmark to keep pace with the evolving landscape of AI capabilities and threats.

# Rethinking Machine Unlearning [88]

Generality: Solutions must be adaptable across different models and scenarios, addressing a variety of unlearning targets and datasets.
Authenticity: The process should convincingly eliminate data influence and model capabilities, with a strong emphasis on evaluation, particularly against adversarial attempts.
Precision: The scope of unlearning needs to be clearly delineated to ensure targeted forgetting while preserving the model's general language capabilities.


 
