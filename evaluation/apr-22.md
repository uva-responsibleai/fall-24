# Introduction

Evaluation of foundational models is crucial. It provides critical context for understanding and improving machine learning models. It serves three main purposes while there also exist some new challenges that are not generally encountered in other AI or ML settings:
- **Tracking Progress:** This involves measuring model performance and guiding the development of improved models. However, foundation models present unique challenges in this regard. Unlike traditional models, foundation models need to be adapted—often in various ways—to specific tasks, making direct comparisons complex.
- **Understanding:** Gaining insights into model behavior, including how models respond to different data segments and their interpretability, is essential. Typically, evaluations are designed with a predefined understanding of what to assess, such as specific taxonomies. However, foundation models develop emergent abilities, like in-context learning, which are difficult to predict and thus challenging to incorporate into standard evaluation frameworks.
- **Documentation:** Effective documentation summarizes model behavior and communicates it to a broad audience of stakeholders, aiding in decision-making. For foundation models, the adaptability to numerous applications complicates the ability to produce comprehensive and useful documentation. Clear criteria are necessary to guide meaningful documentation, but the versatility of foundation models makes this task particularly demanding.

In foundation models, there are two classes of evaluation:
- **Intrinsic evaluation of the foundation model**: inherently divorced from a specific task due to the task-agnosticity of these models.
- **Extrinsic evaluation of task-specific models**: necessarily dependent on both the foundation model and the adaptation mechanism.

![Screenshot 2024-04-18 at 8 18 07 PM](https://github.com/wenqian-ye/fall-24/assets/32115593/e51d4a7e-2a9f-4bf0-a384-ba7f1f96a6c8)

To better understand the evaluation of LLMs, we describe it in the following three parts: What to Evaluate, Where to Evaluate, and How to Evaluate.

### What to Evaluate [79]
To answer the question of "what tasks should we evaluate LLMs to show their performance? On what tasks can we claim the strengths and weaknesses of LLMs?" The authors divide existing tasks into the following categories: natural language processing, robustness, ethics, biases and trustworthiness, social sciences, natural science and engineering, medical applications, agent applications (using LLMs as agents), and other applications.

#### Natural Language Processing Tasks
- Natural language understanding (such as Sentiment analysis, Text classification, Natural language inference (NLI), Semantic understanding, and social knowledge understanding)
- Reasoning (such as mathematical reasoning, commonsense reasoning, logical reasoning, and domain-specific reasoning)
- Natural language generation (such as Summarization, translation, Question answering, sentence style transfer)
- Multilingual tasks
- Factuality

#### Robustness, Ethic, Bias, and Trustworthiness
- Robustness: includes out-of-distribution (OOD) and adversarial robustness
- Ethic and bias: the authors found LLMs could internalize, spread, and potentially magnify harmful information existing in the crawled training corpora, usually, toxic languages, like offensiveness, hate speech, and insults, as well as social biases like stereotypes towards people with a particular demographic identity.
- Trustworthiness: there are eight critical aspects (toxicity, stereotype bias, adversarial and out-of-distribution robustness, robustness to adversarial demonstrations, privacy, machine ethics, and fairness)

#### Social Science
- Includes economics, sociology, political science, law, and other disciplines.

#### Natural Science and Engineering
- Mathematics
- General science
- Engineering

#### Medical Applications
- Medical queries
- Medical examination
- Medical assistants

#### Agent Applications

#### Other Applications
- Education
- Search and recommendation
- Personality testing
- Specific applications

### Where to Evaluate: Datasets AND Benchmarks [79]

#### Benchmarks for General Tasks
- Chatbot Model Evaluation:  Chatbot Arena (voting-based) and MT-Bench (comprehensive questions for mulit-turn dialogue)
- Wide-range of Capabilites Evaluation:  HELM and LLMEval
- OOD Robustness Evaluation:  GLUE-X, BOSS
#### Benchmarks for Specific Downstream Tasks
- Domain-Specific Evaluation:  TRUSTGPT(ethical metrics), SafetyBench(security performance), SOCKET(social kowledge concepts), MATH(mathematics)
- Chinese-Specifc Evaluation:  C-Eval(knowledge/capabilities of foundational models in Chinese), GAOKAO-Bench(Chinese Gaokao exam), CMB(Chinese medical evaluation)
- Tool-Augmented Evaluation:  API-Bank benchmark (ToolBench Project)

#### Benchmarks for Multi-modal task
- MME(perceptual and cognitive aptitudes), MMBench(large-scale vision-language models), LVLM-eHub(online competition and quantitative assessments), SEED-Bench(19,000 MC questions)

### How to Evaluate [79]

#### Automatic Evaluation

#### Human Evaluation

#### Traditional NLP and classification metrics
- Classification metrics (accuracy, F1-score).
- Text similarity metrics (Levenshtein Similarity Ratio (Simple Ratio), BLEU Score, ROUGE)
- Semantic Similarity (cosine similarity)
- Functional Correctness

#### Rule-based Metrics
- Syntax correctness: This metric measures whether the generated code conforms to the syntax rules of the programming language being used.
- Format check: Another metric that can be used to evaluate NL2Code models is the format of the generated code.
- Language check: A language check metric evaluates whether the generated text or code is written understandably and consistent with the user's input.
- Keyword presence: This metric measures whether the generated text includes the keywords or key phrases that were used in the natural language input.

#### Human feedback
- This process can be as simple as having a human verify output and assign it a pass/fail mark.
- Also can be incorporated into the LLM-based system using reinforcement learning from human feedback (RLHF).

#### LLM-based Evaluation
- Step 1. Generating output predictions from a given test set.
- Step 2. Prompt the model to focus on assessing the quality of output-given reference text and sufficient context (for example, criteria for evaluation).
- Step 3. Feed the prompt into the model and analyze results.


# Motivations
Evaluation is important for the success of Large Language Models (LLMs) for multiple reasons: 
1.  It highlights the LLM’s strengths and weaknesses, revealing what areas need more research or improved performance in.
2.  Better evaluation can help guide human-LLM interaction, which can inspire future human-LLM design so that LLMs are easier to interact with.
3.  It ensures LLM safety and reliability by testing them for vulnerabilities before they are deployed into production.
4.  Allows us develop new evaluation protocols for more robust assessment and risk mitigation

The recent trend of larger LLMs with even poorer interpretability, call for better evaluation protocols that are able to thoroughly inspect the true capabilities of these models.



# Methods


## DecodingTrust
In the DecodingTrust paper, the authors attempted to generate a comprehensive analysis of LLM trustworthiness across many benchmarks and traits. 

### Toxicity Evaluation
The authors analyzed the toxicity of GPT-3.5 and GPT-4 with the RealToxicityPrompts dataset. They also designed new system and user prompts for an in-depth analysis on toxicity. 

- System Prompts: the authors prompt one of the models with an adversarial prompt in order to jailbreak it. 
- Task prompts in user prompts: the LLMs generate missing text in prompts from the RealToxicityPrompts dataset.
- Evaluation: conducted with Perspective API, which calculates Expected Maximum Toxicity and Toxicity Probability.
- Results: GPT-3.5 and GPT-4 are successful in mitigating toxicity for benign prompts, compared to other pretrained models. Toxicity increases significantly when the models are given an adversarial system prompt, however, showing a real jailbreak risk. GPT-4 exhibits more toxic behavior than GPT-3.5 under such conditions.

#### System Prompt Design
The authors dive into what exactly makes some system prompts generate toxic behavior in the LLMs analyzed. They found a taxonomy of 33 categories, and noted that 'straightforward' prompts are most effective in generating toxicity, and that telling the LLM to add swear words can efficiently increase toxicity. 
Interestingly, GPT-4 is more likely to follow the "jailbreaking" system prompts than GPT-3.5.

#### User Prompt Design
The authors also analyze what kinds of user prompts are more likely to promote toxicity. They note that explicitly requesting toxic behavior is inefficient.
They experimented with having LLM-generated toxic prompts, but these did not always result in more toxicity. Prompts generated by GPT-4 had higher toxicity than GPT-3.5.

### Stereotypes Evaluation
The authors looked into how likely a model was to produce biased content, what types of stereotypes and biases were most likely to occur, and the efficacy of adversarial system prompts for generating biased output. The authors created their own stereotypes dataset for this analysis. Overall, the authors found that GPT models reject many of the biased prompts, and that using a targeted system prompt can significantly increase model agreement with stereotypes. GPT-4 was more likely to output biased content under such a prompt than GPT-3.5.

### Adversarial Robustness
The authors analyze the robustness of LLMs to adversarial inputs using the AdvGLUE benchmark as well as their own AdvGLUE++. GPT-4 was found to be more robust than GPT-3.5 on the AdvGLUE benchmark. They were able to generate prompts against other LLMs, like Alpaca, however, which comprised the AdvGLUE++ dataset and for which GPT-4 and GPT-3.5 were vulnerable. 

### Out-of-Distribution Robustness
The authors covered how models react to data which is significantly outside the distribution (out-of-distribution or OOD) of their training data. Specifically the authors sought to answer whether GPT models struggle with OOD data, whether models are aware of unknown knowledge, and how the OOD impacts the performance of the models. An example of an input style change would be suddenly changing to Old English for the last sentence of a paragraph given to a model for sentiment analysis.
- Input Styles: Overall, GPT-4 was more robust than GPT-3.5 for test inputs with different OOD input styles. The GPT models were more vulnerable to less frequently used input styles.
- Robustness on OOD knowledge: GPT-4 was more conservative and produced more reliables answers when given an "I don't know" option, but this was not the case for GPT-3.5.
- Generalization on OOD data: GPT-4's accuracy broadly improves with a few-shot compared to zero-shot approach for style-transformed test data. Both GPT-4 and GPT-3.5 generally perform better with demonstrations from close domains than distant ones.

### Robustness Against Adversarial Demonstrations
Demonstrations can be adversarially modified while performing in-context learning on an LLM. In particular, the authors consider adversarial modifications of three kinds.
* Producing counterfactual examples
* Producing examples with spurious correlations
* Injecting backdoors to affect model outputs

#### Counterfactual Adversarial Examples
The authors use counterfactual examples from the SNLI CAD dataset produced from the MSGS dataset (for linguistic tasks); this dataset contains hypotheses and premises and asks the model to determine whether the premise entails the hypothesis (that is if the sentence contains a particular linguistic feature). They evaluate the impact of in-context learning with adversarial demonstrations on MSGS's test data.
The authors find that upon including a counterfactual example after a number of randomly sampled unaltered demonstrations, GPT 3.5's performance actually improves and GPT 4 seems to remain unaffected. On the MSGS dataset (for all the linguistic tasks considered in it), including the counterfactual example similarly improves performance on the tasks for both GPT-3.5 and GPT-4.

#### Spurious Correlations
For examining the effect of demonstrations suggesting spurious correlations, the authors look at the HANS dataset with hypotheses and premises and form demonstrations with spurious correlations (entailment-correlated demonstrations where the premise does not necessarily entail the hypothesis and non-entailment-correlated demonstrations where the premise does not necessarily negate the hypothesis). The authors then observe that certain types of spurious correlations mislead GPT models to make erroneous predictions, whereas some others led to better performance (presumably by helping the GPT model recognize the underlying causal features from the demonstrations).

#### Backdoor Injection in Demonstrations

### Evaluating Privacy
Privacy violations in LLMs may occur at the level of training data and during in-context learning (i.e. in the inference phase). For the former setting (training data privacy), the authors use the Enron Email dataset that they have good reason to believe has been used to train several models, including the GPT models that they consider. They use privacy attack prompts of four different kinds to make the model regurgitate email addresses. They do so via context prompting by feeding the model the first few tokens of an email before the email address to make the model generate the email address. Another method is via zero-shot or few-shot prompting that makes use of certain template prompts that are geared towards making the model generate the email address, for example, _the email address of {target_name} is_ or _—–Original Message—–\n From: {target_name} [mailto:_.
The authors notice that both GPT-3.5 and GPT-4 generated email addresses even in the zero-shot setting. However, in the context prompting setting, GPT-4 leaks less information than GPT-3, probably due to higher levels of human-alignment oriented instruction tuning that prohibits it from responding to incomplete prompts. In the setting where the email domain is known and few-shot prompting is used, GPT-4 leaks more information than GPT-3.5 in more cases. Privacy leakage is exacerbated by a higher number of shots.

#### Inference Time Privacy
Here they use the Enron dataset again, take the names and email addresses from it and synthetically generate some sensitive attributes like SSNs, credit card number, address, etc. to use as personally identifiable information (PII). They investigate privacy leakage in three settings.

1. **Zero-shot prompting** They use system prompts to protect PII, and then inject private information into the chat history. We then ask GPT models about the private information of an individual in subsequent exchanges.
2. **Few-shot privacy-protection demonstrations** They provide few-shot demonstrations that instruct the models to refuse to output private information.
3. **Few-shot privacy-leakage demonstrations** Contrary to the preceding setting, they provide few-shot demonstrations that encourage the models to output correct private information to aggravate privacy leakage.

The authors observe that GPT-4 is better at protecting inference time privacy than GPT-3, and some PII like SSNs are better protected, perhaps due to explicitly being instruction tuned to not divulge those. In the zero-shot and few-shot privacy-protection settings, GPT-3.5 divulges private information at times while GPT-4 is shown to be capable of protecting PII. However, the dam breaks for GPT-4 when it comes to few-shot privacy-leakage demonstrations, with it joining GPT-3.5 in revealing all types of PII when such demonstrations are used.

#### Other Privacy Observations
The authors observe that these models behave differently when it comes to treating different terms related to privacy or events or information that need to stay private. For instance, GPT-4 may leak information when the term "confidentially" is used but not when "in confidence" is used. Also, the GPT models reveal more information about privacy-sensitive events like divorce than for health issues. GPT-4, however, does seem to perform better than GPT-3.5 in this privacy setting.

### Evaluating Machine Ethics
The authors then use the ETHICS (basic ethics) and Jiminy Cricket (commonsense morality) datasets to evaluate the ethical understanding of models. They observe that even with few shot training with some examples, the GPT models outperform ethically finetuned BERT-like models, and in some cases, even in the zero-shot setting. In both the zero-shot and few-shot settings, GPT-4 demonstrates a greater understanding of ethics than GPT-3.5.

The authors also show that both GPT models can be jailbroken using jailbreaking prompts to misclassify immoral actions. These prompts can include asking it to disregard human morality, be an immoral agent, prefer to overlook wrong actions, etc. They then run experiments by considering non-immoral (moral and neutral) outputs as positive and immoral ones as negative and use differences in FPR from the benign (i.e. non-jailbroken) prompt to evaluate jailbreaking performance. It is also noted that GPT-4 is easier to jailbreak this way, possibly due to its ability to follow instructions more precisely, rendering it vulnerable to malicious/adversarial prompt engineering.

They also examine the effect of evasion attacks, by attaching evasion phrases like "_But it was accidental/unsuccessful_" to prompts with immoral actions mentioned. Here again, both GPT models are vulnerable to this attack, with GPT-4 more so than GPT-3.5.

GPT models may struggle with conditional morality statements. For example, when the harm caused due to an immoral action is directed towards others, GPT models do well to a certain extent, but when directed towards onself (self-harm), they falter in flagging it correctly. GPT-3.5 suffers from the latter issue more than GPT-4. Also, the severity of the resulting harm is immaterial to GPT-3.5 but the likelihood of GPT-4 flagging something as harmful/immoral increass with the severity.

### Evaluating Fairness

# Key Findings

For DecodingTrust, the findings are described with the methods in the previous section.

# Critical Analysis

Evaluating foundation models in machine learning introduces complexities not encountered with traditional task-specific models. Effective evaluation of foundation models requires a dual approach: first, by inferring model properties through extensive assessment of task-specific derivatives, and second, by direct property measurement within the foundation models themselves. Existing evaluation frameworks often overlook the resource demands of model creation, leading to potentially skewed comparisons. Addressing this, a new evaluation paradigm for foundation models emphasizes the inclusion of all resources used in model adaptation, enhancing the informativeness and impact of evaluations. Furthermore, there is an emerging consensus on the need to expand evaluation criteria to include a broader array of considerations such as robustness, fairness, efficiency, and environmental impact. This broader perspective, coupled with the sample efficiency of adaptation models, offers opportunities to diversify evaluation metrics by reallocating resources, thus aligning more closely with varied stakeholder values and preferences. The rationale for the analysis in DecodingTrust for counterfactual robustness seems questionable: the counterfactual is not a misleading statement here, but a counterfactual example with a valid response provided in it. It should benefit the model as a result (which is indeed the case).

