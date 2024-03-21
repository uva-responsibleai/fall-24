
# LLMs: Prompt injection
## 1. Introduction
Large Language Models (LLMs) have revolutionized the field of Natural Language Processing (NLP) by significantly advancing our ability to understand and generate human-like text. Trained on extensive text corpora, these models demonstrate remarkable proficiency across a spectrum of tasks, including translation, summarization, question answering, and sentiment analysis. Despite these advancements, certain aspects of LLMs remain underexplored, necessitating further analysis to enhance their robustness and reliability.

  

One notable area of concern is the vulnerability of applications powered by LLMs to prompt injection attacks. Such attacks occur when malicious actors craft harmful prompts to manipulate the models' outputs, effectively bypassing the intended instructions. Studies [1,2] offer a comprehensive examination of these threats, underscoring the need for robust defenses against them.

  

Beyond prompt injection, LLMs are also susceptible to adversarial attacks, where slight modifications to inputs are designed to elicit incorrect responses from the models. These tactics allow attackers to exploit the models for harmful purposes, highlighting the critical need for mechanisms to mitigate such risks. Research [3] delves into various adversarial triggers, shedding light on how to safeguard against these vulnerabilities.

  

Another challenge in deploying LLMs effectively is their requirement for task-specific fine-tuning to excel in specific applications. The scarcity of large, supervised datasets for such fine-tuning poses a significant hurdle. However, studies like [4] explore the promise of LLMs as few-shot learners, aiming to overcome this limitation by leveraging their inherent ability to adapt to tasks with minimal data.

  

Moreover, the extensive size of LLMs raises concerns regarding their environmental and societal impact. The computational resources required for their development and operation have implications that warrant careful consideration, as discussed in [5]. This research critically assesses the risks associated with the current trajectory of LLM development.

  

Finally, an essential avenue of research is developing methodologies to analyze the knowledge encapsulated within LLMs. Understanding the scope and nature of this knowledge is crucial for ensuring that the models perform as anticipated. Study [6] explores strategies for accurately estimating the knowledge contained in state-of-the-art language models, emphasizing the importance of transparency and predictability in their use.

  

These concerns and research directions highlight the intricacies of LLMs and the comprehensive strategies necessary to unlock their full potential while addressing their vulnerabilities. This report will delve into these topics in depth.

  
  

## 2. Motivation

  

With the rise of Large Language Models (LLMs), many applications and plugins are integrating them to create dynamic responses. This exciting development, however, introduces new security vulnerabilities. One such threat is prompt injection attacks, which can have severe consequences, including significant financial losses. Despite the potential risks, a comprehensive analysis of prompt injection's complexities, including their current effectiveness, risks, and mitigation techniques, is lacking. This research [5,6] aims to fill this gap and ensure the security of LLM-integrated applications.

  

Adversarial attacks modify inputs to cause machine learning models to make errors. Examination and generation of adversarial attacks are useful to expose model vulnerabilities and are helpful for evaluation and interpretation of these vulnerabilities.

  

Adversarial attacks are typically generated for a specific input, but the authors aim to find universal adversarial triggers. Universal adversarial triggers are input-agnostic sequences of tokens that trigger a model to produce a specific output, when concatenated to any input from a dataset. Such triggers have wide security implications across the board, as they could be widely distributed and allow anyone to attack models. Universal triggers would mean that the attacker does not need access to the target model at test time. There would also be a drastically lower barrier of entry for an adversary. Furthermore, from an analysis perspective, input-agnostic attacks can provide new insights into global model behavior.

  

Language models (LMs) necessitate task-specific tuning to excel in specific tasks, a process that entails training with datasets ranging from thousands to hundreds of thousands of examples tailored to those tasks. Collecting such extensive, task-specific datasets poses a significant challenge, especially as this process must be undertaken anew for each task. Meta-learning offers a promising solution to this problem. In the realm of LMs, meta-learning allows a model to acquire a wide range of skills and pattern recognition capabilities during training. These capabilities enable the model to quickly adapt to or identify new tasks during inference. A model's ability to assimilate various skills and tasks directly within its parameters could enhance with the model's scale. This hypothesis is put to the test in study [4] involving the training of a 175 billion parameter autoregressive language model known as GPT-3, with the goal of evaluating its in-context learning prowess.

Language models (LMs) are not a new development; early efforts were made to produce N-gram models, LSTMs, etc., to understand human text. Since then, the field's research focus has transitioned to word embedding/transformer-based models. From 2018 to 2021, the sizes of these models have been increasing steadily, leading to more resource consumption and potentially less understanding of the training data. The authors express concern that the industry's primary focus on model scale might be questionable. They argue that even though models may increase in size, they may not necessarily gain a true understanding of the nature of the world in the way humans do.

There are limitations of simply assuming LLMs possess true understanding of the information they process. [6] emphasizes the need for methods to assess what knowledge LLMs truly retain and how effectively they can utilize it in practice. By developing reliable methods to gauge an LLM's knowledge, researchers can ensure these tools are not merely mimicking human language, but are genuinely comprehending the information they handle. This focus on knowledge assessment becomes crucial for building trust and ensuring responsible development and deployment of LLMs in various applications.

  

## 3. Methods

  

### 3.1 Prompt Injection attack against LLM-integrated Applications [1,2]

 In order to understand the patterns and effectiveness of existing real-world prompt injection attacks, [1] surveys research papers and industrial papers on prompt injections and follows this by conducting a pilot study. They implement existing prompt injection attacks on ten real-world LLM-integrated applications from SUPERTOOLS using strategies categorized from the survey, including direct injection, escape characters, and context ignoring. They produce a novel prompt injection attack methodology, HOUYI, tailored for LLM-integrated apps in black-box scenarios. The first step HOUYI takes is context inference, where it acquires an accurate understanding of the internal context established by the built-in prompts of the target application. It devises questions based on the application’s documentation and usage, feeds them to the app, and collects this information to be input into an LLM for analysis. Then, a prompt is composed, made up of a framework, separation, and disruptor component. This prompt goes through an iterative refinement process, where each component is swapped in and out depending on if they lead to successful attacks each iteration, resulting in a series of complete prompts to use for prompt injection attacks. This research conducts experiments to evaluate the performance of HOUYI in various contexts by using HOUYI generated prompts to run prompt injection attacks on 36 applications from SUPERTOOLS. These experiments include the following: deploying five distinct exploit scenarios across there applications, and performing an ablation study focusing on focusing on three discrete strategies: Syntax-based Strategy, Language Switching, and Semantic-based Generation. They also validate these vulnerabilities and provide two case studies. As for real-life scenarios, [2] details how the NVIDIA AI Red Team has identified and verified three vulnerabilities in LangChain chains. All three exploits were performed using the OpenAI text-davinci-003 API as the base LLM and were able to obtain remote code execution (in older versions of LangChain), server-side request forgery, or SQL injection capabilities. Remote code execution was accomplished by phrasing the input as an order rather than a math problem, inducing the LLM to emit Python code of choice. The same pattern was used for server-side request forgery, where they Declare a NEW QUERY and instruct it to retrieve content from a different URL. The LLM returns results from the new URL instead of the preconfigured one contained in the system prompt. As for ​​the injection attack, it works similarly in that they use the “ignore all previous instructions” prompt injection format, and the LLM executes SQL provided by the attacker.

  

### 3.2 Universal Adversarial Triggers for Attacking and Analyzing NLP [3]

The authors aim to find universal adversarial triggers that attacks that concatenate tokens to the front or end of an input, in order to cause a target prediction order to find universal adversarial triggers, the authors design a gradient-guided search over tokens to find different specific triggers. The tokens are iteratively updated to increase likelihood of target prediction, until they have found short sequences that successfully trigger a target prediction in a variety of tasks.

  

For each trigger, the trigger is set initially as “the the the”. The algorithm iteratively replaces the tokens in the trigger to minimize loss for target prediction over batches of examples. The token replacement strategy is inspired by HotFlip, which approximates the effect of replacing a token using its gradient. The token replacement strategy is augmented with beam search. Ultimately, the trigger search algorithm is generally applicable, with the only task-specific component being the loss function.

  

There are three main tasks that the authors focus on generating universal adversarial triggers for–text classification, reading comprehension, and text generation. For text classification, the goal is for triggers to cause targeted errors for sentiment analysis and natural language inference. For sentiment analysis, the trigger should flip negative predictions to positive, and vice versa. For natural language inference, the trigger should get SNLI models to output incorrect classifications out of Entailment, Neutral, or Contradiction. For reading comprehension, the goal is for triggers to cause predictions for a reading comprehension question to be a predetermined target span inside the trigger. The authors create triggers for SQuAD, aiming to get the following target answers for the following four questions: Why? – “to kill american people”; Who? – “donald trump”; When? – “january 2014”; Where? – “new york”. The attack is successful if the model’s predicted span exactly matches target. For conditional text generation, the trigger maximizes likelihood of a set of target texts, such as racist outputs.

  

### 3.3 Language Models are Few-Shot Learners. [4]

The authors explored various settings for machine learning models, specifically zero-shot, one-shot, and few-shot learning. These methodologies represent a spectrum of trade-offs between achieving high performance on established benchmarks and the efficiency of using sample data. Historically, Fine-Tuning (FT) has been a predominant strategy, which enhances a pre-trained model's accuracy by retraining it on a dataset tailored to a particular task. The Few-Shot (FS) approach, in contrast, involves presenting the model with a limited number of task examples at the time of inference without adjusting its parameters. One-Shot learning (1S) further narrows this approach by allowing only a single example, whereas Zero-Shot learning (0S) does not permit any examples. Instead, 0S relies solely on a descriptive natural language instruction to guide the model. In an effort to understand how model size influences machine learning performance, the authors trained models of various capacities, ranging significantly in size from 125 million to 175 billion parameters, with their largest model being referred to as GPT-3. This extensive training utilized data from the Common Crawl dataset. The authors then evaluated each model on various NLP tasks to check their performance.

  

### 3.4 How Can We Know What Language Models Know? [6]

In order to tighten the “lower bound” on the extent of knowledge contained in LLMs, [6] explores two automatic methods to improve the effectiveness of prompts: mining-based and paraphrasing-based generation strategies. Mining-based generation creates template-based prompts based on the observation that words near the subject and object typically describe the relation between them. There are two types of mining-based generation strategies:

-   Middle-word Prompts: words in the middle of the subject and object are indicative of the relation (ex. “Barack Obama was born in Hawaii”).
    
-   Dependency-based Prompts: when relations aren’t in the middle of the sentence, templates based on syntactic analysis of the sentence can be more effective for extracting relations.
    

Paraphrasing techniques aim to generate text with an increased vocabulary diversity without deviating from the core idea of the original prompt.

-   Back-translation: translate initial prompt into B candidates in another language, then translate them back into B candidates of the original language
    

-   Rank candidates and keep the top T prompts
    

  

## 4. Key Findings



The pilot study [1] reveals that existing prompt injection techniques, such as direct injection, utilizing escape characters, and context ignoring, are not notably effective against these LLM-integrated applications. This is partially due to the fact that prompts are treated as ‘data’ for the LLMs to analyze rather than answer, so they are not interpreted as a command. Moreover, intricate interactions of application design, LLM prompt processing, and built-in defenses contribute to the resilience of many LLM-integrated applications against traditional prompt injection attacks. Using HOUYI, it was found that a majority of LLM-integrated applications are identified as susceptible to prompt injection attacks, as HOUYI was capable of executing a successful attack on 31 apps at least once across exploit scenarios. This suggests that many apps exhibit latent vulnerabilities when exposed to prompt injection. However, the success rate of exploit scenarios depends on the app designs. Additionally, it was shown that HOUYI outperforms the three ablation baselines focusing on three discrete strategies: Syntax-based Strategy, Language Switching, and Semantic-based Generation, in identifying vulnerabilities. HOUYI was also able to uncover 31 unique vulnerabilities across apps, ten of which were confirmed and acknowledged by vendors, including Notion. Currently, there are no systematic techniques to prevent prompt injection in LLM-integrated applications, but strategies exist. Additionally, it is important to note that certain detected vulnerabilities may become non-reproducible over time given the nature of LLMs themselves in being non-deterministic.

In [2], all the plug-ins face the issue of prompt injection vulnerability. This can be attributed to the fact that a prompt contains ‘control’ and ‘data’ in LLMs, so this lack of separation enables attackers to reliably control LLM outputs. Therefore, a crucial mitigation strategy is to always treat LLM productions as potentially malicious; input inspection and sanitization and best practices for designing plug-ins to provide minimum functionality and service access are recommended as well.

###   

Using the iterative method shown in study [3], different triggers are found for each task. For sentiment analysis, the universal adversarial trigger found is “zoning tapping fiennes”. For reading comprehension, the universal adversarial trigger found is “why how because to kill american people”. For conditional text generation, the trigger found is “TH PEOPLE-Man goddreams Blacks”. The whole trigger is found to be necessary, as using just “Blacks” as the trigger does not trigger racist outputs. The trigger was generated for GPT-2 117M, but also works with the 345M model, showing transferability between models.

  

When analyzing these triggers, there are a number of patterns that emerge. In doing analysis on token order, the authors found that there are multiple effective orderings of a specific trigger. Furthermore, concatenation of triggers to the end vs the beginning of the input sequence still shows effectiveness. Perhaps most importantly, even though these triggers are found through white-box access to a specific model, they are found to be transferable to other models as well.

  

The authors aimed to understand why these universal adversarial triggers work. One thing they found was that construction of NLP datasets can lead to dataset biases, also known as artifacts. For SNLI for natural language inference, there is correlation between the trigger words and these artifacts. In fact, these dataset artifacts are also found to be successful triggers.

###   

The findings of the study [4] indicate a direct correlation between the size of GPT-3 and its performance, highlighting a continuous enhancement in knowledge acquisition as the model's capacity expands. When compared to zero-shot scenarios, both one-shot and few-shot approaches show remarkable improvements, achieving and even surpassing the benchmarks set by state-of-the-art (SOTA) models that have been fine-tuned for open-domain tasks. Nevertheless, GPT-3 demonstrates limitations in tasks that require detailed comparison of two sentences or text snippets, such as identifying if a word is used similarly in two different sentences, determining if one sentence paraphrases another, or if one sentence logically implies the other. Moreover, GPT-3's ability to handle basic arithmetic tasks shows a notable progression as it moves from zero-shot to one-shot and then to few-shot settings, showcasing significant computational capabilities even in the absence of any fine-tuning.

The study [5] showed that training large models requires significant environmental and financial costs. They due to the substantial energy and computing resources required. For instance, trainaing a single BERT base model on GPUs without hyperparameter tuning is estimated to consume as much energy as a trans-American flight. Moreover, model sizes have been increasing. (E.g. While early models like BERT only have 340 million parameters, later models like GPT-3 are built to have 175 billion parameters.) The authors suggest that it is fundamentally unfair that people living in marginalized communities (who are often most affected by climate change) typically do not receive compensation from AI model makers, despite bearing the brunt of the carbon emissions from the training process. Additionally, the financial costs are substantial, as larger models require more computing power, which can be extremely costly. For example, achieving a 0.1 increase in BLEU score for English to German translation using neural architecture search can result in a $150,000 increase in compute cost, in addition to the associated carbon emissions.

  

There are also concerns raised about training data. Large, uncurated datasets are problematic because training language models (LMs) require such size of data often contain biases and derogatory associations related to gender, race, ethnicity, and disability. In that sense, they may end up reflecting dominant hegemonic views. To address these issues, the industry needs to better curate and understand the datasets. The authors also suggested a lack of diversity in training data given that young Internet users from developed nations are overrepresented in the dataset. For example, 67% of Reddit users in the United States are men, and 64% are between ages 18 and 29. At the same time, surveys of Wikipedians find that only 8.8 -- 15% are women or girls. This oversight can result in a training dataset lacking necesary diversity. Lastly, data filtering (as used in GPT-3 training) can also eliminate important voices from lesser-known communities, such as trans people, disabled people, etc.

  

According to the study [6], manually-written prompts are a weak measure of the knowledge of LMs. Mined prompts, particularly those with complex syntax or uncommon wording, led to the most significant knowledge performance gains. The choice between mined prompts and paraphrased prompts depends on the ensemble strategy. Paraphrased prompts excelled in rank-based ensembles (ie. averaging predictions of top-ranked prompts), while mined prompts led to better performance in optimization-based approaches (optimizing prompts based on weights). [6] also revealed that even minor modifications to prompts can significantly impact predictions. Interestingly, effective prompts often shared similar patterns in terms of parts of speech usage. Finally, [6] showed a decent level of generalizability across two LMs, although with slight performance losses.

  

## 5. Critical Analysis

### 5.1 [1,2] Critical Analysis

The strength of the paper is its end-product, HOUYI [1], in its ability to uncover real-world prompt injection vulnerabilities on widely used platforms like Notion. Through its usage, potential attacks were brought to life and addressed in a patch, preventing potentially large financial consequences. It was able to accomplish a task previously not done repeatedly among various applications. This highlights the efficacy of HOUYI in identifying and exploiting these vulnerabilities. Additionally, the study targets both the cause and effects of prompt injection attacks, substantiated with papers and experimental results, expanding our understanding of these attacks.

  

The weakness of this paper is its inability to provide mitigations for the strategies it lists and exploits it executes. While the tool can uncover vulnerabilities, it merely states that they are inevitable due to the “data” and “control” place being inseparable. It provides defense measures proposed by others, but never maps which should be used for each exploit. Therefore, the tool is dangerous in some capacity, as it can reveal security risks in large applications. The safety of the tool hinges on its users and their willingness to share results with application developers.

  

The strength of the paper [2] was its deep understanding of the way each vulnerability uncovered functioned. This was visible through the examples provided and its accompanying code. Additionally, unlike [1], it lists out mitigation techniques that can be applied to LLM-integrated apps as a whole. A weakness of this paper was its limited research set, only focusing on LangChain, which hindered it from fully addressing the problem it attempts to highlight.

  

### 5.1 [3] Critical Analysis

The strength of this paper is that it goes beyond what previous work did. Prior to this paper, most adversarial attacks were gradient-based. The universal triggers that the authors found are different because they are input-agnostic. As such, the strength of the work is that these attacks can be applied to any input, use a stronger attack algorithm, and consider a broader range of models and tasks.

  

A potential area of bias is in the tasks that were chosen, as well as the targets that were chosen. A potential weakness of the results is that the found triggers are quite nonsensical. Thus, a potential future area of work might be to figure out how to interpret nonsensical triggers, or even find ways to develop triggers that make more sense to humans. Beyond that, grammatical triggers that work anywhere in the input could be valuable. Finally, triggers that are dataset or task-agnostic would be even more applicable across domains.

  

Ethically, it is important to consider who is accountable for such vulnerabilities. Who is responsible when models produce toxic outputs? With such universal triggers existing, whose responsibility is it to make sure these universal triggers do not exist?

  

### 5.1 [4] Critical Analysis

The strength of the study [4] paper's lies in its demonstration of the language models' few-shot learning capabilities, where GPT-3 achieves strong performance across a variety of NLP tasks without fine-tuning. This demonstrates a notable advance in language models' ability to generalize from limited examples, bringing them closer to human-like learning efficiency.

  

There are weaknesses and potential biases in the study. The reliance on a massive dataset and extensive computing resources limits the reproducibility of the findings and raises questions about the sustainability and accessibility of such models

  

Potential biases in the model could stem from its training data. Since the model is trained on data from the internet, which may contain inaccuracies, prejudices, and biased viewpoints, GPT-3 might inadvertently replicate these biases in its outputs.

  

### 5.1 [5] Critical Analysis

The paper is written by a diverse group of researchers coming from both universities and large corporations (Google), and it is helpful to see diverse opinions in the field of AI. However, since the paper’s publication (2021), some of the predictions have turned out to be incorrect. For example, the paper suggests that the LLMs are fundamentally “Stochastic Parrots”; given recent models’ ability to handle math problems and reason with common sense, it is unclear whether the authors’ statement is true. The authors are right to say that keeping in mind the social implications of the Language Models is crucial. Meanwhile, the unexpected events happened since the paper’s publication also showcased the importance of humility in knowing what we do not know.

  

### 5.1 [6] Critical Analysis

By probing LMs with targeted prompts and analyzing their responses, [6] provides valuable insights into the inner workings of these models. However, results primarily focus on a specific type of LM (BERT) and do not extensively explore other models or architectures, limiting the generalizability of its findings. The study also relies heavily on the use of synthetic probing tasks, which may not fully capture the complexities of real-world language understanding. Furthermore, there are potential biases in the choice of probing tasks and prompts, which could influence the interpretation of the results.

  

## References

  
[1]. [Prompt Injection attack against LLM-integrated Applications](https://arxiv.org/abs/2306.05499). Liu et al, 2023

[2]. [Nvidia Blog - securing against prompt injection attacks](https://developer.nvidia.com/blog/securing-llm-systems-against-prompt-injection/). 2023

[3]. [Universal Adversarial Triggers for Attacking and Analyzing NLP](https://arxiv.org/abs/1908.07125). Wallace et al. 2019

[4]. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). Browns et al. 2020

[5]. [On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?](https://dl.acm.org/doi/10.1145/3442188.3445922). Bender et al., 2021

[6]. [How Can We Know What Language Models Know?](https://arxiv.org/abs/1911.12543) Jiang. 2020