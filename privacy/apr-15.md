# Introduction

Privacy in LLMs is a complex topic, with a variety of techniques to both preserve and attack it. As we've seen in the last couple weeks, while techniques exist to mitigate privacy risks for LLMs, they are neither widespread nor foolproof, and may even introduce other problems such as fairness concerns while they are improving privacy. This blog will take a broad overview towards many aspects of Privacy in LLMs, serving as a capstone for this section of the course.  In the following subsection the [Motivations](#motivations) cover some of the basic privacy concerns that arise from LLMs, then the [Methods](#methods) section will review how privacy can be attacked and defended, and the [Key Findings](#key-findings) section will discuss the findings from some of the specific papers being covered. Finally, in the [Critical Analysis](#critical-analysis) Section the current state of privacy in LLMs, the current challenges and shortcomings, and potential future directions for study will all be discussed.

## Background and Motivations
### Memorization
LLMs are capable of memorizing examples from their training sets, which then leaves a vulnerability for an attacker to extract training data, including potentially private information. The authors distinguish between discoverable memorization and extractable memorization, where the former focuses on capturing an upper bound on all training
data and the latter focuses on capturing data that an adversary could recover. Formally, they define extractable memorization as: “Given a model with a generation routine Gen,
an example x from the training set X is extractably memorized if an adversary (without access to X) can construct a prompt p that makes the model produce x (i.e., Gen(p) =
x).” Prior work on extractable memorization has addressed the questions of how to design prompts to best elicit memorization in a model and how to test whether the attack
worked, but rather than improve upon the lower bound of extractability, has shifted too looking at measuring an upper bound on the strength of an extraction attack (discoverable memorization). The authors note the large gap between extractable and discoverable memorization as their motivation for their work, looking to disentangle the
possible explanations of why this gap exists. This area is also touched on in the survey paper [[4]](#ref4), and the diagram below might help clarify (outlines multiple memorization types).

<img src='./images/survey_fig1.PNG' width=60% height=60%>

Furthermore, the survey goes on to test the memorization level based on several factors, generating graphs for the memorization based on the fraction extractible fro varying model sizes, repetitions in training data, and prompt length (figure below).

<img src='./images/survey_fig2.PNG' width=60% height=60%> \\

### Attacks Against Privacy
LLMs are provided with information from different sources in inputs at inference time, and they need to reason about what information to share in the output, for what purpose
and with whom.. In this work, the authors set out to answer the under-explored question “Can LLMs reason about the implications of contextual privacy in interactive settings?”
They center context in reasoning about privacy expectations on existing theory that states that information flow should depend on social context. An example provided in the
text is a healthcare provider giving out your healthcare information to an insurance company (violation of contextual integrity) vs. another healthcare provider who is treating
you (not a violation of contextual integrity). They motivate their work [[2]](#ref2) by stating that social reasoning capabilities are essential for keeping track of different people’s access to information and their relations

This work [[3]](#ref3) focuses on privacy violations through LLM inference, specifically the ability of LLMs to infer personal attributes at inference time. As an example in Figure 1 (below), “there is this nasty intersection on my commute, I always get stuck there waiting for a hook turn”, the LLM could infer the user location based on the colloquial expression “hook turn” that is used in Melbourne. The authors motivate the work by the case of malicious actors scraping a user’s online posts and feeding it to a pretrained LLM, as well as inferring the actual person from unstructured excerpts found on the internet.

<img src='./images/BM_fig1.PNG' width=60% height=60%>



# Methods
## Scalable Extraction of Training Data [[1]](#ref1)
+ They first evaluate past extraction methods on open-source models so they can verify if the extraction attacks work. Specifically, they follow a prior work (Carlini et al.) in
downloading data from Wikipedia, randomly sampling prompts from continuous 5-token blocks, and performing generation on each prompt. They consider an extraction
successful if the model outputs text that contains a substring of at least 50 tokens from the training data.
+ For semi-closed models, the authors build an auxiliary dataset of downloaded internet text by concatenating four large LLM pre-training datasets (Pile, RefinedWeb,
RedPajama, Dolma) and check if any potentially memorized examples exist in the dataset. If a sequence is outputted with sufficient length and entropy, they state it is
unlikely the text appears on the Internet by coincidence.
+ The authors find that aligned language models are less vulnerable to their base-attack, hypothesizing that it is because of the alignment to act as a helpful assistant (e.g.
ChatGPT). They develop a strategy to encourage GPT-3.5-turbo to diverge from the chatbot-style outputs. Specifically,they ask the model to repeat a word forever, and find
that eventually, some of the generations diverge.
## Can LLMs Keep a Secret [2](#ref2)
The authors introduce a benchmark, ConfAide, which has LLMs reason over a wide range of complexity “Tiers”. The first two tiers involve human privacy expectations. Tier 1
(Info-Sensitivity): involves only one information type; Tier 2 (InfoFlow-Expectation): involves contextual actor and a use component which “defines the entity to whom the
information would flow and the purpose of the flow”. The second two tiers involve theory of mind with regard to contextual privacy reasoning. Tier 3 (InfoFlow-Control): assess
whether model can control flow of information in a given context. Tier 4 (InfoFlow-Application): test LLM privacy reasoning on real-world application inspired scenarios. Human annotations are gathered for tiers 1-3 via Amazon Mechanical Turk.
<img src='./images/secret_fig1.PNG' width=60% height=60%>
Figure 1 in paper (above) shows the ConfAIde benchmark with explanations of the tiers.

## Beyond Memorization [3](#ref3)
+ The authors use PersonalReddit (PR) dataset, and select all attribute labels with a certainty rating of at least 3.
+ The authors use 8 SOTA LLMs to infer 8 personal attributes, showing 85-95% top-3 accuracy on real-world data.

# Key Findings
## Scalable Extraction of Training Data [[1]](#ref1)
+ ChatGPT is less vulnerable to their base-attack, but when used in conjunction with a strategy to cause the model to diverge from the chatbot-style generations (and behave
like a base language model), they are able to extract training data (PII, NSFW content, Literature, URLs, UUIDs and accounts, Code, Research papers, Boilerplate text,
MMerged memorized outputs).
+ They find that the gap between discoverable and extractable memorization was largely because of the lack of prior works to be able to verify that the model outputs were
training data.

## Can LLMs Keep a Secret [2](#ref2)
+ As you move up in the tiers, the correlation between human and model privacy expectation decreases, which the author states confirms their hypothesis that LLMs lack
the ability to reason about secret sharing and privacy.
+ The authors also state that the results highlight the need to directly target theory of mind and reasoning in models.
## Beyond Memorization [3](#ref3)
+Malicious LLMs could steer conversations to have the user give sufficient information to infer private information.
+ They state two potential mitigations: using text anonymization tools to remove personal attributes (however, many can still be inferred), and alignment (they find that models do not currently filter privacy invasive prompts)
\
# Critical Analysis
## Scalable Extraction of Training Data [[1]](#ref1)
+ The extraction technique on production models is specific to ChatGPT, and has not been tested on other proprietary models.
+ The methods are quite costly (both in memory and size). The verification is reliant on a massive dataset, and the experiments with ChatGPT were upper bounded at $200, since
the researchers are generating hundreds of tokens for each prompts.

## Can LLMs Keep a Secret [2](#ref2)
+ The models tested are general LLMs (GPT-4, ChatGPT, InstructGPT, Llama-2 Chat, Llama 2, Flan-UL2). Future work in this direction could look at fine-tuned or aligned
models for the privacy reasoning tasks.
## Beyond Memorization [3](#ref3)
+ While inferring personal attributes does pose a privacy risk if used by malicious actors or otherwise leaked, some inference of personal attributes may also be desirable. For
example, there has been work on the limitations of ChatGPT’s output style being restricted and biased towards the language of certain populations. It could be possible
that some personal attribute protection via text anonymization could also hinder personalization abilities of LLMs. This could be a potential future direction in this area.

# References
<a name="ref1"></a>[1] Scalable Extraction of Training Data from (Production) Language Models, Nasar et al 2023. [link](https://arxiv.org/abs/2311.17035) \
<a name="ref2"></a>[2] Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory, Mireshghallah 2023 [link](https://arxiv.org/abs/2310.17884) \
<a name="ref3"></a>[3] Beyond Memorization: Violating Privacy Via Inference with Large Language Models, Staab et al 2023 [link](https://arxiv.org/abs/2310.07298v1) \
<a name="ref4"></a>[4] Privacy issues in Large Language Models: A Survey. Sections 3,4, and 5. Neel 2024. [link](https://arxiv.org/pdf/2312.06717.pdf) \
<a name="ref5"></a>[5]