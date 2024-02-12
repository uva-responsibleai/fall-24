# Introduction

Language model (LM) refers to systems which are trained on string prediction tasks: that is, predicting the likelihood of a token (character, word or string) given either its preceding context or (in bidirectional and masked LMs) its surrounding context. Due to various reasons, toxicity and different types of biases may be introduced into a language model. In this presentation, we will first discuss the reasons for having toxicity and bias in LLMs. We will then discuss different frameworks and benchmarks developed for measuring bias and toxicity, and we will see how well different popular language models perform in these tests. Lastly, some solutions for bias mitigation are introduced.


## Motivations

Bias:
Bias is prejudice in favor of or against one thing, person, or group compared with another, usually in a way considered to be unfair. bias of an estimator (or bias function) is the difference between this estimator's expected value and the true value of the parameter being estimated.




Toxicity: Content that can offend or harm its recipients, including hate speech, racism, and offensive language [Kurita et al. 2019]

The size of data available on the web has enabled language models to achieve high accuracy on specific benchmarks. However, in both application areas, the training data has been shown to have problematic characteristics, resulting in models that encode stereotypical and derogatory associations along gender, race, ethnicity, and disability status. In this section, we discuss how large, uncurated, Internet-based datasets encode the dominant/hegemonic view, which further harms people at the margins.

1- Disproportionate contribution to the training data

The Internet is a large and diverse virtual space, and accordingly, it is easy to imagine that very large datasets must therefore be broadly representative of the ways in which different people view the world. However, on closer examination, we find that there are several factors which narrow Internet participation.

Starting with who is contributing to these Internet text collections, we see that Internet access itself is not evenly distributed, resulting in Internet data over-representing younger users and those from developed countries. For instance, GPT-2’s training data is sourced by scraping outbound links from Reddit, and Pew Internet Research’s 2016 survey reveals 67% of Reddit users in the United States are men, and 64% between ages 18 and 29. Similarly, recent surveys of Wikipedians find that only 8.8–15% are women or girls.

Furthermore, while user-generated content sites like Reddit, Twitter, and Wikipedia present themselves as open and accessible to anyone, there are structural factors including moderation practices which make them less welcoming to marginalized populations. There have been multiple cases documented  where people on the receiving end of death threats on Twitter have had their accounts suspended while the accounts issuing the death threats persist. Even if populations who feel unwelcome in mainstream sites set up different fora for communication, these may be less likely to be included in training data for language models.

2- Static Data/Changing Social Views

A central aspect of social movement formation involves using language strategically to destabilize dominant narratives and call attention to underrepresented social perspectives. Social movements produce new norms, language, and ways of communicating. For instance, the Black Lives Matter movement (BLM) influenced Wikipedia article generation and editing such that, as the BLM movement grew, articles covering shootings of Black people increased in coverage and were generated with reduced latency.

An important caveat is that social movements which are poorly documented and which do not receive significant media attention will not be captured at all. Media coverage may not report on protests and social movements accurately and may present a biased view of events that challenge the government. This is seen when media outlets prioritize violent or dramatic events over peaceful protests, resulting in negative coverage. Consequently, the data used in media reporting may not accurately represent social movements and may favor existing power structures.


# Methods
Several frameworks have been introduced for measuring different types of bias and toxicity in language models. In this section, we will discuss a few, as well as provide examples of how well different language models perform in these frameworks.

1- Hate Speech Detection: ETHOS
 
ETHOS is used to measure the ability of OPT-175B to identify whether or not certain English statements are racist or sexist. OPT-175B is compared with the Davinci model with Zero-shot, One-shot and Few-shot(both binary and multiclass). The model responded with yes/no to the question whether the given text is racist or sexist. OPT175B performs considerably better than Davinci. The author speculates that this 
could be due to: 1)The use of the DaVinci API for evaluation might introduce safety control mechanisms beyond those present in the original GPT-3 model used in prior research. 2) The extensive inclusion of unmoderated social media discussions in the pre-training dataset could contribute to additional bias, aiding in classification tasks.

2- Intersentence Biases: CrowS-Pairs
	
	CrowS-Pairs is a benchmark developed for masked language models. It measures biases within sentences across nine categories, including gender, religion, race/color, sexual orientation, age, nationality, disability, physical appearance, and socioeconomic status. Each example includes a pair of sentences representing either a stereotype or an anti-stereotype about a particular group. The aim is to measure a model's preference for stereotypical expressions, with higher scores indicating greater bias exhibited by the model. OPT175B demonstrates greater stereotypical biases across almost all categories compared to Davinci, with the exception of religion.

3- Stereotypical Bias: StereoSet
	Steroset were used to measure stereotypical bias across 4 categories: profession, gender, religion and race. Steroset includes both intra-sentence measurement and inter-sentence level measurement. To address a potential trade-off between bias detection and language modeling capability, StereoSet incorporates two metrics: Language Modeling Score(LMS) and Stereotype Score(SS),which are then combined to form the Idealized Context Association Test score(ICAT). The scores were being normalized by token count. Davinci and OPT-175B have similar scores for this one. 

4- Toxicity: RealToxicityPrompts
 The RealToxicityPrompts dataset was used to evaluate the tendency of OPT-175B to respond with toxic language. The researchers sample 25 generations of 20 tokens using nucleus sampling  with a probability of 0.9 for each of the 10,000 randomly selected prompts from RTP. They then report the mean toxicity probabilities of these continuations, categorized across bucketed toxicities of the original prompts. For comparison, they provide bucketed toxicity rates from Davinci and PaLM. OPT-175B has a higher toxicity rate than either PaLM or Davinci. As the toxicity of the prompt increases, all 3 models show a higher likelihood of generating toxic continuations. And two metrics were used to evaluate toxicity of generated texts. 


# Key Findings

After examining the propagation of bias and toxicity in language models and acknowledging their widespread prevalence, we will now explore strategies for mitigating these issues.

1- Documentation

When we rely on ever larger datasets, we risk incurring documentation debt, i.e. putting ourselves in a situation where the datasets are both undocumented and too large to document post hoc. While documentation allows for potential accountability, undocumented training data perpetuates harm without recourse.

The solution is to budget for documentation as part of the planned costs of dataset creation, and only collect as much data as can be thoroughly documented within that budget.

2- Detoxification

a) Data-based: In this approach, we pretrain the language model further. We can either perform pre-training on a non-toxic subset of a balanced corpus, or by prepending a corresponding toxicity attribute token (<|toxic|>, <|nontoxic|>) to a random sample of documents and pretraining the language model further,  where in the generation experiments, the <|nontoxic|> token is prepended to our prompts.

b) Decoding-based: In this approach, we only change the generation strategy without changing model parameters. Because of additional costs of training language models further, this approach is more readily usable by many practitioners. One strategy could be altering the past and present hidden representations to better reflect the desired attributes, using gradients from a discriminator.










Another strategy is word filtering, i.e., disallowing a set of words from being generated by the model, where the probability of generating any word from a list of profanity, slurs, and swear words is set to zero.


# Critical Analysis

None of these benchmarks for measuring bias and toxicity are perfect, however. As a case in point, the toxicity detection benchmark uses an imperfect measure of toxicity that could bias the toxicity towards lexical cues, failing to detect more subtle biases and incorrectly flagging non-toxic content. The comment below has been flagged as very rude. Although it contains an expletive, but the content is overwhelmingly positive.










Toxicity detectors may also exhibit bias by inaccurately identifying texts as toxic when written in African American English (AAE).

