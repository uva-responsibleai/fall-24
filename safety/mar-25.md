# Introduction

<!-- Summarize the main idea of the blog here - write this part last -->

![](./images/ChatGPTEthicsWarning.PNG)

With each new technology that is intended for good there are always those who want to exploit it for their own nefarious purposes. Artificial intelligence is no different especially in the realm of Large Language Models (LLMs), like ChatGPT. Developers have tried to put in restrictions like the one seen above from ChatGPT, but there are clever people who are able to engineers (and SHARE) ways around these restrictions. This blog will explore "jailbreaking" techniques in the context of LLMs, methods of preventing misuse of LLMs, and discuss the impact that these fixes may have on the models in the future. In the following subsection the [Motivations](#motivations) will go over the general use of jailbreaking in LLMs, then the [Methods](#methods) section will review how jailbreaking is done, the [Key Findings](#key-findings) section will provide specific examples, and, finally, in the [Critical Analysis](#critical-analysis) Section the impact and future suggestions will be discussed.

<!-- How do they spread -->

## Motivations
<!-- Background information -->

Large Language Models (LLMs) and Visual Language Models (VLMs) are artificial intelligence programs that take user input and output a helpful answer for the given questions. When an adversary decides to attack the model they can bypass the aforementioned censorship by using carefully crafted statements to trick the model. This practice is called "jailbreaking". Jailbreaking can come in multiple forms in the context of LLMs. The two main forms that will be discussed are [textual](#textual-attacks) and [multimodal jailbreaking](#multimodal-attacks).

These toxic prompts are engineered by adversaries that want to abuse the LLMs and then spread through the internet on sites like Reddit or Discord. By the time the developers can design a system that counteracts these prompts the adversaries can already design a new one. Now the question is "why should users, general public, developers, or companies care?". The reason is twofold: (1) users should care because these prompts can be used to construct convincing phishing scams, codes for worms and malware, or used for other negative means and (2) companies who allow this use would be showing neglegence which could effect their reputations. By circumventing the boundaries that have put in place for the LLM many bad actors could have access to a very powerful tool.

<!--

What will the point be?
* Jailbreaking in LLMs
    * How it is used
    * How it spreads and evolves

What will there to be discussed?
* How to stop it

* What are "jailbreaking prompts"?
    * How are they engineered/created?
        * Words
        * Images/multimodal (audio maybe too)
    * Where are they found?
* Show an image of examples of prompt and chatgpt interactions -->

<!-- 
* Why is jailbreaking important to LLMs
* How is it disruptive?
* Definition of censorship in the context of LLMs -->

# Methods

<!-- How? -->

<sup><sub>[Home](#introduction)</sub></sup> 

## Attack Types

### Textual Attacks

Textual jailbreaking can be through ciphers, Mosaic Prompts, or more generally just cleverly written and tricky prompts.  

### Multimodal Attacks

![](./images/visualattack.PNG)

## Censorship and Other Solutions

Some argue that censorship of LLM should be a physical design problem not a problem that is just bandaided by more Machine Learning [3].

$$
\begin{align}
S \in P & \text{  This is an example}
\end{align}
$$

<!-- * How jailbreaking was monitored and looked over
* How it plays into LLMs
* What safegaurds are in place and how effective?
    * Sectin 7 of Do Anything Now: training sets are a limitation
* Protecting against attacks:
    * DiffPure: using diffusion to return image back to original manifold -->

# Key Findings

<sup><sub>[Home](#introduction)</sub></sup>

<!-- What? -->

<!-- 

https://www.reddit.com/r/ChatGPT/comments/1b8qzsd/the_riddle_jailbreak_is_extremely_effective/

https://www.reddit.com/r/ChatGPT/comments/120byvx/using_a_cipher_to_communicate_seems_to_bypass/

* Where has jailbreaking been used and more importantly how has it been counteracted?
    * Visual
    * Textual
* How have multi-modal attacks been counteracted?  -->

# Critical Analysis

<sup><sub>[Home](#introduction)</sub></sup>

<!-- What are your opinions? Where do we go from here? -->

<!-- Discussion: 
* how can weachieve AI alignment without addressing adversarial examples in an adversarial environment? 
* how do we protect against images that are embedded with bad content?
    * how do we protect against other modes?
* What policies could be in place?
    * Do we just make policies to make it so only text is allowed for LLM? -->

# References