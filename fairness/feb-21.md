# Introduction

<!-- Summarize the main idea of the blog here - write this part last -->



## Motivations

<!-- Background information -->
“How to link to other materials: Law affects who can develop AI system and how to develop, including computer fraud and 
abuse act(CFAA) and trade secret, But the law that has the most influence: copyright law”

### Traditional Copyright Law
##### Definition of copy
To discuss if AI training data is in the conflict of the interest of the copyrighted work owners, “copy” should be 
defined first, according to Copyright Act, “copies” are “material objects ... in which a work is fixed by any method 
now known or later developed, and from which the work can be perceived, reproduced, or otherwise communicated, either 
directly or with the aid of a machine or device”. Training data however, is usually stored in the RAM, which makes them 
volatile and “fleeting”. So there is still no agreement of if training data is a “copy”. If it is not a “copy” to 
start with then there is no infringing.


# Methods

<!-- How? Specific laws -->

## Copyright hinderances
Copyright law affects the development of fairer AI systems in three main ways:
### Chill Reverse Engineering
To reverse engineering, normally one needs to understand how the original code works, it would mean that there is a “copy” stored somewhere for the purpose of studying. By doing so the reverse engineer violates the copyright law. 
Most of the time, this act is pushed by a big company in a monopoly market for the maximization of the interest, 
the cost of reverse engineering is thus high for an individual or smaller competitors. 
Consequently this pushes the independent researchers, journalists, etc away.

### Restrict Accountability
The accountability can usually be held up by three main ways:  journalistic reporting, whistleblowers and crowdsourcing audits.
Copyright law restrains all three ways. to make a report piece, which is normally published publicly, 
a part of the dataset will be revealed to the public, which can be considered infringement, similarly, for 
a whistleblower to report something, they will have to “copy” it first. Thus copyright law actually discourage other parties to hold the AI system developers accountable.

### Limit Competition
Training a big model needs a big pile of data, the data used for build large AI system is gained by building or buying, the resources are more accessible for big tech companies, who either have a long history of being in the Internet playfield or have enough wealth to fund such acts. THis makes the other newer parties to enter the field.

### Biased Data
To abide by copyright laws, the developers are driven to use public domain works and Creative Commons-licensed works, 
however these works would embed bias.

#### Public domain biased (old)
Most of them are prior to 1923, thus reflecting a “wealthier, whiter, and more Western” view on society. 
They do not include the more inclusive aspect of the society today.

#### CC-licensed dataset (biased)
CC-license allows the creators to license their “works freely for certain uses, on certain conditions.”, one of CC-licensed dataset most significant example is wikipedia, however there is gender bias, and only 8.5% of the editors were woman in 2011.


# Key Findings

<!-- What? Specific examples -->

### Encourage fair use
Instead of copyright, fair use should be encouraged.

#### What is fair use
Fair use is decided case by case and a combination of the four factors:

1. The purpose and character of the use; 

2. The nature of the copyrighted work; 

3. The amount and substantiality of the portion used in relation to the copyrighted work as a whole; 

4. The effect of the use upon the potential market for or value of the copyrighted work.

To create fairer AI, the dataset should be able to include copy-right protected works, so the dataset can be bigger, 
minimizing the effect of BLFD and researchers can add selected data to balance out the bias.

#### Why using copyright protected works is beneficial and harming the interest of the copyright oweners:

##### Highly transformative
Defination of transformative use: "Must be productive and must employ the quoted matter in a different manner or for a different purpose from the original ... if the quoted matter is used as raw material, transformed in the creation of ... new insights and understandings—this is the very type of activity that the fair use doctrine intends to protect for the enrichment of society."

To use the data for training AI system is transformative, produce new knowledge and contribute to society. Also training is a “nonexpressive use”. So it is fair use to use copyrighted works for training AI systems and it is beneficial to the society.

##### Factual nature
AI is learning the concept and the fact of the works but not “creative component”. Infringement is when one party use a part of the originial work to try to connect with the audience dishonestly and it is not the case for AI training.

##### Copying Entire Works to Train AI Systems Takes a Reasonable Amount and Substantiality of the Copyrighted Works
To examine the system some part of the work should be available, and the work should not be perceived as expressive work but only to identify bias. With the strict definition like this, it should be assumed that the parties who are examining the AI system will not be viewing the work as an expressive work but as an informative work.

##### Does Not Harm the Commercial Market for Copyrighted Works
It is not fair use if “may serve as a market substitute for the original.”, it is not the case for AI training as 
the product is not the data itself but a model. Apart from that if fair use is widely adopted, there will not be a licensing market as it would still be favorable to big tech companies.



# Critical Analysis

<!-- What are your opinions? Where do we go from here? -->

It is a controversial topic, as the copyright law protects the interest of the copyright owners, and it seems that by encouraging fair use, the interest of the copyright owners will be harmed. However, like open source software, in my opinion, in an environemnt where the resources are opener to everyone, there would be more accountability, more competition and less black boxes, and I believe the society as a whole will benefit from it and it would open up more doors for the researchers and parties with smaller resources to contribute to the research.


2. **Big Data's Disparate Impact**
# Introduction and Motivation
## Data Mining Is A Double Edged Sword
When considering society and stakeholders, data mining is a double-edged sword. From one angle, big data has revolutionized many industries through enabling activities such as optimizing delivery routes, mining prescription side effects, and studying human interactions. Additional  advocates of data mining suggest that it can result in a reduction of discrimination by limiting the opportunity for individual bias to impact important assessments and requiring reliable empirical foundations on which decisions must be based. From the other angle, the outcomes of data mining are dependent on the quality of the data the algorithms are training on, or as the common adage goes, “garbage in, garbage out.” However, in the context of data mining, an often insidious result is discriminatory outcomes, whether reflective of historically prejudiced patterns in society and its institutions, prejudices of prior decision makers, or more nefarious (but much less prevalent) intent such as developer use of proxy variables for protected classes. 

# Methods
## Discrimination Resulting From Data Mining Poses New Challenges for Historical Areas of  Antidiscrimination Law 
While discrimination in data mining is in and of itself difficult to root out, it poses additional challenges when viewed from a legal lens of trying to reduce discriminatory outcomes. In this work, they specifically look at Title VII, which prohibits discrimination in employment. 

# Findings
## Limited Liability Under Title VII In Most Cases
Title VII poses several challenges with respect to trying to reduce discriminatory outcomes from data mining. Most notably, liability for discrimination from data mining would be difficult due to  the common situation of not having discriminatory intent, or “disparate intent”. For example, if the data reflects historical patterns of prejudice, if there is no intention to harm, Title VII may hold that data mining is a justifiable business necessity if the outcome is predictive of future employment outcomes. From a technical aspect as well, every step in the data mining process (defining target variable, labeling and collecting training data, using feature selection, making decisions on the basis of the resulting model) could potentially result in discriminatory outcomes for protected classes, so identifying intent somewhere in the pipeline would be a formidable challenge. Rather, under Title VII, potential triggers of liability would include “disparate impact” (discriminatory effects without intent), or “disparate treatment” (data mining systems treat everyone differently). Another solution, however, rather than relying solely on the letter of the law and enforcement agencies, rests on the employers themselves through means such as education. For example, employers may not be aware of the technological underpinnings of data mining and the potential for discrimination.

# Critical Analysis
TODO