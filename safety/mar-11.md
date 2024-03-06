
# Introduction and Motivations
Adversarial attacks manipulate machine learning models by exploiting vulnerabilities in their decision-making processes. They can be split into two categories: causative adversarial attacks and exploratory adversarial attacks. Poisoning attacks fall under the causative adversarial attack category because it involves either injecting new malicious points or manipulating existing ones in the training data to degrade the model's performance generally or for specific targets. Subsequently, poisoning attacks can be categorized into either poisoning availability attacks or poisoning integrity attacks. Poisoning availability attacks aim to disrupt the availability of the model's prediction results indiscriminately, often leading to denial-of-service (DoS) scenarios. Conversely, poisoning integrity attacks target specific mispredictions at test time, therefore undermining the integrity of the model's output for malicious purposes while also in most cases keeping their attack undected.
    
In the context of online training, where machine learning models are continuously updated based on incoming data, poisoning attacks can be particularly potent. Depending on the training system, control over a small number of devices that submit fake information adversarially-poison the dataset.


# Methods

(I think we should split up the methods into poisoning attacks and proposed defenses
## Attacks


## Defenses

# Key Findings
The [Poisoning Attacks against SVMs](https://arxiv.org/abs/1206.6389) paper contributes these findings:
- An adversary can "construct malicious data" based on predictions of how an SVM's decision function will change in reaction to malicious input
- A "gradient ascent" attack strategy against SVMs can be kernelized, allowing for attacks to be constructed in the input space
    - The attack method can be kernelized because it only depends on "the gradients of the dot products in the input space"
    - This is a big advantage for the attacker when compared to a method which can't be kernelized; an attacker using a non-kernelizable method "has no practical means to access the feature space"
        - The authors note that this emphasizes that "resistance against adversarial data" should be considered in the design of learning algorithms

The [Certified Defenses for Data Poisoning Attacks](https://arxiv.org/abs/1706.03691) paper discusses how certain defenses against data poisoning can be tested against "the entire space of attacks".
- Specifically, the defense must be one that removes "outliers residing outside a feasible set," and minimizes "a margin-based loss on the remaining data"
- The authors' approach can find the upper bound of efficacy of any data poisoning attack on such a classifier
- The authors analyze the case where a model is poisoned and where both the model and its poisoning detector are given poisoned data

# Critical Analysis
The [Poisoning Attacks against SVMs](https://arxiv.org/abs/1206.6389) paper describes a kernelizable method of attacks against SVM models. The authors note that this work "breaks new ground" in the research of data-driven attacks against such models because their attack strategy is kernelizable. Their findings and conclusions are intuitive and easy to follow from the reader's point of view. Their attack strategy description is accompanied by a clear example: an attack on a digit classifier. They clearly show how the attacker is able to use their strategy to modify training input data to produce classification error. These attack modifications are understandable to the user as well: they clearly show how the attacker modifies the data so that the "7 straightens to resemble a 1, the lower segment of the 9 becomes more round thus mimicking an 8," and so on. Overall, the authors of this paper adequately communicate the technical aspects of their attack strategy in a way that even non-technical readers can easily understand, and the conclusions that follow are well-supported by the evidence they gathered in their attack.

    
