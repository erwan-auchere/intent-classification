# Intent classification
link to the report https://www.overleaf.com/8533345687jrjfdvwmhmty

The interaction between individuals is at the center of many organizations. In order to effectively assist their clients, companies could automate customer interaction. Therefore, finding more suitable models to meet this need appears to be a noble task. Several authors have addressed this issue with different analysis methods. 

Among these methods, we have the "linear Conditional Random Field (CRF)," which is a discriminant model (modeling is done using conditional distributions for sequence data, it allows modeling the dependence between each state (a dialogue intent) and all input sequences). We can also mention classification algorithm methods for short texts, Bag-of-Words (BoW), and  Continuous Bag-of-Words (CBoW) trained via a SVM model.

To take into account the complex dependencies between words in the representation of a statement, recurrent neural networks have been introduced. More recently, LSTMs and their simplification Gated Recurrent Unit (GRU) have been used for intention classification. In our work, we propose to use methods based on neural networks for intention classification.
