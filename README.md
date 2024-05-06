# Deep-Learning
Multi-layer perceptron for Sentiment Analysis
1.	Tokenization Strategy: We have a custom method that does the tokenization.  It removes stop words using NLTK english stop words, and NLTKs porter stemmer to stem words as well.  It then uses regex to remove all non-lowercase/numerical values.  We chose to keep numerical values because they have a relative importance for positive or negative reviews.  After the pre-processing is complete, we use the TensorFlow Keras tokenizer to vectorize our training, validation, and testing sets by fitting it on the training data.

2.	Hyper-parameter Optimization Steps: The hyper-parameters such as the learning rate, batch size, number of epochs, and the architecture of the MLP models (number of hidden layers and their sizes) are set manually. The optimization process involves experimenting with different values for these hyper-parameters to find the combination that results in the best performance on the validation set. This process also involves multiple iterations of training and evaluating the models with different hyper-parameter configurations.

Difficulties in hyper-parameter optimization arose due to several reasons:

High computational cost: Training multiple models with different hyper-parameters was computationally expensive, and took a lot of time.  One thing we noticed in regards to this is that increasing batch size decreased computation time but also decreased initial accuracy(epoch 1 accuracy).
Lack of intuition: It was challenging to predict the effect of each hyper-parameter on the model's performance.
Overfitting or underfitting: Finding the right balance between model complexity and generalization performance was difficult, leading to overfitting or underfitting issues.  Increasing the L2 learning rate lambda helped battle this but we never completely escaped overfitting the data.

From hyper-parameter optimization, we learnt the importance of fine-tuning the model's parameters for achieving better performance, as well as gaining insights into the sensitivity of the model's performance to different hyper-parameter settings.

3.	Loss Landscape with Regularization: Introducing regularization (L2 regularization) to the model affected the loss landscape by penalizing large weights in the model. This regularization term was added to the loss function during training, effectively constraining the model's weights and reducing overfitting.
Before regularization, the loss landscape displayed sharper and narrower minima, potentially indicating overfitting to the training data. As regularization strength increased, the loss landscape became smoother, with wider and flatter minima. This regularization-induced smoothing effect helped the model generalize better to unseen data by avoiding overly complex solutions that fit the noise in the training data.
We observed overfitting prior to regularization showing a significant gap between the training and validation loss curves, indicating that the model is fitting the training data too closely, capturing noise rather than underlying patterns. Introducing regularization reduced this gap somewhat and improved generalization performance on the validation set.
4.	Loss Curves for Train and Validation Loss:
Below are the loss curves for both MLP-1 and MLP-2, showing the training and validation losses over epochs:
MLP-1
 

MLP-2
The loss curves demonstrate the training progress of the models over epochs. Lower values of loss indicate better convergence and performance on the training and validation datasets.
5.	Confusion matrices: We constructed confusion matrices using the true labels and predicted labels for both MLP-1 and MLP-2 models.
The confusion matrices visualize the performance of the models in terms of true positive, true negative, false positive, and false negative predictions, providing insights into their classification capabilities.  Having a final accuracy for both models over 0.86 well exceeds the requirements of having an accuracy above 0.786 and we were very excited to see it.  Running both models an additional 10 times with different seeds returned a mean accuracy of 0.865 with a Stddev of 0.0029 for MLP1 and a mean accuracy of 0.858 with a Stddev of 0.0036 for MLP2.
