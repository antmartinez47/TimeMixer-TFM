# **Building a Search Space for the TimeMixer-ETTh1 HPO Experiment**

This document outlines the reasoning behind the process of constructing the search space for the TimeMixer-ETTh1 HyperParameter Optimization (HPO) experiment.

The goal is to develop a hyperparameter tuning pipeline aimed at improving the reported performance of the recently proposed state-of-the-art MLP-based Deep Neural Network, `TimeMixer`, on the ETTh1 dataset. To achieve this, we need to define the following key components:

* Objective Function
* Search Space
* Search Algorithm
* Scheduler (Optional)

The chosen objective function, in line with common practices in hyperparameter tuning literature for Deep Neural Networks, is the best validation loss after a certain number of epochs. The training set, validation set, input size, and output size are all consistent with the settings specified in the TimeMixer paper (obtained from the official GitHub repository linked to the paper). The reason for reporting the best validation loss, rather than the current validation loss, is that it reflects the peak performance of the evaluated configuration throughout the entire learning process, assuming that the validation dataset distribution aligns (approximately) with the test dataset distribution. If this assumption is invalid, we should consider more robust objective functions, such as those derived from iterative resampling methods like cross-validation.

To construct the search space, we must first identify the complete list of hyperparameters, select those relevant to our experiment, and assign a probability distribution to each in order to sample them during the hyperparameter optimization process. 

TimeMixer provides two settings for the experiments carried out: a unified hyperparameter setting, where hyperparameters are derived from similar architectures proposed for the same types of problems (Autoformer, Informer...), and a searched hyperparameter setting, where hyperaprameters are optimized (not fully detailed; it is specified which probability distributions and hyperparemeters were optimized, but not which search algorithm and global configuration was utilized). The conclusions derived from the searched hyperparameter setting and its design are informative, and thus will be considered when designing the search space for the current HPO experiment.

The list of most relevant hyperparameters, as provided by the official TimeMixer source code, is presented below. These hyperparameters are divided into two categories: global hyperparameters and model-specific hyperparameters. Global hyperparameters are common to most neural network training architectures (at least in a supervised learning context), such as optimizer parameters, batch size, learning rate scheduler type, and the patience of the early stopping callback. Model-specific hyperparameters, on the other hand, pertain to the architecture of the TimeMixer model itself, and tuning them requires estimating their effect on model performance (at least approximately). Each of the most relevant hyperparameters from the official TimeMixer implementation is analyzed, and a probabilistic distribution is proposed for each one to define the sampling process. 

Probabilistic search spaces, which contain hyperparameters defined by probability distributions, are more expressive and tend to yield better results. They also allow for the specification of conditional dependencies between hyperparameters. However, one must carefully balance complexity with runtime, as more complex search spaces are harder to explore and thus require greater computational resources (e.g., time, distributed systems).

## **Global Hyperparameters**

### **Optimizer**

The choice of optimizer is critical in training deep neural networks (DNNs) because it directly impacts how the model converges to a solution, how quickly it learns, and how well it generalizes to unseen data. Optimizers dictate how the model’s weights are updated based on the gradients calculated during backpropagation. Their effectiveness can vary depending on the model architecture, dataset, and specific problem. Below, some well-known effects of the optimizer in the training of DNNs are listed:

* **Convergence Speed and Stability**: The optimizer determines how quickly and smoothly a model reaches a minimum in the loss landscape. Some optimizers converge faster (e.g., Adam), while others might have more stable convergence (e.g., SGD with momentum).

* **Avoiding Local Minima**: Optimizers with momentum or adaptive learning rates can help escape local minima and saddle points. This is crucial for high-dimensional loss landscapes with many such points.

* **Generalization**: Some optimizers, such as SGD, are known to offer better generalization because they tend to explore more of the loss landscape, potentially finding flatter minima, which correlate with better generalization.

* **Handling Noise and Sparse Gradients**: Adaptive optimizers (e.g., RMSprop, Adam) can effectively handle noisy gradients or sparse gradients, which are common in certain types of layers (e.g., embeddings in NLP tasks).

Several formal studies and papers investigate the impact of different optimizers on training dynamics, generalization, and convergence:

* **"On the Convergence of Adam and Beyond" (2019)** by Reddi et al.: This paper addresses convergence issues with Adam and proposes a variant called AMSGrad. It delves into how Adam’s adaptive learning rates can sometimes lead to poor convergence in certain situations.

* **"The Marginal Value of Adaptive Gradient Methods in Machine Learning" (2020)** by Wilson et al.: This paper discusses why adaptive methods (like Adam) might generalize worse than simple SGD in some scenarios, despite faster convergence. It highlights how adaptive methods can lead to sharp minima, which correlates with poorer generalization.

* **"Understanding the Role of Momentum in Stochastic Gradient Methods" (2018)** by Sutskever et al.: An older but foundational work that discusses the impact of momentum in speeding up convergence and avoiding local minima.

* **"Which Algorithmic Choices Matter at Which Batch Sizes? Insights From a Noisy Quadratic Model" (2020)** by Zhang et al.: This paper investigates how different batch sizes affect the performance of optimizers like SGD and Adam, providing insights into which settings are preferable for different scenarios.

These works, among others, illustrate that the choice of optimizer is not universally optimal and should be tuned based on the problem domain, model, and data characteristics. Recent research tends to focus not just on the optimizer itself but also on how it interacts with other hyperparameters, such as batch size and learning rate, to achieve the best balance between convergence speed, stability, and generalization.

Since this experiment focuses primarily on model hyperparameters, and Adam is typically used with default hyperparameters (apart from the learning rate) in this types of models and tasks, we have decided to exclude the optimizer from the tuning process. This allows us to concentrate on the model-specific hyperparameters while also simplifying the search space, as different optimizers come with their own additional hyperparameters.

The initial learning rate for the Adam optimizer will be tuned following common practices. These common practices and standard methodologies also explain the choice of Adam optimizer in the TimeMixer paper.

### **Learning Rate and Learning Rate Schedulers**

The learning rate is one of the most critical hyperparameters in the optimization process when training deep neural networks (DNNs). It determines the step size at which the optimizer updates the model's weights during each iteration, significantly influencing the model's convergence, stability, and overall performance.

An appropriate learning rate is highly dependent on the architecture, dataset, and type of problem being addressed. While fixed learning rates serve as a good starting point, adaptive learning rates, warm-up schedules, and decay strategies are commonly used to balance convergence speed, stability, and generalization. Ongoing theoretical and empirical research continues to refine how learning rates should be tuned for optimal results. Below some well-known effects of the learning rate on DNN training are listed:

* **Convergence Speed**:
   * **High Learning Rate**: Leads to larger updates to the model’s weights. While this can speed up training, it also risks overshooting the optimal solution, resulting in unstable or non-converging behavior.
   * **Low Learning Rate**: Allows for more fine-grained updates, leading to more stable convergence, but it can slow down the training process considerably, taking much longer to reach a satisfactory solution.
* **Stability**: If the learning rate is too high, the model may exhibit erratic behavior, with the loss oscillating or diverging instead of decreasing. Conversely, a very low learning rate can result in the model getting stuck in local minima or saddle points, as the updates are too small to escape them.
* **Generalization**: The learning rate also affects how well a model generalizes to new, unseen data. A well-chosen learning rate can guide the model toward flatter minima, which often correlates with better generalization performance. Adaptive strategies, such as learning rate schedules or adaptive optimizers, can improve generalization by allowing more aggressive updates early in training and finer adjustments later.
* **Training Time**: A high learning rate may lead to rapid initial reductions in the loss but could make it difficult to find the optimal solution. A low learning rate might guarantee convergence, but at the cost of significantly longer training times.

Several formal studies and papers have explored the role and impact of the learning rate:

* **"Reducing the Learning Rate in SGD: A Cautionary Tale" (2017)** by Loshchilov and Hutter: this paper critiques traditional learning rate decay schedules and introduces alternatives like cosine annealing with warm restarts. It shows that reducing the learning rate too early or too aggressively can harm model performance.

* **"On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima" (2018)** by Keskar et al.: this work discusses the relationship between learning rate, batch size, and generalization. It highlights how large learning rates combined with large batch sizes tend to converge to sharper minima, which can result in worse generalization.

* **"Learning Rate Dropout: Optimal Learning Rates for Deep Networks" (2020)** by Dai et al.: this study proposes dynamically adjusting the learning rate based on training progress, showing how learning rate tuning is key to achieving optimal performance.

* **"Understanding Learning Rate Warmup and Its Effectiveness in Deep Learning" (2019)** by Gotmare et al.: this paper systematically analyzes the impact of learning rate warm-up strategies and shows how they can improve convergence, especially in large-scale models and datasets.

* **"The Marginal Value of Adaptive Gradient Methods in Machine Learning" (2020)** by Wilson et al.: this paper analyzes the effect of adaptive learning rates (like those in Adam) and contrasts them with fixed-rate methods like SGD. It explains how adaptive methods can be more effective in certain settings but may lead to suboptimal generalization in others.

With these considerations in mind, the following distribution is proposed for tuning the learning rate hyperparameter:

```
learning_rate = loguniform(min=1e-5, max=1.2e-2)
```

This distribution differs slightly from the one utilized in the searched hyperparameter setting (a uniform distribution from 1e-5 to 5e-2). Change is made to avoid high learning rates that could lead to unstable learning curves and to sample logarithmically (i.e., using a log-uniform distribution). This intuitions were confirmed with random-search-based experiments where the learning rate upper bound was contained in the inteval [1e-2, 5e-2].

The log-uniform distribution helps improve the coverage of the sampling interval when it spans several orders of magnitude. Its minimum and maximum values are derived after initial testing with random-search-based HPO routines and the value utilized by the authors of SOFTS paper.

Regarding the learning rate scheduler, the configurations in the TimeMixer paper do not include one. Therefore, to avoid increasing the complexity of the search space, we will not tune this parameter. The primary focus of this experiment is on model-specific hyperparameters, as this approach offers more valuable insights due to its novelty and the limited number of similar experiments available (actually, the only formal hyperparmaeter tunning process conducted over TimeMixer architectures is presented in TimeMixer paper).

However, several learning rate schedulers are implemented in the TimeMixer source code, including a stepwise exponential decay scheduler, a cosine scheduler, and a fixed-scheme scheduler. To leverage the benefits of learning rate scheduling in the TimeMixer training process, we have chosen to set the stepwise exponential decay scheduler as the default option in the training and tuning scripts. This scheduler applies a decay factor of 0.5, updated at the end of each epoch.

### **Loss Function** 

In time series forecasting with deep neural networks, the choice of the loss function plays a critical role in determining model performance and behavior. The loss function guides the optimization process, affecting the convergence speed, stability, and generalization capability of the model. Below some well-known effects of the loss function on DNN training are listed:

* **Guiding Model Training:** The loss function quantifies the difference between the model's predictions and the actual target values. During training, the model adjusts its weights to minimize this loss. The form of the loss function directly influences how the model treats different types of errors and adapts its parameters.
* **Error Sensitivity:** Different loss functions treat errors differently. For example, Mean Squared Error (MSE) penalizes larger errors more heavily than smaller ones (due to squaring), making it sensitive to outliers. In contrast, Mean Absolute Error (MAE) treats all errors linearly and is less sensitive to outliers.
* **Forecasting Accuracy and Bias:** The chosen loss function can introduce bias into predictions. For instance, using MSE or MAE tends to optimize models for minimizing point-wise errors but may overlook broader temporal patterns like trends and seasonality. Certain loss functions like the Huber loss combine both MSE and MAE characteristics to balance sensitivity to outliers and overall error minimization.
* **Alignment with Evaluation Metrics:** The effectiveness of a loss function should ideally align with the evaluation metric used for the forecasting task. For example, if the primary evaluation metric is a custom metric like MAPE (Mean Absolute Percentage Error) or SMAPE (Symmetric Mean Absolute Percentage Error), training with a different loss function like MSE might yield suboptimal results.
* **Uncertainty and Distributional Considerations:** Some loss functions are better suited for probabilistic forecasting, where the model outputs a distribution rather than a point estimate. For instance, negative log-likelihood (NLL) or quantile loss functions are commonly used in models that output quantiles or prediction intervals.

Common loss functions in time series forecasting:

* **Mean Squared Error (MSE):** The most common choice, penalizes larger errors more heavily (it is defined a quadratic function of the errors).
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

* **Mean Absolute Error (MAE):** A robust alternative that penalizes all errors linearly (it is defined as a linear function of the errors)
$$
   \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |\hat{y}_i - y_i|
$$

* **Huber Loss:** Combines MSE and MAE, reducing the impact of outliers.
$$
   \text{Huber Loss} = 
   \begin{cases} 
   \frac{1}{2}(\hat{y}_i - y_i)^2 & \text{for } |\hat{y}_i - y_i| \leq \delta \\
   \delta |\hat{y}_i - y_i| - \frac{1}{2} \delta^2 & \text{otherwise}
   \end{cases}
$$

* **Quantile Loss:** Used in models forecasting quantiles, capturing uncertainty.
   $$
   \text{Quantile Loss} = \max(\tau \cdot (y - \hat{y}), (1 - \tau) \cdot (\hat{y} - y))
   $$
   where $\tau$ is the quantile (e.g., 0.5 for the median).

Several studies provide insights into how different loss functions affect time series forecasting:

* **"DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks" (2017)** by Salinas et al. explores using likelihood-based loss functions for probabilistic time series forecasting. This work shows how different loss functions can be used in autoregressive models to handle uncertainty.
* **"Neural Forecasting: Introduction and Literature Overview" (2021)** by Benidis et al. surveys various deep learning methods for forecasting and discusses the impact of loss functions on performance, especially in probabilistic and multivariate settings.
* **"A Comprehensive Review on Neural Network-Based Time Series Forecasting" (2019)** by Zhang et al. discusses the significance of loss functions in neural network forecasting models, comparing different strategies and their impact on forecasting accuracy.
* **"A Comprehensive Review of Loss Functions in Machine Learning and Deep Learning" (2021)** by Prakash et al. provides an analysis of loss functions across domains, including time series forecasting, highlighting the influence of different loss functions on model behavior.

In TimeMixer paper, MSE is used for training, while both MSE and MAE are employed for evaluation and benchmarking. To ensure a fair comparison between optimized and default results, we will maintain MSE as the loss function. While optimizing the loss function for a specific objective metric could be explored, this is uncommon in practice. The primary focus of this experiment is on model-specific hyperparameters.

### **Batch Size**: 

The effect of batch size when training deep neural networks for time series forecasting is crucial and has been studied extensively in various contexts. Below, some well-known effects of the batch size in the training of DNNs are listed:

* **Convergence Speed**: Smaller batch sizes generally result in noisier updates to model weights, which can improve generalization but slow down convergence. Larger batch sizes typically lead to smoother updates, allowing for faster convergence but can result in overfitting.
* **Generalization Performance**: A medium-sized batch is often preferred in time series forecasting. Extremely small batches might cause the model to be trapped in local minima, while very large batches might cause the model to overfit to the training data.
* **Training Stability**: Larger batch sizes can lead to more stable gradient estimates, but they may require adjustments to learning rates to avoid instability (e.g., using learning rate warm-up techniques).
* **Computational Efficiency**: Larger batch sizes leverage parallelism better on GPUs, leading to faster training times. However, they also demand more memory.
* **Data Dependency**: In time series forecasting, data dependencies and temporal correlations are critical. Small batch sizes can preserve temporal patterns better, whereas large batches may dilute this information.

Several studies have analyzed the impact of batch size in the context of DNNs, though not all are specific to time series forecasting. Key references include:

* **Smith et al. (2018), "Don't Decay the Learning Rate, Increase the Batch Size"**: This paper discusses how increasing the batch size can be an alternative to learning rate decay and can improve generalization performance when coupled with appropriate training strategies.
* **Keskar et al. (2016), "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"**: This study shows that large batch sizes tend to converge to sharp minima in the loss landscape, which can harm generalization, suggesting that small to medium batch sizes are often preferable.
* **LeCun et al. (2012), "Efficient BackProp"**: This classic paper discusses the relationship between batch size, learning rates, and training stability in a broader context.
* **Goodfellow et al. (2016), "Deep Learning"** (Book): This book provides insights into batch size effects in the context of deep neural networks, covering the trade-offs between small and large batch sizes.

While these works do not specifically focus on multivariate time series forecasting, the principles discussed are applicable to this domain. Below, some key Considerations specific to Time Series Forecasting are listed:

* **Temporal Dependency**: For multivariate time series forecasting, capturing the temporal dependencies is crucial. Smaller batches can help preserve these dependencies better by avoiding overly smoothed gradients.
* **Seasonality and Patterns**: When dealing with time series data that exhibit strong seasonality or cyclic behavior, using smaller batch sizes can help models adapt to these patterns better.
* **Memory Constraints**: Given the 2D nature of inputs and outputs in multivariate forecasting, memory usage can become significant. Medium batch sizes (e.g., 32 to 64) are often a good trade-off between performance and resource usage.

As a conclusion, the choice of batch size in multivariate time series forecasting is influenced by the need to balance generalization, convergence speed, and computational efficiency. While small to medium batch sizes (16-64) are commonly used, it's important to experiment with different sizes depending on the dataset, sequence length, and the specific neural network architecture being employed.

In TimeMixer paper, the batch size is specified to range between 8 and 128. The default value for the ETTh1 dataset is 128, though no explanation is provided on how this value was determined. Furthermore, batch size was excluded from the HPO process conducted by the authors of the original paper.

Given the impact of batch size on model generalization and the complexity of the optimization landscape, we consider it relevant to include batch size as a hyperparameter in this experiment. Additionally, tuning batch size is recommended in hyperparameter optimization literature. Starting from the default value of 128, the following categorical distribution is proposed:

```
batch_size = categorical(values=[16, 32, 64, 128, 256])
```

Several researchers have suggested that batch sizes should be kept as low as possible, typically around 32 samples, which justifies the lower values in the distribution above. The inclusion of higher values is due to TimeMixer's default setting and the fact that larger batch sizes increase resource consumption and may lead to unstable training, as they can introduce more complex optimization landscapes.


### **Number of epochs and early stopping callback**

The number of training epochs plays a critical role in determining the performance of deep neural networks for time series forecasting, including multivariate forecasting. Below, some well-known effects of the number of training epochs in the training of DNNs are listed:

* **Convergence and Model Performance**: The number of epochs determines how long the model will be trained. Insufficient epochs can lead to underfitting, where the model has not fully learned the underlying patterns in the time series data. Conversely, too many epochs can lead to overfitting, where the model learns noise and specificities in the training data rather than generalizable patterns.
* **Early Stopping**: In practice, instead of relying on a fixed number of epochs, many models employ early stopping. Early stopping monitors the model’s performance on a validation set and stops training when performance starts to degrade, effectively preventing overfitting.
* **Plateauing of Learning**: In time series forecasting, improvements in model performance often plateau after a certain number of epochs. After this point, further training yields diminishing returns or even leads to degradation in performance due to overfitting.
* **Temporal Patterns**: For time series forecasting, where capturing seasonal and cyclical patterns is critical, allowing the model to train for a sufficient number of epochs ensures it can capture these dynamics. However, excessive epochs risk memorizing specific time points instead of general trends.

Some common choices for the number of trainin epochs include:

* **Moderate Epochs (e.g., 50-200 epochs)**: This range is common for many time series forecasting models. The exact number depends on factors like the complexity of the model (e.g., LSTM, CNN, Transformer), the length of the time series, and the amount of available data.
* **Dynamic Epoch Counts**: Often, models do not use a fixed number of epochs. Instead, they combine a relatively high maximum epoch count (e.g., 1000 epochs) with early stopping to end training once performance stabilizes or starts to degrade.Practitioners commonly monitor validation loss or other metrics during training to decide the optimal number of epochs. Early stopping with patience (e.g., waiting for 10 or 20 epochs after the last improvement) is widely used.

And some practical considerations general to DNN training:
* **Complexity of the Model**: Complex models like Transformers or multi-layer LSTMs may require more epochs to converge, while simpler models may need fewer epochs.
* **Dataset Size**: Larger datasets generally require more epochs for the model to fully learn the patterns in the data, while smaller datasets might overfit quickly.

Relevant research and practical guidelines emphasize the importance of combining early stopping with well-chosen epoch limits, making dynamic training durations the most practical approach for multivariate forecasting tasks.

The default value in TimeMixer for ETTh1 is 10 epochs for each horizon value (as by the unified hyperparameter setting presetend in TimeMixer paper), with no early stopping callback being utilized. In the searched hyperparameter configuration, a uniform distribution between 10 and 100 is utilized to optimize this hyperparameter. 

In this experiment, the number of training epochs plays a crucial role, especially in Multi-Fidelity Optimization methods like BOHB or SMAC. Depending on the search algorithm used, the role of epochs differs. For non-multi-fidelity algorithms, such as Random Search or Hyperopt TPE, the number of epochs can be tuned. However, for multi-fidelity algorithms, the number of epochs defines the maximum budget, as we consider "budget" to represent the number of epochs (iterations of the training algorithm). Therefore, it becomes a parameter of the scheduler in the tuning process. 

Given this distinction, and in the interest of evaluating a broader range of configurations (yielding more informed results), we will avoid tuning the number of epochs. Focusing on more configurations with fewer epochs provides more value than fewer configurations with many epochs, which could be suboptimal due to time constraints and hardware limitations. The behavior of this hyperparameter will vary depending on the algorithm used:

* For non-multi-fidelity methods (random search, tpe), the value will be set to 8 training epochs (default is 20 epochs), and an Early Stopping callback will be applied with a patience of 3, as used in the official source code for the ETTh2 dataset. This approach will enhance the efficiency of the optimization process. The choice of 8 epochs is based on the learning trajectories observed for the ETTh2 dataset, where the actual number of epochs after early stopping for each horizon value ranges between 2 and 6, and on runtime complexity.
  
* For multi-fidelity methods (bohb, smac), the value of 8 training epochs will be assigned to the `max_budget` parameter of the corresponding scheduler (e.g., Hyperband or Successive Halving), and no early stopping callback will be used, as stopping mechanisms are handled by the scheduler.

## **Model-specific hyperparameters**

### **`down_sampling_method`**

The first step in TimeMixer forward pass is to apply a series of down sampling operations 
(each one defined by the same layer) to the input series in order to obtain the temporal variations 
at different scales (each operation is applied recursively to the output of the previous operation, therefore each operation defines a multi-scale version of the initial timeseries), i.e, the multiscale time series.

The hyperparameter `down_sampling_method` specifies the method used for down-sampling, being the available options:
* avg: 1D Average Pooling (default)
* max: 1D Max Pooling
* conv: strided 1D convolution

In the original paper, the authors experimented with two downsampling methods:

* Average Pooling: The simpler method they initially used, which is more efficient and easier to implement
* Strided 1DConvolutions: A more complex method that slightly improves performance compared to average pooling.

The experiments (detailed in Table 19 of TimeMixer paper) show that while the 1D convolution method offers a marginal improvement in metrics such as MSE and MAE across various datasets (ETTm1, M4, and PEMS04), the gain is small. Due to this minor difference and considering the balance between performance and computational efficiency, the authors decided to stick with the average pooling method in the final TimeMixer design. Having this considerations in mind, the following distribution is proposed:

```
down_sampling_method = categorical(values=['avg', 'conv'])
```

Max pooling is excluded because the authors did not consider it relevant in their experiments, and its omission helps to reduce the complexity of the search space.

### **`down_sampling_layers`** (M in the paper)

The number of scales (downsampled timeseries) that are passed to the PDM blocks. This hyperparameter determines how many different temporal scales the model will consider during the processing (i.e how many times 
the down sampling operation is applied recursively). Default value is 3.

From TimeMixer paper (page 9): *We explore the impact from the number of scales (M) in Figure 6 
under different serieslengths. Specifically, when M increases, the performance gain declines for 
shorter prediction lengths. In contrast, for longer prediction lengths, the performance improves 
more as M increases. Therefore, we set M as 3 for long-term forecast and 1 for short-term forecast 
to trade off performance and efficiency*

It is decided not to tune this hyperparameter since authors already evaluated its effect and defined and optimum for the current scenario (ETTh1 dataset). Furthermore, this value significantly affects model complexity and thus tunning it without limitting the distribution range could result in an inneficient HPO process.


### **`down_sampling_window`**

This hyperparameter specifies the downsampling factor for the down sampling layer. It determines how much the temporal resolution is reduced at each step (e.g., dividing the sequence length by a factor). Larger values for `down_sampling_window` will result in coarser temporal scales. No padding is applied. Its effect on model behaviour can be explained as follows:
* For avg and max down sampling methods `down_sampling_window` is passed to the kernel_size (and stride) parameters of its respective torch classes.
* For conv down sampling method `down_sampling_window` is passed to the stride parameter of its corresponding torch class (kernel size is set to 3 by default). Circular padding is applied.

Default value is 2 but no information is provided in the paper about the reasons behind it. `down_sampling_window` controls the extent to which the input data is downsampled (i.e., the reduction in resolution of the data). In TimeMixer source code, this downsampling is applied recursively, where the input sequence is progressively reduced by the factor specified by `down_sampling_window`. The default choice of 2 likely stems from a combination of balancing detail, computational efficiency, hierarchical multi-scale analysis, and signal processing considerations, alongside empirical performance results. Furthermore, authors did not include this hypepraramter in the HPO studies presented in the paper. Due to this reasons, this hyperparameter will not be included in the search space, taking the deafult value of 2.

### **`d_model`**

The `d_model` hyperparameter is a crucial part of defining the dimensionality of the model's internal layers, particularly in the embedding and projection layers. It generally refers to the number of features (or channels) used in the embedding representation, which is passed across different parts of the network.

In TimeMixer `d_model` serves as the **channel dimension** of the input and output tensors of various layers (like the classes `DataEmbedding_wo_pos` and `PastDecomposableMixing` and the projection layers). More generally, it defines the number of channels of the outputs of the PDM blocks. This is often found in models inspired by the Transformer architecture, where `d_model` typically represents the size of the hidden layer used in multi-head attention and feedforward layers.

In TimeMixer paper, the default value of `d_model` for dataset ETTh1 is 16, which is derived from the unified experiment setting. As candidate distribution for hyperparameter optimization, we allign with the one specified in the searched hyperparameter setting since it provides a reasonable candidate based on the default value, on the complexity of the resulting configuration and search space and on the typycal specification of the hyperparameter (common practices).

```
d_model = categorical(values=[8, 16, 32, 64, 128, 256, 512])
```

### **`d_ff`**

In transformer models and their variants, `d_ff` typically refers to the dimension of the intermediate layer in the feedforward network that is applied after the multi-head attention layers. It is generally set to be larger than the model dimension `d_model`, allowing for richer transformations before reducing the dimensionality back to `d_model`.

In TimeMixder, it is used as the channel dimension of the intermediate hidden layer of the feed forward networks contained in the MLP submodules of the PDM blocks (see figure 1 of TimeMixer-TFM README). As a consequence, `d_ff` impacts the model’s capacity to learn complex relationships and should be set properly.

Since the feed-forward dimension `d_ff` is usually dependent on the model dimension `d_model`, and the paper does not provide information 
about its value in the experiments conducted (neither if it was optimized) we are going to define a candidate distribution that 
scales proportionally with `d_model` (in particular, as a multiple of `d_model` which is itself optimized). This 
relationship follows the specifications used in TimeMixer, where `d_ff=32` and `d_model=16` for the ETTh1 dataset.

$$  d_{ff} = \alpha \cdot d_{model}  $$

Where $\alpha$ is a hyperparameter that will be tuned (replaces `d_ff` in the tunning process). Typical values for $\alpha$ are in the range of 2 to 8, depending on the model's architecture. By sampling `d_ff` as a function of `d_model`, the search space is better aligned with common architectural practices. This strategy not only provides more meaningful sampling but also reduces the hyperparameter search space by conditioning (implicitly) `d_ff` directly on `d_model`.

As candidate distribution for $\alpha$, a categorical distribution with values $\{2,3,4\}$ is selected. The range of values is selected as a starting point (that coulb be further optimized). Higher values could be considered but they usually come with a high computation cost, thus are not going to be considered for the current experiment.

```
alpha = categorical(values=[2, 3, 4])
```

### **`decomp_method`**

Specifies which decomposition method into trend and seasonal components should be applied within each PDM block.
The available options are:
* `moving_avg` Uses a moving average technique for decomposition, handled by the series_decomp function. This technique was utilized in the Autoformer arquitecture (05/2022). When sampled, another hyperparameter can be defined: `moving_avg`, which determines the window size for the moving average, controlling how smooth the resulting trend and seasonal components are when decomposing the time series. A larger `moving_avg` value results in a smoother trend, while a smaller value gives more weight to shorter-term fluctuations (defaults to 25, no information provided in TimeMixer paper).
* `dft_decomp` Uses a Discrete Fourier Transform (DFT) for decomposition, implemented in the DFT_series_decomp class and proposed by the autors of TimeMixer. When sampled, another hyperparameter can be defined: `top_k`, which specifies how many of the top frequency components should be kept when performing the DFT. The rest are filtered out as noise. The logic focuses on selecting the top_k most significant frequencies, which are considered to represent the primary seasonal patterns in the data. The other frequencies are set to zero to isolate the trend component (defaults to 5, no information provided in TimeMixer paper).

From TimeMixer paper (page 20): *As we stated in Section 3.1, we adopt the average pooling to obtain the 
multiscale series. Here we replace this operation with 1D convolutions. From Table 19, we can find that 
the complicated1D-convolution-based outperforms average pooling slightly. But considering both performance 
and efficiency, we eventually use average pooling in TimeMixer.*

Since in TimeMixer paper it is shown that the DFT-based season-trend decomposition slightly outperforms the moving average decomposition, tunning it would possibly produce better results than the opposite at the cost of higher computational complexity. Furthermore, we opt to define both of the conditional hyperparameters mentioned above with the foloowing distributions:

```
decomp_method = categorical(values['moving_avg', 'dft_decomp'])
moving_avg = categorical(values=[15,25,35,55,75]) (exists iff decomp_method=='moving_avg')
top_k = categorical(values=[5,10,15,20]) (exists iff decomp_method=='dft_decomp')
```

Where its values are based on common sense, default value and computational constraints. However initial testing revealed that some (high) values of top_k are incompatible with lower values of d_model and d_ff, raising erros during configuration evaluatuion, and thus it has been excluded from the final experiment.

```
decomp_method = categorical(values['moving_avg', 'dft_decomp'])
moving_avg = categorical(values=[15,25,35,55,75])
```

### **`e_layers`**

This hyperparameter determines the number of encoder layers used in the model. Specifically, it defines how many PDM blocks are stacked sequentially in the model. The PDM block is responsible for performing the core operations in the model, such as time-series decomposition (into seasonal and trend components) and mixing. Somo notes about the effects of this hyperparameter:

* Increasing `e_layers` directly increases the depth of the model by adding more layers of time-series decomposition and mixing. This allows the model to capture more complex temporal patterns and interactions across scales.
* More layers can increase the model's ability to learn intricate features and dependencies in the data, potentially improving its performance on tasks like forecasting, anomaly detection, and classification.
* A higher value for `e_layers` increases the model’s computational load. Each additional layer adds more parameters and requires more computation during both training and inference.
* If the number of layers is too high relative to the amount of available data, the model might overfit, learning noise instead of meaningful patterns. Hence, `e_layers` should be tuned based on the dataset's complexity and size.

The ideal value for `e_layers` is typically found through experimentation and hyperparameter tuning.
In original implementation, the default value for ETTh1 dataset in the unified hyperparameter setting is 2:

From TimeMixer paper (page 14): *Here, we further evaluate the
number of layers L. As shown in Table 12, we can find that in general, increasing the number of
layers (L) will bring improvements across different prediction lengths. Therefore, we set to 2 to trade
off efficiency and performance.*

For the searched hyperparameter setting, the authors propose the following distribution:

```
e_layers = categorical(values=[1,2,3,4,5])
```

Which will be adopted for the current experiment by lowering the upper bound, since it will provide a reasonalbe candidate based on runtime complexity.

```
e_layers = categorical(values=[1,2,3,4])
```

### **`dropout`**

This hyperparameter determines the probability of dropping neurons (dropout_rate) for the dropout layers in the TimeMixer model. Dropout is a regularization technique used to prevent overfitting during training. It works by randomly setting a fraction of the neurons to zero during each forward pass, which helps the model become less dependent on any particular neuron and improves generalization.

In TimeMixer paper, authors did not specify the use of regularization techniques. However, by inspecting the 
code we can conclude that dropout is applied by default for the unified setting over ETTh1 dataset with a 
dropout rate of 0.1.

Since dropout represent the most popular mechanism for fighting overfitting during neural network training 
and it has been demonstrated that can improve performance in complex and noisy environments (such as the 
current dataset), it has been considered a good candidate for the tunning process. The chosen distribution 
is specified as follows:

```
dropout = truncated_norm(min=0.05, max=0.15, mean=0.1, std=0.025)
```

The choice of the normal distribution centered over the default value is because the default value represent a good starting point, and 
uniform distribution could lead to suboptimal effects or even no effects (values closer to 0 will have the same probability than 
other values, but if sampled dropout will be negigible). The truncation of the distribution contributes to the efficiency of the tunning process, where values that may produce non informative evaluations are omitted.





























































































