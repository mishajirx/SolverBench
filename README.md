# MysterySolver: Neural Network Architecture Research

This is a research project on comparing different Neural Network architectures to evaluate the effect of various architectural decisions on the model's performance on a classification problem (205 input features, 5 target classes).  

By systematically experimenting with network depth, activation functions, regularization techniques, and normalizations, this project evaluates the impact of each architectural decision on model performance.

## Project Structure and Models Explored

The research is organized into several Jupyter Notebooks, each testing a specific variation of a deep learning model using PyTorch:

*   **`basic_nn_solver copy.ipynb` (Model 0):** A baseline 3-layer Multi-Layer Perceptron (MLP) using standard ReLU activations.
*   **`four_layers_nn_solver.ipynb` (Model 1):** A deeper 4-layer MLP baseline.
*   **`dropout_three_layers_nn_solver.ipynb` (Model 2):** The 3-layer baseline augmented with Dropout (p=0.2) to combat overfitting.
*   **`norm_three_layers_nn_solver.ipynb` (Model 3):** The 3-layer baseline augmented with 1D Batch Normalization.
*   **`gelu_three_layers_nn_solver.ipynb` (Model 4):** The 3-layer baseline where ReLU is replaced with the GELU activation function.
*   **`cnnish_solver.ipynb` (Model 5):** An experimental Convolutional Neural Network (CNN) approach applied to the feature set.
*   **`altogether_solver.ipynb` (Model 6):** The culmination of the research, combining all beneficial techniques (4 layers, BatchNorm1d, Dropout, and GELU).

The `experiment_logs.csv` tracks the training and validation loss/accuracy across epochs for every model. The `plot_performances.ipynb` notebook provides a visual comparison of these learning curves using `matplotlib` and `seaborn`.

## Key Findings

Analyzing the performance logs yields several interesting insights into the dataset and architecture choices:

1.  **Baseline Performance:** The standard 3-layer ReLU network achieved a respectable **80.25%** validation accuracy.
2.  **The Perils of Unregularized Depth:** Increasing the network depth to 4 layers without adding regularization or normalization actually *hurt* performance (dropping to **74.38%**), likely due to overfitting on the training set.
3.  **Activation Functions Matter:** Swapping standard ReLU for GELU (Gaussian Error Linear Unit) yielded one of the most significant single-technique improvements, raising accuracy to **83.69%**.
4.  **Regularization:** Applying Dropout (0.2) to the baseline provided a marginal but consistent improvement over the unregularized baseline (**80.75%** vs 80.25%).
5.  **CNN Suitability:** The experimental CNN approach underperformed the MLP baselines (**77.50%**), suggesting that the 205 input features do not possess strong spatial locality that a CNN typically exploits.
6.  **Synergy is Key (The Winning Model):** The final `altogether_solver` combined all the techniques: a deeper 4-layer architecture stabilized by Batch Normalization, regularized by Dropout, and utilizing GELU activations. This model significantly outperformed all others, achieving a peak validation accuracy of **91.00%** with the lowest validation loss (0.2801).

## Conclusion
This research demonstrates that while blindly adding network layers can degrade performance, thoughtfully combining depth with modern architectural elements (BatchNorm, GELU, and Dropout) can unlock vastly superior predictive capabilities.

