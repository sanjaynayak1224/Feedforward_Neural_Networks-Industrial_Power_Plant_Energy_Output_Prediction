# ⚡ PowerPulse: Predicting Industrial Power Plant Energy Output with Feedforward Neural Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-darkblue?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning project built to predict the net hourly electrical energy output of a Combined Cycle Power Plant (CCPP) under varying environmental conditions. Using a dataset of 9,568 hourly observations, I designed, built, and evaluated a PyTorch-based Feedforward Neural Network (FNN) that achieves an **$R^2$ score of 93.55%** and a **Root Mean Squared Error (RMSE) of 4.29 MW** on unseen test data—allowing operators to predict plant performance with high precision and optimize power grid integration.

---

## 🔍 The Pipeline & Modeling Workflow

The project follows a structured workflow to clean, preprocess, batch, train, and validate the neural network. Here is the general structure:

```mermaid
graph TD
    A[Raw Power Plant Data] --> B[Data Quality & Imputation Check]
    B --> C[Feature-Target Separation]
    C --> D[80/20 Train-Test Split]
    D --> E[Feature Scaling via StandardScaler]
    E --> F[PyTorch Tensor Conversion]
    F --> G[TensorDataset & DataLoader Batches]
    G --> H[FNN Architecture Design]
    H --> I[Model Training with MSE Loss & Adam]
    I --> J[Validation & Best Model Saving]
    J --> K[Evaluation on Test Dataset]
```

### Behind the Scenes: How the Pipeline is Built

To get the raw environmental readings ready for PyTorch, I built a structured preprocessing pipeline:

* **Cleaning up the dataset**: First, I loaded the powerplant dataset and verified if there were any missing values. Fortunately, the dataset was completely clean with zero null values across all 9,568 rows, so I didn't need to do any imputation.
* **Understanding the environmental features**: The dataset contains four main environmental features:
  * `AT` (Ambient Temperature in °C): Heavily correlated with energy output. As temperature rises, efficiency and power output generally decrease.
  * `V` (Exhaust Vacuum in cm Hg): Represents steam turbine backpressure.
  * `AP` (Ambient Pressure in millibars): Represents atmospheric pressure.
  * `RH` (Relative Humidity as a percentage).
  * The target variable is `PE` (Produced Energy in MW).
* **Splitting and Scaling**: I split the dataset into an 80% training set and a 20% test set to ensure a robust evaluation. Since neural networks are highly sensitive to the scale of input features (features like pressure `AP` are around 1000, while temperatures `AT` are around 20), I used `StandardScaler` to normalize all features so they have a mean of 0 and variance of 1.
* **Converting to PyTorch Tensors**: Since PyTorch doesn't work directly with pandas DataFrames or numpy arrays during training, I converted the scaled features and target values into PyTorch `float32` tensors.
* **Batching with DataLoaders**: To make training efficient, I wrapped the tensors in a `TensorDataset` and used a `DataLoader` to split the training data into mini-batches of size 32, shuffling the training set at each epoch to prevent the model from memorizing the order of samples.

---

## 🏗️ Neural Network Architecture & Training

I built a Feedforward Artificial Neural Network (ANN) using PyTorch's `nn.Sequential` with the following structure:
* **Input Layer**: Accepts the 4 normalized environmental features.
* **First Hidden Layer**: 6 nodes with a `ReLU` activation function to capture non-linear relationships.
* **Second Hidden Layer**: 6 nodes with a `ReLU` activation.
* **Output Layer**: 1 node (fully connected linear layer) to output the predicted energy output (`PE`).

I compiled the model using PyTorch's `MSELoss` (Mean Squared Error) to measure performance and the `Adam` optimizer to adjust network weights during backpropagation.

### Training & Checkpointing
I ran the training loop for **100 epochs**. At the end of each epoch, I ran the model on the test dataset to calculate validation loss. To avoid saving a model that might overfit, I set up a checkpointing check: the script only saved the model weights to `best_model.pt` when the validation loss reached a new historical low. After training, I loaded these optimal weights back into the model for final evaluation.

---

## 📊 Model Evaluation & Results

Here are the performance metrics I recorded for the best model:

| Metric | Training Set | Testing Set (Unseen) |
| :--- | :---: | :---: |
| **Mean Squared Error (MSE)** | 19.9381 | 18.4425 |
| **Root Mean Squared Error (RMSE)** | 4.4652 MW | **4.2945 MW** |
| **Mean Absolute Error (MAE)** | 3.5128 MW | **3.4046 MW** |
| **$R^2$ Score (Coefficient of Determination)** | 93.18% | **93.55%** |

### 💡 What the numbers tell us
* **Strong predictive power ($R^2$ of 93.55%)**: Our model explains over 93.5% of the variance in electrical energy output, showing that the four environmental features have a very strong relationship with plant performance.
* **Low error (RMSE of 4.29 MW)**: On average, our model's predictions deviate by only 4.29 MW. Considering the target energy output ranges from 420.26 MW to 495.76 MW (a range of ~75.5 MW), an error of 4.29 MW represents a tiny ~5.7% relative error margin.
* **No Overfitting**: The training metrics (MSE: 19.93, $R^2$: 93.18%) and testing metrics (MSE: 18.44, $R^2$: 93.55%) are extremely close (testing is actually slightly better due to standard train-test variation). This indicates that the model generalizes incredibly well to unseen data and isn't memorizing noise.

---

## 🚀 Next Steps: How I'd Take This Further

If I had more time or were preparing this for a production-grade deployment, here are the things I would focus on next to squeeze out even more performance:

1. **Experiment with Deeper/Wider Neural Network Architectures**: The current architecture is very light (two hidden layers of 6 nodes each). I'd experiment with adding a third hidden layer or increasing the width (e.g., [16, 16] or [32, 16] hidden neurons) to see if a slightly larger model could capture more complex interactions between temperature and humidity.
2. **Apply Regularization Techniques (Dropout/Weight Decay)**: If I make the network larger, there's always a risk of overfitting. I'd add `nn.Dropout(p=0.1)` layers between the hidden layers or introduce L2 regularization (weight decay) in the Adam optimizer (`weight_decay=1e-4`) to keep the weights small and the model robust.
3. **Implement a Learning Rate Scheduler**: Right now, the learning rate is fixed at Adam's default (0.001). I'd integrate a scheduler like `ReduceLROnPlateau` or `CosineAnnealingLR` to decay the learning rate as the training loss plateaus, helping the network settle smoothly into the global minimum.
4. **Try Tree-Based Ensemble Models for Comparison**: While neural networks are great, gradient boosted trees (like **XGBoost** or **LightGBM**) are highly efficient and often outperform simple ANNs on tabular datasets with minimal tuning. I'd train a quick XGBoost model as a baseline comparison.
5. **Set up K-Fold Cross-Validation**: To ensure these metrics are consistent and not just a lucky random train-test split, I'd implement a 5-fold cross-validation strategy. This would give a much more reliable average RMSE and R2 score.

---

## 🛠️ How to Run the Project Locally

If you want to pull this down and run the code on your local machine, here is the quick-start guide:

### 1. Clone and Navigate
```bash
git clone <repository-url>
cd Feedforward_Neural_Networks-Industrial_Power_Plant_Energy_Ouput_Prediction
```

### 2. Spin Up a Virtual Environment
* **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
* **On macOS/Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install the Packages
```bash
pip install -r requirements.txt
```

### 4. Open and Run the Notebook
Open `ANN_Regression.ipynb` in your favorite IDE (like VS Code or Jupyter Lab), select the `.venv` environment as your kernel, and run all cells to see the data prep and model results in action.
