# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier implemented using scikit-learn. It is trained to predict whether an individual's income exceeds $50K/year based on demographic and employment attributes from the UCI Census Income dataset. The model uses one-hot encoding for categorical features and a label binarizer for the target variable.

## Intended Use

The model is intended for educational and demonstration purposes, specifically for illustrating the deployment of a scalable machine learning pipeline using FastAPI. It is not intended for production or high-stakes decision-making. The model can be used to predict income categories for individuals based on their census data.

## Training Data

The model was trained on the UCI Census Income dataset (also known as "Adult" dataset), which contains demographic and employment information for over 30,000 individuals. The dataset includes features such as workclass, education, marital status, occupation, relationship, race, sex, and native country.

## Evaluation Data

A random 20% split of the original dataset was used as the test set for evaluation. The same preprocessing steps (one-hot encoding and label binarization) were applied to the evaluation data as to the training data.

## Metrics

The following metrics were used to evaluate model performance:
- **Precision**
- **Recall**
- **F1 Score (beta=1)**

Precision: 0.7419 | Recall: 0.6384 | F1: 0.6863

## Ethical Considerations

- The model may reflect biases present in the original census data, including those related to gender, race, and nationality.
- Predictions should not be used for real-world decision-making without careful consideration of fairness, accountability, and transparency.
- The model does not account for changes in economic or social conditions since the data was collected.

## Caveats and Recommendations

- The model is for demonstration and educational use only.
- Performance may vary for subgroups; users should review slice metrics for fairness analysis.
- Further tuning and validation are recommended before any real-world deployment.
- Users should not use this model for employment, credit, or other high-stakes decisions.