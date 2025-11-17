# Model Card

## Model Details
- **Model Type**: Random Forest Classifier
- **Model Version**: 1.0
- **Date**: November 2025
- **Framework**: scikit-learn
- **Parameters**: 
  - n_estimators: 100
  - max_depth: 20
  - random_state: 42

## Intended Use
This model predicts whether an individual's income exceeds $50K/year based on census data. It is intended for educational purposes and demonstration of MLOps practices.

**Primary Use Cases**:
- Educational demonstration of ML pipeline deployment
- Understanding income prediction based on demographic features

**Out-of-Scope Uses**:
- This model should NOT be used for making actual hiring, lending, or other decisions that impact individuals
- Should not be used for production systems without further validation

## Training Data
- **Dataset**: 1994 US Census Bureau database
- **Size**: ~32,000 records (80% used for training)
- **Features**: 14 features including age, education, occupation, work class, marital status, race, sex, and others

**Categorical Features**:
- workclass, education, marital-status, occupation, relationship, race, sex, native-country

**Numerical Features**:
- age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week

## Evaluation Data
- **Size**: ~6,500 records (20% held out for testing)
- **Same distribution** as training data from the census dataset

## Metrics
The model was evaluated using the following metrics on the test dataset:

- **Precision**: 0.7915
- **Recall**: 0.6136  
- **F1 Score**: 0.6913

**Interpretation**:
- The model correctly identifies high earners (>50K) 79% of the time (precision)
- The model captures 61% of actual high earners (recall)
- The balanced F1 score of 0.69 indicates reasonable overall performance

## Ethical Considerations
**Potential Biases**:
- The model uses demographic features like race and sex which may perpetuate historical biases
- Census data from 1994 may not reflect current demographic and economic realities
- The binary income threshold of $50K is arbitrary and may not be meaningful across all regions

**Fairness Concerns**:
- Performance should be evaluated across demographic slices to ensure equitable performance
- Model predictions should not be used to make decisions that could discriminate against protected groups

## Caveats and Recommendations
**Limitations**:
- Model trained on 1994 data - economic conditions have changed significantly
- Binary classification may oversimplify complex income distributions
- Limited to features available in census data

**Recommendations**:
- Use only for educational and demonstration purposes
- Evaluate model performance on specific demographic slices before any deployment
- Consider more recent data sources for any production use
- Implement human oversight for any decision-making processes
- Regular retraining would be necessary with updated data

**Technical Notes**:
- Model performance varies across different categorical feature values (see slice_output.txt)
- Some demographic groups have limited representation in training data
- Feature engineering and hyperparameter tuning could improve performance
