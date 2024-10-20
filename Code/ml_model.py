#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Importing essential libraies.
from data_imports import *


# In[3]:


class MODEL:
    def __init__(self, input_data, target_var, model):
        """
        Initialize the Model with input data and target variable.

        :param input_data: DataFrame containing the input data.
        :param target_var: Name of the target variable column.
        """
        self.input_data = input_data
        self.target_var = target_var
        self.model = model
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_test_metrics = None
        
    def prepare_data(self):
        """
        Prepare data by separating features and target variable,
        and splitting into training and testing sets.
        """
        df_not_missing = self.input_data.dropna(subset=[self.target_var])
        X = df_not_missing.drop(columns=[self.target_var])
        y = df_not_missing[self.target_var]
        
        # Split the data into train and test subsets.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=1
        )
    
    def train_model(self):
        """
        Train the model.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data has not been prepared. Call prepare_data() first.")
        self.model.fit(self.X_train, self.y_train)
    
    def predict(self, new_data):
        """
        Predict target values for new data using the trained model.

        :param new_data: DataFrame containing the new input data.
        :return: Array of predicted values.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train_model() first.")
        return self.model.predict(new_data)
    
    def fill_missing_values(self):
        """
        Predict and fill missing values in the original input data.
        
        :return: DataFrame with missing values filled.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data has not been prepared. Call prepare_data() first.")
        if self.model is None:
            raise ValueError("Model has not been trained. Call train_model() first.")
        
        df_missing = self.input_data[self.input_data[self.target_var].isnull()]
        if not df_missing.empty:
            X_missing = df_missing.drop(columns=[self.target_var])
            self.input_data.loc[self.input_data[self.target_var].isnull(), self.target_var] = self.model.predict(X_missing)
        
        return self.input_data
    
    def evaluate_model(self):
        """
        Evaluate the model using regression metrics for both training and testing sets.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data has not been prepared. Call prepare_data() first.")
        if self.model is None:
            raise ValueError("Model has not been trained. Call train_model() first.")
        
        # Predictions on the training and test set.
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        # Calculating regression metrics for the training set.
        mse_train = mean_squared_error(self.y_train, y_train_pred)
        rmse_train = mse_train ** 0.5
        mae_train = mean_absolute_error(self.y_train, y_train_pred)
        r2_train = r2_score(self.y_train, y_train_pred)

        # Calculating regression metrics for the test set.
        mse_test = mean_squared_error(self.y_test, y_test_pred)
        rmse_test = mse_test ** 0.5
        mae_test = mean_absolute_error(self.y_test, y_test_pred)
        r2_test = r2_score(self.y_test, y_test_pred)
        
        self.train_test_metrics = {
            'train': {'mse': mse_train, 'rmse': rmse_train, 'mae': mae_train, 'r2': r2_train},
            'test': {'mse': mse_test, 'rmse': rmse_test, 'mae': mae_test, 'r2': r2_test}
        }

        # Print training and testing set regression metrics.
        print(f'Training Set - MSE: {mse_train:8.5f}, RMSE: {rmse_train:8.5f}, MAE: {mae_train:8.5f}, R2: {r2_train:8.5f}')
        print(f'Testing Set  - MSE: {mse_test:8.5f}, RMSE: {rmse_test:8.5f}, MAE: {mae_test:8.5f}, R2: {r2_test:8.5f}')
    
    def model_metrics(self):
        """
        Model evaluation metrics, used for plots.
        
        :return: Dictionary of training and testing metrics.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data has not been prepared. Call prepare_data() first.")
        if self.model is None:
            raise ValueError("Model has not been trained. Call train_model() first.")

        return self.train_test_metrics
    
    def shap_summary(self, file_name, title):
        """
        Summarise the top SHAP features for the model.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data has not been prepared. Call prepare_data() first.")
        if self.model is None:
            raise ValueError("Model has not been trained. Call train_model() first.")
            
        # SHAP analysis
        explainer = shap.Explainer(self.model)
        self.shap_values = explainer.shap_values(self.X_test)
        shap.summary_plot(self.shap_values, self.X_test, show = False)
        
        plt.title(title)
        plt.savefig(f'/Users/callumwilson/Documents/GitHub/ML-Project/Output/{file_name}.png', bbox_inches='tight')
        plt.show()
    
    def shap_top_features(self, top_n):
        """
        Calculate and return the top N features based on SHAP values.

        :param top_n: Number of top features to return.
        :return: DataFrame with the top N features and the target variable.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data has not been prepared. Call prepare_data() first.")
        if self.model is None:
            raise ValueError("Model has not been trained. Call train_model() first.")

        # Calculate mean absolute SHAP values for each feature
        shap_importance = np.abs(self.shap_values).mean(axis=0)
        top_n_indices = np.argsort(shap_importance)[-top_n:][::-1]  # Indices of the top N features
        top_n_features = self.X_test.columns[top_n_indices]
        
        # Return DataFrame with top N features and target variable
        return self.input_data[top_n_features.to_list() + [self.target_var]]
    
    def feature_importance(self, features, file_name):
        
        feature_range = range(1, features + 1)

        metrics = {
        'MSE': {'train': [], 'test': []},
        'RMSE': {'train': [], 'test': []},
        'MAE': {'train': [], 'test': []},
        'R2': {'train': [], 'test': []}
        }
        
        for feature in feature_range:
            # Get the top `feature` number of SHAP features
            selected_features = self.shap_top_features(feature)  # Get the DataFrame with top `feature` features

            # Create a new MODEL instance with the selected features
            new_model = MODEL(input_data=selected_features, target_var=self.target_var, model=self.model)

            # Prepare the data for the new model
            new_model.prepare_data()

            # Train the new model
            new_model.train_model()

            # Evaluate the new model
            new_model.evaluate_model()
            
            # Get the metrics from the evaluation and append them to the lists
            metrics['MSE']['train'].append(new_model.train_test_metrics['train']['mse'])
            metrics['MSE']['test'].append(new_model.train_test_metrics['test']['mse'])

            metrics['RMSE']['train'].append(new_model.train_test_metrics['train']['rmse'])
            metrics['RMSE']['test'].append(new_model.train_test_metrics['test']['rmse'])

            metrics['MAE']['train'].append(new_model.train_test_metrics['train']['mae'])
            metrics['MAE']['test'].append(new_model.train_test_metrics['test']['mae'])

            metrics['R2']['train'].append(new_model.train_test_metrics['train']['r2'])
            metrics['R2']['test'].append(new_model.train_test_metrics['test']['r2'])

        # Plotting the metrics
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Flatten the axes array to easily loop through them
        axs = axs.flatten()
        
        # Metric names for the legend and titles
        metric_names = ['MSE', 'RMSE', 'MAE', 'R2']

        for i, metric in enumerate(metrics.values()):
            axs[i].plot(feature_range, metric['train'], marker='o', label=f'Training {list(metrics)[i]}')
            axs[i].plot(feature_range, metric['test'], marker='o', label=f'Testing {list(metrics)[i]}')
            axs[i].set_title(f'{list(metrics)[i]} Over Top {features} Features')
            axs[i].set_xlabel('Number of Features')
            axs[i].set_ylabel(list(metrics)[i])
            axs[i].legend()
            axs[i].grid(True)
            axs[i].set_xticks(range(1, features + 1))
            
        # Adjust layout to avoid overlap
        plt.tight_layout()
        plt.savefig(f'/Users/callumwilson/Documents/GitHub/ML-Project/Output/{file_name}.png', bbox_inches='tight')
        plt.show()

