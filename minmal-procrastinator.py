import hopsworks
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

project = hopsworks.login()
fs = project.get_feature_store()

# Generate and encode data in one go
n = 1000
data = np.column_stack((
    np.random.randint(1, 11, n),  # procrastination_level
    np.random.randint(0, 8, n),   # coffee_cups
    np.random.choice([0, 1], n),  # last_minute_panic
    LabelEncoder().fit_transform(np.random.choice(['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces'], n)),
    np.random.rand(n)  # task_completion (target)
))

# Create feature group and view in one step
fg = fs.get_or_create_feature_group('procrast', version=1, primary_key=['procrastination_level', 'coffee_cups', 'last_minute_panic', 'zodiac_sign'], description='Procrastination features')
fg.insert(data)
fv = fs.get_or_create_feature_view('procrast', version=1, query=fg.select_all(), labels=['task_completion'])

# Train and deploy model in one go
X, y = fv.get_training_data(1.0)[0][:, :-1], fv.get_training_data(1.0)[0][:, -1]
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
mr = project.get_model_registry()
model_metrics = {'mse': ((y - model.predict(X))**2).mean()}
saved_model = mr.sklearn.create_model('procrast', metrics=model_metrics).save('procrast_model')
deployment = saved_model.deploy('procrastinator3001')

# Make a prediction
print(f"Prediction: {deployment.predict({'instances': [[7, 1, 0, 4]]})['predictions'][0]:.2%}")