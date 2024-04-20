from sklearn.preprocessing import OneHotEncoder

# Example data
data = [['Male', 1], ['Female', 3], ['Female', 2]]

# Create encoder instance
encoder = OneHotEncoder(handle_unknown='ignore')

# Fit and transform data
encoded_data = encoder.fit_transform(data)

print(encoded_data.toarray())
