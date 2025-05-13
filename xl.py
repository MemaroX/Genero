import pandas as pd
from faker import Faker
import random

# Initialize faker for realistic test data
fake = Faker()

# Generate sample review data
def generate_reviews(num_reviews=50):
    reviews = []
    sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
    
    for _ in range(num_reviews):
        # Randomly select sentiment
        sentiment = random.choice(sentiments)
        
        # Generate appropriate review based on sentiment
        if sentiment == 'POSITIVE':
            review = fake.sentence(nb_words=10, variable_nb_words=True) + " " + random.choice([
                "Great product!", "Highly recommend!", "Love it!", 
                "Works perfectly!", "Excellent quality!"
            ])
        elif sentiment == 'NEGATIVE':
            review = fake.sentence(nb_words=10, variable_nb_words=True) + " " + random.choice([
                "Terrible experience!", "Would not recommend.", 
                "Poor quality.", "Broken on arrival.", "Waste of money."
            ])
        else:
            review = fake.sentence(nb_words=15, variable_nb_words=True)
        
        reviews.append({
            'Review_ID': fake.uuid4(),
            'Product': random.choice(['Smartphone', 'Laptop', 'Headphones', 'Smartwatch', 'Tablet']),
            'Customer': fake.name(),
            'Rating': random.randint(1, 5),
            'Reviews': review,
            'Date': fake.date_between(start_date='-1y', end_date='today')
        })
    
    return pd.DataFrame(reviews)

# Create and save the test file
test_data = generate_reviews(100)
test_data.to_excel('test_reviews.xlsx', index=False)

print("Test file 'test_reviews.xlsx' generated successfully!")
print(f"Contains {len(test_data)} sample reviews.")