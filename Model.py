import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import time
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load your dataset into 'df' (assuming it contains the necessary columns)
# ...
df = pd.read_csv('final.csv')

# Separate features (X) and target variable (y)
X = df[['Price per Unit', 'Shelf Life (days)/ Warranty', 'Product Score', 'comp_1', 'comp_2', 'comp_3']]
y = df['Lag Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Lasso Regression model
lasso_model = Lasso()
lasso_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lasso = lasso_model.predict(X_test)

# Evaluate the Lasso Regression model using mean squared error
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
# print(f"Mean Squared Error (Lasso Regression): {mse_lasso}")

# Function to get a random user agent
def get_random_user_agent():
    ua = UserAgent()
    return ua.random

# Function to scrape data from Amazon and return a DataFrame
# Function to scrape data from Amazon and return a DataFrame
def scrape_amazon_data(search_item, lower_bound, upper_bound):
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run Chrome in headless mode
    chrome_options.add_argument(f'user-agent={get_random_user_agent()}')

    try:
        # Initialize the Chrome browser
        driver = webdriver.Chrome(options=chrome_options)

        # Navigate to Amazon's search page
        base_url = 'https://www.amazon.in'
        search_url = f'{base_url}/s?k={search_item}'
        driver.get(search_url)

        # Wait for the content to load (you might need to adjust the wait time)
        time.sleep(5)

        # Access the page source
        page_source = driver.page_source

        # Parse HTML content using BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')

        # Extract product names and prices
        product_elements = soup.find_all('span', class_=['a-size-base-plus a-color-base a-text-normal', 'a-size-medium a-color-base a-text-normal'])
        price_elements = soup.find_all('span', class_='a-price-whole')

        # Filter products based on price
        products = []
        prices = []

        for product, price in zip(product_elements, price_elements):
            product_name = product.get_text()
            price_text = price.get_text()
            cleaned_price = ''.join(filter(str.isdigit, price_text))
            
            try:
                price_float = float(cleaned_price)
                if lower_bound <= price_float <= upper_bound:
                    products.append(product_name)
                    prices.append(price_float)
            except ValueError:
                print(f"Unable to convert price to float: {price_text}")

        # Return top 3 product names and prices
        return list(zip(products[:3], prices[:3]))

    except Exception as e:
        print(f"Amazon Error: {e}")
        return []

    finally:
        # Close the browser window
        driver.quit()


# User input for product details
product_name = input("Enter the product name: ")
price_per_unit = float(input("Enter the price per unit: "))
shelf_life_warranty = float(input("Enter the shelf life (days) or warranty: "))
product_score = float(input("Enter the product score: "))

# Scrape Amazon data
amazon_results = scrape_amazon_data(product_name, 0.85*price_per_unit, 1.7*price_per_unit)

if amazon_results:
    # Convert scraped prices to floats
    for i, (product, price) in enumerate(amazon_results, start=1):
        print(f"Competitor {i}: {product} - Price: {price}")
    
        # Extract only the prices from the Amazon results
        amazon_prices = []
        for _, price in amazon_results:
            try:
                # Convert the cleaned price to float and append to the list
                amazon_prices.append(float(price))
            except ValueError:
                print(f"Unable to convert price to float: {price}")


    # Create DataFrame
    data = {
        'Price per Unit': [price_per_unit],
        'Shelf Life (days)/ Warranty': [shelf_life_warranty],
        'Product Score': [product_score],
        'comp_1': [amazon_prices[0]],
        'comp_2': [amazon_prices[1]],
        'comp_3': [amazon_prices[2]]
    }

    df_input = pd.DataFrame(data)

    # Model Prediction
    predicted_price_lasso = lasso_model.predict(df_input)

    # Display the predicted price
    print(f"Predicted Price: {predicted_price_lasso[0]}")

else:
    print("Unable to scrape Amazon data. Please check your input and try again.")
