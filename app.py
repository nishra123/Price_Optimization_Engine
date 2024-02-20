import streamlit as st
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

from fake_useragent.errors import FakeUserAgentError

import random

# Function to generate a random user agent string
def get_random_user_agent():
    platform = random.choice(['Macintosh', 'Windows', 'X11', 'Linux'])
    browser = random.choice(['chrome', 'firefox', 'safari'])
    if browser == 'chrome':
        webkit_version = random.randint(500, 599)
        version = f"{random.randint(0, 99)}.0.{random.randint(0, 9999)}.{random.randint(0, 99)}"
    elif browser == 'firefox':
        webkit_version = random.randint(500, 599)
        version = f"{random.randint(1, 99)}.0.{random.randint(1, 99)}"
    elif browser == 'safari':
        webkit_version = random.randint(500, 599)
        version = f"{random.randint(1, 99)}.0.{random.randint(1, 99)}"
    else:
        webkit_version = random.randint(500, 599)
        version = f"{random.randint(0, 99)}.0.{random.randint(0, 9999)}.{random.randint(0, 99)}"

    if platform == 'Windows':
        platform = f"Windows NT {random.choice(['5.0', '5.1', '5.2', '6.0', '6.1', '6.2', '6.3'])}; Win64; x64"
    elif platform == 'Macintosh':
        platform = 'Macintosh; Intel Mac OS X 10_{random.randint(10, 15)}_{random.randint(0, 9)}'
    elif platform == 'Linux':
        platform = 'X11; Linux x86_64'

    if browser == 'chrome':
        return f"Mozilla/5.0 ({platform}) AppleWebKit/{webkit_version}.0 (KHTML, like Gecko) Chrome/{version} Safari/{webkit_version}.0"
    elif browser == 'firefox':
        return f"Mozilla/5.0 ({platform}; rv:{version}) Gecko/20100101 Firefox/{version}"
    elif browser == 'safari':
        return f"Mozilla/5.0 ({platform}) AppleWebKit/{webkit_version}.36 (KHTML, like Gecko) Version/{version} Safari/{webkit_version}.36"
    else:
        return f"Mozilla/5.0 (compatible; MSIE {random.randint(5, 9)}.0; {platform}; Trident/{random.randint(3, 5)}.{random.randint(0, 1)})"



# Function to scrape data from Amazon and return a DataFrame
def scrape_amazon_data(search_item, lower_bound, upper_bound):
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run Chrome in headless mode
    chrome_options.add_argument(f'user-agent={get_random_user_agent()}')
    driver = None

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
        if driver:
        # Close the browser window
            driver.quit()

def main():
    st.title("Product Retail Price Optimization")

    # User input for product details
    product_name = st.text_input("Enter the Product Name:")
    price_per_unit = st.number_input("Enter the Price per Unit:")
    shelf_life_warranty = st.number_input("Enter the Shelf Life (in days) or Warranty:")
    product_score = st.number_input("Enter the Product Score:")

    if st.button("Predict Price"):
        # Scrape Amazon data
        amazon_results = scrape_amazon_data(product_name, 0.85*price_per_unit, 1.7*price_per_unit)

        if amazon_results:
            # Convert scraped prices to floats
            amazon_prices = [price for _, price in amazon_results]

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

            # Display Competitor Prices
            st.subheader("Product Competitor Prices:")
            for i, (product, price) in enumerate(amazon_results, start=1):
                st.write(f"Competitor {i}: {product}")
                st.write(f"Price: ₹{price}")

            # Display the predicted price rounded off to the nearest integer
            st.subheader("Optimized Product Price based on Competitor Prices:")
            st.write(f"₹{round(predicted_price_lasso[0])}")

        else:
            st.write("Unable to scrape Amazon data. Please check your input and try again.")

if __name__ == "__main__":
    main()
