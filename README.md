# Xpired
Xpired is a machine learning project designed to help you keep track of the expiration dates of your food products. By using Optical Character Recognition (OCR) technology, Xpired captures images of expiration dates from food packaging, processes this information, and uploads it to a database. The application then sends reminders as products approach their expiration dates, ensuring that you consume your food items in a timely manner in hopes to reduce food waste. In addition the LLM (GPT3.5T) is integrated to provide intelligent recipe suggestions based on available ingredients, helping users make the most of their pantry.

# Features:
- OCR for Expiration Dates: Utilizes an OCR model to scan expiration dates food product packaging.
- Database Integration: Stores the captured expiration dates and product information in a database.
- Expiration Reminders: Sends notifications to users when products are nearing their expiration dates.
- Utilizes LLMs to recommend recipes based on available ingredients..

# Future Enhancements:
Enhanced OCR Accuracy – Improve text recognition for different packaging styles and fonts.
Mobile App Integration – Enable users to scan food items via cameras and mobile application.
Smart Inventory Management – Predict consumption patterns and optimize grocery shopping.

# Getting Started:
To get started with Xpired, follow these steps:

Clone the repository:

- git clone https://github.com/Stephen1eung/xpired.git
- cd xpired

Install dependencies:

- pip install -r requirements.txt

Ensure you have a PostgreSQL database set up.
Configure your database connection settings in the config.py file.
