# Xpired
### Background and motivation:
Xpired is a machine learning project designed to help you keep track of the expiration dates of your food products. By using Optical Character Recognition (OCR) technology, Xpired captures images of expiration dates from food packaging, processes this information, and uploads it to a database. The application then sends reminders as products approach their expiration dates, ensuring that you consume your food items in a timely manner in hopes to reduce food waste. In addition I've integrated a LLM (GPT 4o-mini) to provide intelligent recipe suggestions based on available ingredients, helping users make the most of their pantry.

# Features:
- OCR for Expiration Dates: Utilizes an OCR model to scan expiration dates food product packaging.
- Database Integration: Stores the captured expiration dates and product information in a database.
- Expiration Reminders: Notifies to users when products are nearing their expiration dates.
- LLM Integration: Utilizes LLMs to recommend recipes based on available ingredients.

# Future Enhancements:
Enhanced OCR Accuracy – Improve text recognition for different packaging styles and fonts.
Mobile App Integration – Enable users to scan food items via cameras and mobile application.
Smart Inventory Management – Predict consumption patterns and optimize grocery shopping.

# Repo Architecture:
Xpired/
│── data/              # Sample images for testing OCR
│── models/            # Saved ML models
│── src/               # Source code
│   │── ocr.py
│   │── database.py
│   │── notifier.py
│   │── app.py
│── tests/             # Basic unit tests
│── requirements.txt   # Python dependencies
│── README.md          # Project documentation
│── .env               # Environment variables

# Getting Started:
To get started with Xpired, follow these steps:

Clone the repository:

- git clone https://github.com/Stephen1eung/xpired.git
- cd xpired

Install dependencies:

- pip install -r requirements.txt

Set up environment variables:

- Copy .env.example to .env
- Fill in the values for the environment variables

## Database Setup

Xpired uses a local SQLite database to manage your food inventory and expiration tracking. The database automatically handles:

# TODO: investigate if we need this many tables or if we can just have a master table 
### Core Functionality
- **Food Item Storage**: Track products with names, brands, categories, and barcodes
- **Expiration Management**: Monitor expiration dates with quantities and storage locations
- **Smart Notifications**: Automatic alerts for items approaching expiration
- **Recipe Integration**: Provides ingredient lists to the LLM for recipe suggestions
- **OCR Results**: Stores text extraction results from food packaging images

### Quick Start
```python
from db.db import initialize_database

# Initialize database (creates xpired.db automatically)
db = initialize_database()
```

### Database Schema
The system uses five main tables:
- **`food_items`** - Product information and metadata
- **`expiration_dates`** - Expiration tracking with status management
- **`notifications`** - Expiration alerts and reminders
- **`ocr_results`** - YOLOv8 OCR processing results
- **`user_preferences`** - System settings and user preferences

### Integration Points
- **OCR Pipeline**: Automatically stores YOLOv8 text extraction results
- **LLM Service**: Provides ingredient lists for GPT-4o-mini recipe generation
- **Notification System**: Triggers alerts based on expiration proximity
- **Status Tracking**: Monitors food consumption to reduce waste
