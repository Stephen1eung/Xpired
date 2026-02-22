"""SQLite database utilities for Xpired."""

import sqlite3
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XpiredDatabase:
    """
    Local SQLite database for the Xpired food tracking system.
    Manages food items, expiration dates, and user notifications.
    """

    def __init__(self, db_path: str = "xpired.db"):
        """
        Initialize the database connection and create tables if they don't exist.

        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        self.connect()
        self.create_tables()

    def connect(self) -> None:
        """Establish connection to the SQLite database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            logger.info("Connected to database: %s", self.db_path)
        except sqlite3.Error as e:
            logger.error("Error connecting to database: %s", e)
            raise

    def create_tables(self) -> None:
        """Create all necessary tables for the Xpired system."""
        cursor = self.connection.cursor()

        try:
            # Food Items table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS food_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    brand TEXT,
                    category TEXT,
                    barcode TEXT,
                    image_path TEXT,
                    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT
                )
            ''')

            # Expiration Dates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS expiration_dates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    food_item_id INTEGER NOT NULL,
                    expiration_date DATE NOT NULL,
                    quantity INTEGER DEFAULT 1,
                    location TEXT DEFAULT 'pantry',
                    status TEXT DEFAULT 'active' CHECK(status IN ('active', 'consumed', 'expired', 'discarded')),
                    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    date_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (food_item_id) REFERENCES food_items (id) ON DELETE CASCADE
                )
            ''')

            # Notifications table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    food_item_id INTEGER NOT NULL,
                    expiration_id INTEGER NOT NULL,
                    notification_type TEXT NOT NULL CHECK(notification_type IN ('warning', 'urgent', 'expired')),
                    message TEXT NOT NULL,
                    sent_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_read BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (food_item_id) REFERENCES food_items (id) ON DELETE CASCADE,
                    FOREIGN KEY (expiration_id) REFERENCES expiration_dates (id) ON DELETE CASCADE
                )
            ''')

            # OCR Results table (for tracking OCR processing)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ocr_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    extracted_text TEXT,
                    confidence_score REAL,
                    processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    food_item_id INTEGER,
                    FOREIGN KEY (food_item_id) REFERENCES food_items (id) ON DELETE SET NULL
                )
            ''')

            # User Preferences table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    preference_key TEXT UNIQUE NOT NULL,
                    preference_value TEXT NOT NULL,
                    date_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes for better performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_expiration_date
                ON expiration_dates(expiration_date)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_food_item_name
                ON food_items(name)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_notification_type
                ON notifications(notification_type)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_status
                ON expiration_dates(status)
            ''')

            self.connection.commit()
            logger.info("Database tables created successfully")

        except sqlite3.Error as e:
            logger.error("Error creating tables: %s", e)
            self.connection.rollback()
            raise

    def add_food_item(
        self,
        name: str,
        *,
        brand: str = None,
        category: str = None,
        barcode: str = None,
        image_path: str = None,
        notes: str = None,
    ) -> int:
        """
        Add a new food item to the database.

        Args:
            name (str): Name of the food item
            brand (str): Brand name (optional)
            category (str): Food category (optional)
            barcode (str): Barcode if available (optional)
            image_path (str): Path to the food item image (optional)
            notes (str): Additional notes (optional)

        Returns:
            int: ID of the newly created food item
        """
        cursor = self.connection.cursor()

        try:
            cursor.execute('''
                INSERT INTO food_items (name, brand, category, barcode, image_path, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (name, brand, category, barcode, image_path, notes))

            self.connection.commit()
            food_item_id = cursor.lastrowid
            logger.info("Added food item: %s (ID: %s)", name, food_item_id)
            return food_item_id

        except sqlite3.Error as e:
            logger.error("Error adding food item: %s", e)
            self.connection.rollback()
            raise

    def add_expiration_date(self, food_item_id: int, expiration_date: str, 
                           quantity: int = 1, location: str = 'pantry') -> int:
        """
        Add an expiration date for a food item.

        Args:
            food_item_id (int): ID of the food item
            expiration_date (str): Expiration date in YYYY-MM-DD format
            quantity (int): Quantity of items with this expiration date
            location (str): Storage location

        Returns:
            int: ID of the newly created expiration record
        """
        cursor = self.connection.cursor()

        try:
            cursor.execute('''
                INSERT INTO expiration_dates (food_item_id, expiration_date, quantity, location)
                VALUES (?, ?, ?, ?)
            ''', (food_item_id, expiration_date, quantity, location))

            self.connection.commit()
            expiration_id = cursor.lastrowid
            logger.info(
                "Added expiration date for food item %s: %s",
                food_item_id,
                expiration_date,
            )
            return expiration_id

        except sqlite3.Error as e:
            logger.error("Error adding expiration date: %s", e)
            self.connection.rollback()
            raise

    def get_expiring_items(self, days_ahead: int = 7) -> List[Dict]:
        """
        Get items that are expiring within the specified number of days.

        Args:
            days_ahead (int): Number of days to look ahead for expiring items

        Returns:
            List[Dict]: List of expiring items with their details
        """
        cursor = self.connection.cursor()

        try:
            future_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

            cursor.execute('''
                SELECT 
                    fi.id as food_id,
                    fi.name,
                    fi.brand,
                    fi.category,
                    ed.id as expiration_id,
                    ed.expiration_date,
                    ed.quantity,
                    ed.location,
                    ed.status,
                    CASE 
                        WHEN ed.expiration_date < date('now') THEN 'expired'
                        WHEN ed.expiration_date <= date('now', '+3 days') THEN 'urgent'
                        ELSE 'warning'
                    END as urgency_level
                FROM food_items fi
                JOIN expiration_dates ed ON fi.id = ed.food_item_id
                WHERE ed.expiration_date <= ? 
                AND ed.status = 'active'
                ORDER BY ed.expiration_date ASC
            ''', (future_date,))

            results = cursor.fetchall()
            return [dict(row) for row in results]

        except sqlite3.Error as e:
            logger.error("Error getting expiring items: %s", e)
            raise

    def get_ingredients_for_recipes(self) -> List[str]:
        """
        Get a list of active food items that can be used for recipe generation.

        Returns:
            List[str]: List of food item names
        """
        cursor = self.connection.cursor()

        try:
            cursor.execute('''
                SELECT DISTINCT fi.name
                FROM food_items fi
                JOIN expiration_dates ed ON fi.id = ed.food_item_id
                WHERE ed.status = 'active'
                AND ed.expiration_date >= date('now')
                ORDER BY fi.name
            ''')

            results = cursor.fetchall()
            return [row['name'] for row in results]

        except sqlite3.Error as e:
            logger.error("Error getting ingredients: %s", e)
            raise

    def add_notification(self, food_item_id: int, expiration_id: int, 
                        notification_type: str, message: str) -> int:
        """
        Add a notification record.

        Args:
            food_item_id (int): ID of the food item
            expiration_id (int): ID of the expiration record
            notification_type (str): Type of notification ('warning', 'urgent', 'expired')
            message (str): Notification message

        Returns:
            int: ID of the newly created notification
        """
        cursor = self.connection.cursor()

        try:
            cursor.execute('''
                INSERT INTO notifications (food_item_id, expiration_id, notification_type, message)
                VALUES (?, ?, ?, ?)
            ''', (food_item_id, expiration_id, notification_type, message))

            self.connection.commit()
            notification_id = cursor.lastrowid
            logger.info("Added notification for food item %s", food_item_id)
            return notification_id

        except sqlite3.Error as e:
            logger.error("Error adding notification: %s", e)
            self.connection.rollback()
            raise

    def update_item_status(self, expiration_id: int, status: str) -> None:
        """
        Update the status of an expiration record.

        Args:
            expiration_id (int): ID of the expiration record
            status (str): New status ('active', 'consumed', 'expired', 'discarded')
        """
        cursor = self.connection.cursor()

        try:
            cursor.execute('''
                UPDATE expiration_dates 
                SET status = ?, date_updated = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, expiration_id))

            self.connection.commit()

            logger.info(
                "Updated expiration record %s status to %s",
                expiration_id,
                status,
            )

        except sqlite3.Error as e:
            logger.error("Error updating item status: %s", e)
            self.connection.rollback()
            raise

    def get_all_food_items(self) -> List[Dict]:
        """
        Get all food items with their expiration information.

        Returns:
            List[Dict]: List of all food items with expiration details
        """
        cursor = self.connection.cursor()

        try:
            cursor.execute('''
                SELECT 
                    fi.id as food_id,
                    fi.name,
                    fi.brand,
                    fi.category,
                    fi.date_added,
                    COUNT(ed.id) as total_items,
                    MIN(ed.expiration_date) as earliest_expiration,
                    SUM(CASE WHEN ed.status = 'active' THEN ed.quantity ELSE 0 END) as active_quantity
                FROM food_items fi
                LEFT JOIN expiration_dates ed ON fi.id = ed.food_item_id
                GROUP BY fi.id, fi.name, fi.brand, fi.category, fi.date_added
                ORDER BY fi.name
            ''')

            results = cursor.fetchall()
            return [dict(row) for row in results]

        except sqlite3.Error as e:
            logger.error("Error getting all food items: %s", e)
            raise

    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


def initialize_database(db_path: str = "xpired.db") -> XpiredDatabase:
    """
    Initialize and return a database instance.

    Args:
        db_path (str): Path to the SQLite database file

    Returns:
        XpiredDatabase: Initialized database instance
    """
    return XpiredDatabase(db_path)


# Example usage and testing
if __name__ == "__main__":
    # Initialize database
    db = initialize_database()

    try:
        # Add sample data
        print("Adding sample food items...")

        # Add some food items
        milk_id = db.add_food_item("Milk", brand="Organic Valley", category="Dairy", notes="2% milk")
        bread_id = db.add_food_item("Whole Wheat Bread", brand="Dave's Killer Bread", category="Bakery")
        eggs_id = db.add_food_item("Eggs", brand="Happy Eggs", category="Dairy", notes="Free range")

        # Add expiration dates
        db.add_expiration_date(milk_id, "2024-01-15", quantity=1, location="refrigerator")
        db.add_expiration_date(bread_id, "2024-01-10", quantity=1, location="pantry")
        db.add_expiration_date(eggs_id, "2024-01-20", quantity=12, location="refrigerator")

        # Get expiring items
        print("\nItems expiring in the next 7 days:")
        expiring_items = db.get_expiring_items(7)
        for item in expiring_items:
            print(
                f"- {item['name']} ({item['brand']}) expires on {item['expiration_date']} "
                f"- {item['urgency_level']}"
            )

        # Get ingredients for recipes
        print("\nAvailable ingredients for recipes:")
        ingredients = db.get_ingredients_for_recipes()
        for ingredient in ingredients:
            print(f"- {ingredient}")

        # Get all food items
        print("\nAll food items in database:")
        all_items = db.get_all_food_items()
        for item in all_items:
            print(
                f"- {item['name']} (Active: {item['active_quantity']}, "
                f"Earliest expiration: {item['earliest_expiration']})"
            )

        print("\nDatabase setup completed successfully!")

    except (sqlite3.Error, ValueError) as e:
        print(f"Error during database operations: {e}")

    finally:
        db.close()