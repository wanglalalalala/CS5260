"""
Generate a realistic sample product dataset for development and testing.
When a real Kaggle/Amazon dataset is available, replace this with the actual
data-loading logic in build_index.py.
"""

import json
import pathlib

PRODUCTS = [
    # ── Consumer Electronics ──────────────────────────────────────
    {
        "id": "CE001",
        "title": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
        "brand": "Sony",
        "price": 348.00,
        "category": "Electronics",
        "subcategory": "Headphones",
        "description": "Industry-leading noise cancellation with Auto NC Optimizer. 30-hour battery life, crystal clear hands-free calling, and multipoint connectivity.",
        "specifications": {"driver_size": "30mm", "battery_life": "30h", "weight": "250g", "bluetooth": "5.2", "noise_cancellation": "Yes"},
        "rating": 4.7,
        "review_summary": "Excellent noise cancellation and comfort. Slightly expensive but worth every penny for frequent travelers.",
    },
    {
        "id": "CE002",
        "title": "Apple AirPods Pro 2nd Generation",
        "brand": "Apple",
        "price": 249.00,
        "category": "Electronics",
        "subcategory": "Earbuds",
        "description": "Active Noise Cancellation, Adaptive Transparency, and Personalized Spatial Audio with dynamic head tracking.",
        "specifications": {"driver_size": "custom", "battery_life": "6h (30h with case)", "weight": "5.3g each", "bluetooth": "5.3", "noise_cancellation": "Yes"},
        "rating": 4.6,
        "review_summary": "Great sound quality and seamless integration with Apple ecosystem. Battery case is a game-changer.",
    },
    {
        "id": "CE003",
        "title": "Samsung Galaxy Tab S9 FE 10.9-inch Tablet",
        "brand": "Samsung",
        "price": 449.00,
        "category": "Electronics",
        "subcategory": "Tablets",
        "description": "10.9-inch display, Exynos 1380 processor, 6GB RAM, 128GB storage, S Pen included, IP68 water resistance.",
        "specifications": {"screen_size": "10.9in", "processor": "Exynos 1380", "ram": "6GB", "storage": "128GB", "battery": "8000mAh"},
        "rating": 4.4,
        "review_summary": "Great value tablet with S Pen. Good for note-taking and media consumption. Not the fastest for heavy gaming.",
    },
    {
        "id": "CE004",
        "title": "Anker PowerCore 26800mAh Portable Charger",
        "brand": "Anker",
        "price": 65.99,
        "category": "Electronics",
        "subcategory": "Power Banks",
        "description": "Ultra-high capacity 26800mAh portable charger with dual USB output. Charges iPhone 14 over 6 times.",
        "specifications": {"capacity": "26800mAh", "output_ports": "2x USB-A", "input": "Micro USB", "weight": "495g", "fast_charge": "No"},
        "rating": 4.5,
        "review_summary": "Massive battery capacity. A bit heavy but perfect for long trips. Reliable and well-built.",
    },
    {
        "id": "CE005",
        "title": "Kindle Paperwhite 11th Generation E-Reader",
        "brand": "Amazon",
        "price": 139.99,
        "category": "Electronics",
        "subcategory": "E-Readers",
        "description": "6.8-inch glare-free display, adjustable warm light, waterproof IPX8, 16GB storage, 10-week battery life.",
        "specifications": {"screen_size": "6.8in", "storage": "16GB", "waterproof": "IPX8", "battery_life": "10 weeks", "weight": "205g"},
        "rating": 4.7,
        "review_summary": "Best e-reader on the market. Warm light is easy on the eyes. Perfect gift for book lovers.",
    },
    {
        "id": "CE006",
        "title": "JBL Charge 5 Portable Bluetooth Speaker",
        "brand": "JBL",
        "price": 179.95,
        "category": "Electronics",
        "subcategory": "Speakers",
        "description": "Powerful JBL Original Pro Sound, IP67 dust and waterproof, 20-hour playtime, built-in power bank.",
        "specifications": {"battery_life": "20h", "waterproof": "IP67", "weight": "960g", "bluetooth": "5.1", "power_bank": "Yes"},
        "rating": 4.6,
        "review_summary": "Amazing sound for its size. Very durable and waterproof. Great for outdoor parties.",
    },
    # ── Health & Wellness ─────────────────────────────────────────
    {
        "id": "HW001",
        "title": "Renpho Shiatsu Neck and Back Massager with Heat",
        "brand": "Renpho",
        "price": 59.99,
        "category": "Health",
        "subcategory": "Massagers",
        "description": "Deep-kneading Shiatsu massage nodes with optional heat. Ergonomic design for neck, shoulders, and back. Adjustable intensity.",
        "specifications": {"massage_type": "Shiatsu", "heat": "Yes", "nodes": "8", "power": "AC adapter + car adapter", "weight": "1.6kg"},
        "rating": 4.4,
        "review_summary": "Relieves neck pain effectively. Great gift for elderly parents. A bit noisy at highest setting.",
    },
    {
        "id": "HW002",
        "title": "Theragun Mini 2.0 Handheld Percussion Massager",
        "brand": "Therabody",
        "price": 199.00,
        "category": "Health",
        "subcategory": "Massagers",
        "description": "Ultra-portable percussion therapy device. QuietForce Technology, 3 speeds, 150-minute battery life. Weighs only 1.36 lbs.",
        "specifications": {"speed_settings": "3", "battery_life": "150min", "weight": "0.6kg", "noise_level": "quiet", "attachments": "1"},
        "rating": 4.5,
        "review_summary": "Compact and powerful. Perfect for post-workout recovery. Premium build quality.",
    },
    {
        "id": "HW003",
        "title": "Fitbit Charge 6 Advanced Fitness Tracker",
        "brand": "Fitbit",
        "price": 159.95,
        "category": "Health",
        "subcategory": "Fitness Trackers",
        "description": "Built-in GPS, heart rate monitoring, stress management score, sleep tracking, 7-day battery, Google integration.",
        "specifications": {"gps": "Built-in", "heart_rate": "Yes", "battery_life": "7 days", "waterproof": "50m", "weight": "37g"},
        "rating": 4.3,
        "review_summary": "Accurate health tracking. Google Wallet integration is handy. Band could be more comfortable.",
    },
    {
        "id": "HW004",
        "title": "Comfier Heated Neck Wrap with Vibration",
        "brand": "Comfier",
        "price": 39.99,
        "category": "Health",
        "subcategory": "Massagers",
        "description": "Weighted heated neck wrap with vibration massage. 3 heat levels, 2 vibration modes. USB rechargeable, cordless design.",
        "specifications": {"heat_levels": "3", "vibration_modes": "2", "battery": "USB rechargeable", "weight": "0.5kg", "cordless": "Yes"},
        "rating": 4.3,
        "review_summary": "Very soothing for neck stiffness. Lightweight and portable. Good value for money.",
    },
    {
        "id": "HW005",
        "title": "Omron Evolv Wireless Upper Arm Blood Pressure Monitor",
        "brand": "Omron",
        "price": 74.99,
        "category": "Health",
        "subcategory": "Health Monitors",
        "description": "Clinically validated, tubeless, wireless design. Bluetooth connectivity, stores up to 100 readings, irregular heartbeat detection.",
        "specifications": {"connectivity": "Bluetooth", "memory": "100 readings", "cuff_size": "9-17in", "power": "4 AAA batteries", "weight": "240g"},
        "rating": 4.4,
        "review_summary": "Very accurate readings. Sleek tubeless design is convenient for elderly users. App works well.",
    },
    # ── Travel & Gift ─────────────────────────────────────────────
    {
        "id": "TG001",
        "title": "Osprey Daylite Plus 20L Daypack",
        "brand": "Osprey",
        "price": 75.00,
        "category": "Travel",
        "subcategory": "Backpacks",
        "description": "Lightweight 20L daypack with padded laptop sleeve, mesh back panel, and multiple pockets. Perfect for daily commute or light hiking.",
        "specifications": {"capacity": "20L", "laptop_sleeve": "15in", "weight": "560g", "material": "Recycled nylon", "waterproof": "DWR coating"},
        "rating": 4.6,
        "review_summary": "Comfortable, well-organized, and lightweight. Excellent quality for the price. Great everyday carry.",
    },
    {
        "id": "TG002",
        "title": "TUMI Alpha 3 Compact Laptop Brief",
        "brand": "TUMI",
        "price": 495.00,
        "category": "Travel",
        "subcategory": "Laptop Bags",
        "description": "Premium ballistic nylon laptop brief. Fits up to 15-inch laptop. Multiple organizational pockets, leather trim, TUMI Tracer.",
        "specifications": {"laptop_size": "15in", "material": "Ballistic nylon", "weight": "1.2kg", "dimensions": "30x41x10cm", "warranty": "Lifetime"},
        "rating": 4.7,
        "review_summary": "Exceptional build quality. Professional look. Expensive but extremely durable and functional.",
    },
    {
        "id": "TG003",
        "title": "Travel Adapter Universal All-in-One Worldwide",
        "brand": "EPICKA",
        "price": 22.99,
        "category": "Travel",
        "subcategory": "Travel Accessories",
        "description": "Universal travel adapter with 4 USB ports and 1 USB-C. Works in 150+ countries. Built-in safety fuse.",
        "specifications": {"usb_ports": "4x USB-A + 1x USB-C", "countries": "150+", "voltage": "100-250V", "weight": "195g", "safety": "Fuse protected"},
        "rating": 4.5,
        "review_summary": "Essential travel companion. Compact and reliable. USB-C port is a nice bonus.",
    },
    {
        "id": "TG004",
        "title": "Samsonite Freeform 24-inch Hardside Luggage",
        "brand": "Samsonite",
        "price": 189.99,
        "category": "Travel",
        "subcategory": "Luggage",
        "description": "Lightweight polycarbonate hardside spinner. Double wheels, TSA lock, expandable, full interior organization.",
        "specifications": {"size": "24in (checked)", "material": "Polycarbonate", "weight": "3.9kg", "expandable": "Yes", "lock": "TSA approved"},
        "rating": 4.5,
        "review_summary": "Lightweight for its size. Smooth-rolling wheels. Survived multiple flights without a scratch.",
    },
    {
        "id": "TG005",
        "title": "Tile Mate Bluetooth Tracker 4-Pack",
        "brand": "Tile",
        "price": 54.99,
        "category": "Travel",
        "subcategory": "Travel Accessories",
        "description": "Bluetooth tracker for keys, bags, and more. 76m range, water-resistant, 3-year battery, works with Alexa and Google.",
        "specifications": {"range": "76m", "battery_life": "3 years", "waterproof": "IP67", "compatibility": "iOS/Android", "weight": "7.5g each"},
        "rating": 4.2,
        "review_summary": "Handy for finding lost items. Range could be better. Subscription needed for premium features.",
    },
    {
        "id": "TG006",
        "title": "Ember Temperature Control Smart Mug 2 - 14oz",
        "brand": "Ember",
        "price": 149.95,
        "category": "Travel",
        "subcategory": "Travel Accessories",
        "description": "App-controlled heated coffee mug. Choose exact drinking temperature. 1.5-hour battery, charging coaster included.",
        "specifications": {"capacity": "14oz", "temp_range": "120-145°F", "battery_life": "1.5h", "charging": "Coaster included", "app": "Yes"},
        "rating": 4.3,
        "review_summary": "Keeps coffee at perfect temperature. Great gift for coffee lovers. Short battery life is the only downside.",
    },
]


def get_sample_products() -> list[dict]:
    return PRODUCTS


def save_to_json(output_path: str = "data/products.json"):
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(PRODUCTS, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(PRODUCTS)} products -> {out}")


if __name__ == "__main__":
    save_to_json()
