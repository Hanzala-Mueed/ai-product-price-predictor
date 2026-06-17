from pricer.items import Item
from pricer.preprocessor import ProductPreprocessor


item = Item(
    title="Wireless Bluetooth Headphones",
    category="Electronics",
    brand="SoundPro",
    description="Comfortable wireless headphones with Bluetooth 5.3, noise reduction, built-in microphone, and 40-hour battery life.",
    features="Foldable design, USB-C charging, soft ear cushions, lightweight body.",
)

preprocessor = ProductPreprocessor()
processed_item = preprocessor.preprocess_item(item)

print(processed_item.summary)