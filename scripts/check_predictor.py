from pricer.items import Item
from pricer.predictor import PricePredictor


item = Item(
    title="Wireless Bluetooth Headphones",
    category="Electronics",
    brand="SoundPro",
    summary=(
        "Title: Wireless Bluetooth Headphones\n"
        "Category: Electronics\n"
        "Brand: SoundPro\n"
        "Description: Comfortable wireless headphones with Bluetooth 5.3, noise reduction, built-in microphone, and 40-hour battery life.\n"
        "Details: Foldable design, USB-C charging, soft ear cushions, and lightweight body."
    ),
)

predictor = PricePredictor()
price = predictor.predict(item)

print(f"Predicted price: ${price:.2f}")