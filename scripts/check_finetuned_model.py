from pricer.items import Item
from pricer.predictor import PricePredictor

item = Item(
    title="Samsung Galaxy A32 4G Smartphone, 64GB, Black",
    category="Electronics",
    summary=(
        "Title: Samsung Galaxy A32 4G Smartphone, 64GB, Black\n"
        "Category: Electronics\n"
        "Brand: Samsung\n"
        "Description: 6.5 inch HD+ smartphone with 48MP quad camera, 5000mAh battery, octa-core processor, 4GB RAM, and 64GB storage.\n"
        "Details: Unlocked consumer smartphone in black with mid-range Android features."
    ),
)

predictor = PricePredictor()
price = predictor.predict(item)

print(f"Predicted price: ${price:.2f}")