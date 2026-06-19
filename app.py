import streamlit as st

from pricer.items import Item
from pricer.preprocessor import ProductPreprocessor
from pricer.predictor import PricePredictor
from config.settings import APP_NAME, APP_VERSION, OLLAMA_MODEL
from config.logging_config import setup_logging

logger = setup_logging()


def build_item(title, category, brand, description, features):
    try:
        return Item(
            title=title,
            category=category,
            brand=brand,
            description=description,
            features=features,
        )
    except Exception as e:
        logger.exception("Failed to build product item from form data.")
        raise RuntimeError(f"Failed to build product item: {e}") from e


def predict_price(item):
    try:
        preprocessor = ProductPreprocessor()
        predictor = PricePredictor()

        processed_item = preprocessor.preprocess_item(item)
        predicted_price = predictor.predict(processed_item)

        return processed_item, predicted_price

    except Exception as e:
        logger.exception("Failed to predict price from Streamlit app.")
        raise RuntimeError(f"Failed to predict price: {e}") from e


def main():
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="💰",
        layout="centered",
    )

    st.title("💰 AI Product Price Predictor")
    st.caption(f"Version {APP_VERSION} | Model: {OLLAMA_MODEL}")

    st.write(
        "Enter product details below. The app will summarize the product with Ollama "
        "and estimate a realistic marketplace price."
    )

    with st.form("price_prediction_form"):
        title = st.text_input("Product Title", placeholder="Wireless Bluetooth Headphones")
        category = st.text_input("Category", placeholder="Electronics")
        brand = st.text_input("Brand", placeholder="SoundPro")
        description = st.text_area(
            "Description",
            placeholder="Comfortable wireless headphones with Bluetooth, microphone, and long battery life.",
        )
        features = st.text_area(
            "Features",
            placeholder="Foldable design, USB-C charging, soft ear cushions, lightweight body.",
        )

        submitted = st.form_submit_button("Predict Price")

    if submitted:
        if not title.strip() or not category.strip():
            st.error("Product title and category are required.")
            return

        try:
            with st.spinner("Analyzing product and predicting price..."):
                item = build_item(
                    title=title,
                    category=category,
                    brand=brand,
                    description=description,
                    features=features,
                )

                processed_item, predicted_price = predict_price(item)

            st.success("Prediction completed.")

            st.metric(
                label="Estimated Product Price",
                value=f"${predicted_price:,.2f}",
            )

            with st.expander("View AI Product Summary"):
                st.text(processed_item.summary)

            with st.expander("View Raw Product Text"):
                st.text(processed_item.full)

        except Exception as e:
            st.error("Prediction failed. Please check if Ollama is running.")
            st.exception(e)


if __name__ == "__main__":
    main()