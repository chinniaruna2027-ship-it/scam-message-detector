
import streamlit as st
import joblib
import re
from urllib.parse import urlparse
from scipy.sparse import csr_matrix, hstack

bundle = joblib.load("scam_detector_bundle.pkl")

model = bundle['model']
vectorizer = bundle['vectorizer']
shorteners = bundle['shorteners']
suspicious_words = bundle['suspicious_words']

def extract_link_features(message):
    urls = re.findall(
        r'https?://\S+|www\.\S+',
        message.lower()
    )

    short_count = 0
    suspicious_count = 0

    for url in urls:
        url = url.rstrip('.,!?;:)')

        if url.startswith('www.'):
            url = 'http://' + url

        domain = urlparse(url).netloc

        if any(s in domain for s in shorteners):
            short_count += 1

        if any(word in url for word in suspicious_words):
            suspicious_count += 1

    return [
        len(urls),
        short_count,
        suspicious_count
    ]

st.set_page_config(
    page_title="Scam Message Detector",
    page_icon="🛡️"
)

st.title("🛡️ Scam Message Detector")
st.write("Detect different types of suspicious messages.")

message = st.text_area(
    "Paste your message here:"
)

if st.button("Detect Message"):

    if not message.strip():
        st.warning("Please enter a message.")

    else:
        text_features = vectorizer.transform([message])

        link_features = csr_matrix(
            [extract_link_features(message)]
        )

        final_features = hstack([
            text_features,
            link_features
        ])

        prediction = model.predict(final_features)[0]

        probabilities = model.predict_proba(
            final_features
        )[0]

        confidence = max(probabilities) * 100

        st.subheader("Prediction")

        if prediction == "genuine":
            st.success("Genuine message")
        else:
            st.error("Predicted category: " + prediction)

        st.metric(
            "Model confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Message analysis")

        features = extract_link_features(message)

        st.write("Number of links:", features[0])
        st.write("Shortened URLs:", features[1])
        st.write("Suspicious link indicators:", features[2])

        found_words = [
            word for word in suspicious_words
            if word in message.lower()
        ]

        st.write("Suspicious words:", found_words)

        st.info(
            "This is an AI prediction, not a guarantee. "
            "Always verify messages through official channels."
        )
