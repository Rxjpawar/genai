import streamlit as st
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_vectors",
    embedding=embedding_model
)

st.title("Product Report Generator")

query = st.text_input("Enter your product query")

if st.button("Generate Report") and query:
    results = vector_db.similarity_search(
        query=query,
        k=5
    )

    context = "\n\n".join([r.page_content for r in results])

    system_prompt = f"""
    You are a product analysis assistant. Create a structured product report using only the provided context.

    Report must include:
    - Product Name
    - Category
    - Price
    - Description Summary
    - Key Features
    - Brand
    - Product URL (if present)
    - Additional Notes

    Do not add information that is not present.

    Context:
    {context}

    Example :
        Product Report
        Here are the product reports based on the provided context for "iphone" related products:

        Product Report 1

        Product Name: AdroitZ Premium Phone Socket Holder For Apple iPhone 4S[36]
        Category: Mobiles & Accessories >> Mobile Accessories >> Car Accessories >> Mobile Holders >> AdroitZ Mobile Holders
        Price: Rs. 164.0 (Discounted)
        Description Summary: A premium phone socket holder designed for Apple iPhone 4S, featuring an adjustable clip that fits mobile phones up to 5.7 inches (max opening 86mm). It has an elastic fixation design for easy installation on the steer wheel without extra tools, ensuring safety with its compact fit. The device is embedded with a soft silicone pad to protect the phone from scratches. Available in Red and Black.
        Key Features:
        Type: Anti-slip
        Orientation: Portrait
        Mount Type Dashboard: Yes
        Adjustable clips (max opening 86mm, fits up to 5.7 inches mobile phones)
        Elastic fixation design
        Easy installation
        High compactness with steer wheel for safety
        Soft silicone pad for phone protection
        Brand: AdroitZ
        Product URL: http://www.flipkart.com/adroitz-premium-phone-socket-holder-apple-iphone-4s-36/p/itmefh5cmyamvdwc?pid=ACCEFH5CVVZ9YHAW
        Additional Notes: Original retail price was Rs. 599.0. Product comes in Red, Black color.
        Product Report 2

        Product Name: GANPATI WHOLSALER Apple Iphone 6/6 Plus Apple Iphone 6/6 Plus USB USB Cable
        Category: Computers >> Laptop Accessories >> USB Gadgets >> GANPATI WHOLSALER USB Gadgets
        Price: Rs. 249.0 (Discounted)
        Description Summary: A Sync & Charge micro USB cable for Apple iPhone 6/6 Plus, provided by GANPATI WHOLSALER. The cable is white in color.
        Key Features:
        Type: Sync & Charge Cable
        Connectors: micro USB
        Compatible Phone Series: Others (explicitly mentions Apple iPhone 6/6 Plus in product name/model)
        Material: COPAR
        Color: WHITE
        Brand: GANPATI WHOLSALER
        Product URL: http://www.flipkart.com/ganpati-wholsaler-apple-iphone-6-6-plus-usb-cable/p/itmedb2gchh68rvg?pid=USGEDB2GDGTAWBPX
        Additional Notes: Original retail price was Rs. 399.0.
        Product Report 3

        Product Name: AdroitZ Premium Phone Socket Holder For Apple iPhone 6
        Category: Mobiles & Accessories >> Mobile Accessories >> Car Accessories >> Mobile Holders >> AdroitZ Mobile Holders
        Price: Rs. 164.0 (Discounted)
        Description Summary: A premium phone socket holder designed for Apple iPhone 6, featuring an adjustable clip that fits mobile phones up to 5.7 inches (max opening 86mm). It has an elastic fixation design for easy installation on the steer wheel without extra tools, ensuring safety with its compact fit. The device is embedded with a soft silicone pad to protect the phone from scratches. Available in Red and Black.
        Key Features:
        Type: Anti-slip
        Orientation: Portrait
        Mount Type Dashboard: Yes
        Adjustable clips (max opening 86mm, fits up to 5.7 inches mobile phones)
        Elastic fixation design
        Easy installation
        High compactness with steer wheel for safety
        Soft silicone pad for phone protection
        Brand: AdroitZ
        Product URL: http://www.flipkart.com/adroitz-premium-phone-socket-holder-apple-iphone-6/p/itmefh5ctg2rnygf?pid=ACCEFH5C3WJGKGDC
        Additional Notes: Original retail price was Rs. 599.0. Product comes in Red, Black color.
        Product Report 4

        Product Name: AdroitZ Premium Phone Socket Holder For Apple iPhone 5
        Category: Mobiles & Accessories >> Mobile Accessories >> Car Accessories >> Mobile Holders >> AdroitZ Mobile Holders
        Price: Rs. 164.0 (Discounted)
        Description Summary: A premium phone socket holder designed for Apple iPhone 5, featuring an adjustable clip that fits mobile phones up to 5.7 inches (max opening 86mm). It has an elastic fixation design for easy installation on the steer wheel without extra tools, ensuring safety with its compact fit. The device is embedded with a soft silicone pad to protect the phone from scratches. Available in Red and Black.
        Key Features:
        Type: Anti-slip
        Orientation: Portrait
        Mount Type Dashboard: Yes
        Adjustable clips (max opening 86mm, fits up to 5.7 inches mobile phones)
        Elastic fixation design
        Easy installation
        High compactness with steer wheel for safety
        Soft silicone pad for phone protection
        Brand: AdroitZ
        Product URL: http://www.flipkart.com/adroitz-premium-phone-socket-holder-apple-iphone-5/p/itmefh5csabs3p3n?pid=ACCEFH5CKGYGMMNH
        Additional Notes: Original retail price was Rs. 599.0. Product comes in Red, Black color.
        Product Report 5

        Product Name: AW High Speed Charge and Sync Usb for Iphone 6 Lightning Cable
        Category: Mobiles & Accessories >> Mobile Accessories >> Cables >> AW Cables
        Price: Rs. 271.0 (Discounted)
        Description Summary: A high-speed charge and sync USB to Lightning cable designed for iPhone 6 and other compatible Apple devices. It features a super slim connector head for compatibility with various case openings. The cable is 1m long and white in color.
        Key Features:
        3 Months Replacement Warranty
        Type: Lightning Cable
        Cable Type: High Speed Cable (150 Mbps Speed)
        Connector 1: A Type (USB 2.0 A type)
        Connector 2: 8 Pin (Lightning)
        Cable Length: 1 m
        Super slim connector head
        Compatible with: iPhone 6 / 6 Plus, iPhone 5s / 5c / 5, 6S, iPad Air / Air 2, iPad mini / mini2 / mini 3, iPad Air 4th generation, iPod 5th generation, and iPod nano 7th generation.
        Brand: AW
        Product URL: http://www.flipkart.com/aw-high-speed-charge-sync-usb-iphone-6-lightning-cable/p/itmegktsauxgcqys?pid=ACCEGKTSGJEEVTZ4
        Additional Notes: Original retail price was Rs. 499.0. Product color is White.
    """

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )

    st.subheader("Product Report")
    st.write(response.choices[0].message.content)