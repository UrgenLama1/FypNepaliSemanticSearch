import streamlit as st
from neural_searcher import NeuralSearcher

# Initialize the NeuralSearcher instance
searcher = NeuralSearcher(collection_name="nepali")

# Create the Streamlit UI
def main():
    st.title("Nepali Text Search")

    # Input area for user query
    query = st.text_input("Enter your query:")

    # Button to trigger the search
    if st.button("Search"):
        # Perform the search using the NeuralSearcher instance
        results = searcher.search(query)

        # Display the search results
        st.write("Search Results:")
        for payload, score in results:
            st.write(f"Score: {score}")
            st.write(f"Payload: {payload}")
            st.write("-----------")

# Run the Streamlit app
if __name__ == "__main__":
    main()
