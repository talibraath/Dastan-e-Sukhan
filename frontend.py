import streamlit as st
import backend
import matplotlib.pyplot as plt
from io import BytesIO
from collections import Counter

# Configure the Streamlit page
st.set_page_config(page_title=" Dastan-e-Sukhan", layout="centered")

# 🎨 Custom Styling: Dark background with coral accents
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #FF7F50;
        font-family: 'Poppins', sans-serif;
    }
    .title {
        font-size: 60px;
        color: #FF7F50;
        text-align: center;
        margin: 20px 0;
    }
    .btn {
        background-color: #FF7F50;
        border: none;
        border-radius: 8px;
        color: #121212;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        margin: 5px;
    }
    .btn:hover {
        background-color: #ff8c69;
    }
    </style>
""", unsafe_allow_html=True)

# Title (displayed plainly without a surrounding rectangle)
st.markdown("<h1 class='title'> Dastan-e-Sukhan</h1>", unsafe_allow_html=True)

# Sidebar: Customize Generation Parameters
st.sidebar.header("Customize Your Poetry")
seed_text = st.sidebar.text_input("Seed Text", "Enter Some Text For Poetry")
next_words = st.sidebar.slider("Number of Words to Generate", 20, 100, 50, 5)
temperature = st.sidebar.slider("Creativity (Temperature)", 0.5, 2.0, 1.0, 0.1)
words_per_line = st.sidebar.slider("Words Per Line", 3, 10, 7, 1)
sequence_length = st.sidebar.number_input("Sequence Length", 10, 50, 20, 1)

# Initialize like/dislike counters if not present
if 'like_count' not in st.session_state:
    st.session_state.like_count = 0
if 'dislike_count' not in st.session_state:
    st.session_state.dislike_count = 0

# Load tokenizer once and store in session state
if 'tokenizer' not in st.session_state:
    with st.spinner("Loading tokenizer..."):
        st.session_state.tokenizer = backend.load_tokenizer()

# Generate Poetry: When clicked, store the result in session state (separate from like/dislike clicks)
if st.button("Generate Poetry"):
    with st.spinner("Crafting your poetic masterpiece..."):
        poetry = backend.generate_poetry(seed_text, sequence_length, st.session_state.tokenizer, next_words, temperature, words_per_line)
    st.session_state.poetry = poetry

# Display generated poetry if available
if 'poetry' in st.session_state:
    st.text_area("Generated Poetry", st.session_state.poetry, height=300)
    
    # Feedback Section: Like/Dislike with persistent counters
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Like", key="like"):
            st.session_state.like_count += 1
    with col2:
        if st.button("👎 Dislike", key="dislike"):
            st.session_state.dislike_count += 1
    st.write(f"Likes: {st.session_state.like_count} | Dislikes: {st.session_state.dislike_count}")
    
    # Beginner-Friendly Poetry Analysis Section
    st.markdown("---")
    st.subheader("Poetry Analysis")
    analysis = backend.analyze_poetry(st.session_state.poetry)
    st.write("Your generated poem contains a total of **{} words** spread across **{} lines**.".format(
        analysis["word_count"], analysis["line_count"]
    ))
    st.write("This means the poem has {} words and is divided into {} separate lines.".format(
        analysis["word_count"], analysis["line_count"]
    ))
    st.write("The **Top 5 Frequent Words** in your poem are:")
    st.write(analysis["common_words"])
    st.write("These words appear most often in your poem and may give you an idea of its main themes and style.")
    
    # Sentiment Analysis Section: Compute sentiment per line, classify, and plot graph
    st.markdown("---")
    st.subheader("Sentiment Analysis")
    sentiment_results = backend.sentiment_analysis(st.session_state.poetry)
    polarity_values = [s[0] for s in sentiment_results]
    sentiment_labels = [s[1] for s in sentiment_results]
    
    if polarity_values:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(polarity_values) + 1), polarity_values, marker='o', linestyle='-', color='#FF7F50')
        ax.set_title("Sentiment Polarity per Line")
        ax.set_xlabel("Line Number")
        ax.set_ylabel("Sentiment Polarity")
        st.pyplot(fig)
        
        # Convert the plot to a PNG for download
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="Download Sentiment Graph as PNG",
            data=buf,
            file_name="sentiment_graph.png",
            mime="image/png"
        )
        
        # Display overall sentiment counts with explanations
        sentiment_counter = Counter(sentiment_labels)
        st.write("### Overall Sentiment Counts")
        st.write("Below is a summary of the sentiment analysis of your poem. This shows the number of lines classified as Positive, Negative, or Neutral:")
        st.write(sentiment_counter)
    else:
        st.write("No sentiment data available.")
    
    # Download the generated poetry as a text file
    st.markdown("---")
    st.subheader("Download Your Poetry")
    st.download_button(
        label="Download Poetry as TXT",
        data=st.session_state.poetry,
        file_name="generated_poetry.txt",
        mime="text/plain"
    )
