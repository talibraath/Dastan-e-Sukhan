import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from collections import Counter
from textblob import TextBlob

def generate_poetry(seed_text, sequence_length, tokenizer, next_words=50, temperature=1.0, words_per_line=7):
    """
    Generate poetry using a pre-trained LSTM model.
    
    Parameters:
        seed_text (str): The initial text to start generation.
        sequence_length (int): The fixed sequence length for model input.
        tokenizer: A fitted tokenizer.
        next_words (int): Number of words to generate.
        temperature (float): Sampling temperature to control creativity.
        words_per_line (int): Number of words per output line.
        
    Returns:
        str: Generated poetry with line breaks inserted after every specified number of words.
    """
    # Load the model (each call loads it; for efficiency, consider caching this in production)
    model = tf.keras.models.load_model("lstm_poetry_model.h5")
    # Start with the seed text split into words
    words = seed_text.split()

    for i in range(next_words):
        # Use only the last 'sequence_length' words for prediction
        current_sequence = " ".join(words[-sequence_length:])
        token_list = tokenizer.texts_to_sequences([current_sequence])[0]
        token_list = pad_sequences([token_list], maxlen=sequence_length, padding='pre')

        predicted = model.predict(token_list, verbose=0)[0]
        predicted = np.asarray(predicted).astype('float64')
        predicted = np.log(predicted + 1e-7) / temperature
        predicted = np.exp(predicted) / np.sum(np.exp(predicted))
        predicted_word_index = np.random.choice(len(predicted), p=predicted)

        # Find the word corresponding to the predicted index
        predicted_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                predicted_word = word
                break
        words.append(predicted_word)
    
    # Reformat the complete word list so that each line has exactly 'words_per_line' words.
    lines = [" ".join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
    formatted_poetry = "\n".join(lines)
    return formatted_poetry

def load_tokenizer():
    """
    Load and return the tokenizer from the saved pickle file.
    """
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

def save_poetry(generated_text, file_path="generated_poetry.txt"):
    """
    Save the generated poetry to a text file.
    
    Parameters:
        generated_text (str): The poetry text to save.
        file_path (str): Destination file path.
    
    Returns:
        str: The file path where the poetry was saved.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(generated_text)
    return file_path

def analyze_poetry(generated_text):
    """
    Analyze the generated poetry.
    
    Returns:
        dict: Contains word_count, line_count, and the top 5 frequent words.
    """
    lines = generated_text.strip().split("\n")
    words = generated_text.split()
    word_count = len(words)
    line_count = len(lines)
    common_words = Counter(words).most_common(5)
    return {
        "word_count": word_count,
        "line_count": line_count,
        "common_words": common_words
    }

def sentiment_analysis(generated_text):
    """
    Compute sentiment polarity for each line using TextBlob.
    Each line is classified as Positive, Negative, or Neutral.
    
    Returns:
        List[tuple]: Each tuple contains (polarity, sentiment_label) for a line.
    """
    lines = generated_text.strip().split("\n")
    sentiments = []
    for line in lines:
        if line.strip():
            polarity = TextBlob(line).sentiment.polarity
            if polarity > 0.1:
                sentiment = "Positive"
            elif polarity < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            sentiments.append((polarity, sentiment))
        else:
            sentiments.append((0, "Neutral"))
    return sentiments

# For local testing:
if __name__ == "__main__":
    tokenizer = load_tokenizer()
    seed = "dil ka khel"
    seq_length = 20
    poetry = generate_poetry(seed, seq_length, tokenizer, next_words=50, temperature=1.0, words_per_line=7)
    print("Generated Poetry:\n", poetry)
    stats = analyze_poetry(poetry)
    print("\nAnalysis:", stats)
    sentiments = sentiment_analysis(poetry)
    print("\nSentiments:", sentiments)
