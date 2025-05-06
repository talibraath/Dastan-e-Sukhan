# Dastan-e-Sukhan

Dastan-e-Sukhan is a machine learning-based poetry generation project that uses a pre-trained LSTM (Long Short-Term Memory) model to create Urdu poetry. This project combines natural language processing, deep learning, and sentiment analysis to generate creative poetic verses.

## Features
- **Poetry Generation**: Generate Urdu poetry based on a seed text and predefined sequence length.
- **LSTM-based Model**: The project uses an LSTM neural network trained on Urdu poetry data.
- **Sentiment Analysis**: Evaluate the sentiment of the generated poetry (Positive, Neutral, Negative) using TextBlob.
- **Customizable Creativity**: Control the randomness of text generation with a `temperature` parameter.
- **Word Frequency Analysis**: Analyze the most common words in the generated poetry.

## Technologies Used
- **TensorFlow/Keras**: For building and running the LSTM model.
- **TextBlob**: For sentiment analysis.
- **NumPy**: For numerical computations.
- **Python Standard Libraries**: Used for file handling, tokenization, and other utilities.

## How It Works
1. **Generate Poetry**:
   - Input a seed text.
   - Specify the sequence length and the number of words to generate.
   - The LSTM model predicts the next words based on the input sequence.
2. **Analyze Poetry**:
   - The generated poetry is analyzed to compute word count, line count, and frequent words.
3. **Sentiment Analysis**:
   - Each line of poetry is classified as Positive, Negative, or Neutral based on sentiment polarity.

## Example Usage
Here’s an example of how to use the project:

### Poetry Generation
```python
from backend import load_tokenizer, generate_poetry

# Load the tokenizer
tokenizer = load_tokenizer()

# Generate poetry
seed_text = "dil ka khel"
sequence_length = 20
poetry = generate_poetry(seed_text, sequence_length, tokenizer, next_words=50, temperature=1.0)
print("Generated Poetry:\n", poetry)
