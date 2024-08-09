
# MCQ Generator and QA System

This application is a Streamlit-based web app that generates multiple-choice questions (MCQs) from a given context and provides answers to user queries based on that context. It utilizes pre-trained language models from Hugging Face to create questions and find answers.

## Features

- **MCQ Generation**: Generate MCQs from a given context. Users can specify the number of questions they want to create.
- **Question Answering**: Enter a question, and the app will provide an answer based on the context.
- **Interactive UI**: A simple and interactive interface for generating and displaying questions and answers.

## How It Works

1. **MCQ Generation**:
   - The app uses a T5 model to generate questions based on the provided context.
   - It extracts correct answers using a question-answering model (`distilbert-base-uncased-distilled-squad`).
   - Additional plausible answers are generated, shuffled, and displayed as multiple-choice options.

2. **Question Answering**:
   - Users can input a specific question related to the context.
   - The QA model analyzes the question and context to extract and display the most relevant answer.

## Installation

To run the application, you need to have Python installed with the following dependencies:

```bash
pip install streamlit transformers
```

## Usage

1. **Run the Streamlit App**:
   - Execute the following command in your terminal to start the app:
     ```bash
     streamlit run app.py
     ```

2. **Generate MCQs**:
   - Enter the context text in the provided text area.
   - Specify the number of MCQs you want to generate.
   - Click on "Generate MCQs" to see the generated questions and options.

3. **Answer Questions**:
   - Enter a question in the respective text area to get an answer based on the context.
   - Click on "Answer Question" to see the answer.

## Application

This app is ideal for educators, content creators, and anyone interested in generating educational material automatically. It can be used for:

- **Educational Content**: Create quiz questions for learning material.
- **Training and Assessment**: Develop practice questions for exams or training sessions.
- **Research and Analysis**: Quickly analyze text content and generate questions for comprehension checks.

## Code Explanation

### Model Loading

```python
# Load models
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
t5_model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
t5_tokenizer = AutoTokenizer.from_pretrained('t5-small')
```

The application uses the `distilbert-base-uncased-distilled-squad` for the question-answering pipeline and `t5-small` for generating questions.

### Generating MCQs

```python
def generate_single_mcq(context):
    # Generate the question
    input_text = f"Generate a question with multiple-choice options based on the following context: {context}"
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=256, truncation=True)
    outputs = t5_model.generate(inputs, max_length=100, num_beams=1, early_stopping=True)
    question = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Get the correct answer
    try:
        correct_answer = qa_model(question=question, context=context)['answer']
    except Exception as e:
        st.error(f"Error in QA model: {e}")
        correct_answer = "Answer could not be determined"

    # Generate potential answers
    potential_answers = [correct_answer]
    for _ in range(3):  # Generate 3 more options
        input_text = f"Generate a plausible incorrect answer based on the following context: {context}"
        inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=256, truncation=True)
        outputs = t5_model.generate(inputs, max_length=50, num_beams=1, early_stopping=True)
        answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if answer and answer != correct_answer and answer not in potential_answers:
            potential_answers.append(answer)

    # Ensure we have exactly 4 options
    if len(potential_answers) < 4:
        potential_answers += ["Placeholder Answer"] * (4 - len(potential_answers))

    # Shuffle options
    random.shuffle(potential_answers)

    return {
        "question": question,
        "options": potential_answers[:4],  # Return only 4 options
        "answer": correct_answer
    }
```

The `generate_single_mcq` function creates a question and possible answers using the T5 model and question-answering pipeline. It ensures that four options are available and shuffled before returning.

### User Interface

The Streamlit interface allows users to input context and questions, choose the number of MCQs, and generate or answer questions through buttons and text areas.

```python
# Streamlit UI
st.title("MCQ Generator and QA System")

# Text area for input context
context = st.text_area("Enter the context (text) based on which MCQs will be generated:")

# Number input for the number of questions
num_questions = st.number_input("How many MCQs do you want to generate?", min_value=1, max_value=10, step=1, value=1)

if st.button("Generate MCQs"):
    if context:
        with st.spinner("Generating MCQs..."):
            mcqs = generate_mcqs(context, num_questions)
            st.subheader("Generated MCQs:")
            for i, mcq in enumerate(mcqs):
                st.write(f"**Q{i+1}: {mcq['question']}**")
                for j, option in enumerate(mcq['options']):
                    st.write(f"{chr(97 + j)}. {option}")  # List options as a, b, c, d
                st.write(f"*Answer: {mcq['answer']}*")
    else:
        st.warning("Please enter the context text.")

# Text area for entering a question
question = st.text_area("Enter a question to get an answer based on the context above:")

if st.button("Answer Question"):
    if context and question:
        with st.spinner("Finding answer..."):
            answer = answer_question(question, context)
            st.subheader("Answer:")
            st.write(answer)
    else:
        st.warning("Please enter both the context and the question.")
```

The app uses Streamlit widgets to facilitate user input and display results. The questions and answers are generated dynamically based on user-provided context.

