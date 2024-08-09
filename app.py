from flask import Flask, render_template, request
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import random

app = Flask(__name__)

# Load models
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
t5_model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
t5_tokenizer = AutoTokenizer.from_pretrained('t5-small')

def generate_single_mcq(context):
    # Generate the question
    input_text = f"Generate a meaningful question with multiple-choice options based on the following context: {context}"
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(inputs, max_length=100, num_beams=3, early_stopping=True)
    question = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if not question:
        question = "Generated question is not meaningful."

    # Get the correct answer
    try:
        qa_result = qa_model(question=question, context=context)
        correct_answer = qa_result.get('answer', "Answer could not be determined")
    except Exception as e:
        correct_answer = "Answer could not be determined"
        print(f"Error obtaining answer: {e}")

    # Generate potential answers
    potential_answers = [correct_answer]
    for _ in range(3):  # Generate 3 more options
        input_text = f"Generate a plausible incorrect answer based on the following context: {context}"
        inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=256, truncation=True)
        outputs = t5_model.generate(inputs, max_length=50, num_beams=3, early_stopping=True)
        answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if answer and answer != correct_answer and answer not in potential_answers:
            potential_answers.append(answer)

    # Ensure we have exactly 4 options
    while len(potential_answers) < 4:
        potential_answers.append(f"Incorrect Answer {len(potential_answers) + 1}")

    # Shuffle options
    random.shuffle(potential_answers)
    
    return {
        "question": question,
        "options": potential_answers[:4],  # Return only 4 options
        "answer": correct_answer
    }

def generate_mcqs(context, num_questions):
    mcqs = []
    for _ in range(num_questions):
        mcq = generate_single_mcq(context)
        mcqs.append(mcq)
    return mcqs

def answer_question(question, context):
    try:
        result = qa_model(question=question, context=context)
        return result.get('answer', "Answer could not be determined")
    except Exception as e:
        print(f"Error obtaining answer: {e}")
        return "Answer could not be determined"

@app.route('/', methods=['GET', 'POST'])
def index():
    mcqs = None
    answer = None
    error = None
    
    if request.method == 'POST':
        context = request.form.get('context')
        num_questions = int(request.form.get('num_questions', 1))
        question = request.form.get('question')
        
        if 'generate_mcqs' in request.form:
            if context:
                try:
                    mcqs = generate_mcqs(context, num_questions)
                except Exception as e:
                    error = f"Error generating MCQs: {str(e)}"
            else:
                error = "Please enter the context text."

        elif 'answer_question' in request.form:
            if context and question:
                try:
                    answer = answer_question(question, context)
                except Exception as e:
                    error = f"Error obtaining answer: {str(e)}"
            else:
                error = "Please enter both the context and the question."
    
    # Convert indices to letters
    def index_to_letter(index):
        return chr(index + 97)  # Convert 0-based index to a, b, c, d

    return render_template('index.html', mcqs=mcqs, answer=answer, error=error, index_to_letter=index_to_letter)

if __name__ == '__main__':
    app.run(debug=True)
