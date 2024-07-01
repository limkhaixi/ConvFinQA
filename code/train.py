import json
from openai import OpenAI
import os
from dotenv import load_dotenv
import time

load_dotenv()

# openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


print("Key", os.getenv("OPENAI_API_KEY"))

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

train_data = load_data(r'code\data\train.json')
# dev_data = load_data(r'code\data\dev_turn.json')

print(len(train_data))
print(train_data[0].keys())

model_name = "gpt-3.5-turbo"



SYSTEM_PROMPT = """
You are a helpful assistant. You answer question based on financial documents.
All of the questions will be numerical.

For each question, you will be provided with texts before the table, the table itself and texts after the table.

Text before the table:
{pre_text}

Table:
The table below starts with a comma-delimited row of headers, and | delimeted rows.
{table}

Text after the table:
{post_text}


The answer to the question will be provided in the source documents. 

Use chain of thoughts method to work out the correct answer if necessary

If you don't know the answer to the question, return 0

If calculating a percentage, calculate to at least 2 decimal places: eg 14.13% or more accurate

"""

# All questions will be numerical, only return numerical answers. Do not generate any string answers.


SYSTEM_PROMPT_ANSWER_HANDLING = """
You are a helpful assistant. 

Your job is to extract the numerical answer from the response provided and transform it into numbers ONLY. 

Do not generate any string answers. If you see any currency or symbols, remove it.

Keep negative values if the relevant answer is negative as this could indicated losses or a decrease

If the answer is that it doesn't know, or if you don;t know what the answer is supposed to be, return 0

If the answer returns a percentage, return the percentage in decimal points (14% should be returned as 0.14)
"""

conversation_number_list = []
question_list = []
answer_list = []
generated_answer_list = []
handled_answer_list = []
tools = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_math_expression",
            "description": "To be called when needed to perform a mathematical operation. Evaluates a mathamtical expression in python",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to calculate in python",
                    },
                },
            },
        },
    }
]

def evaluate_math_expression(expression):
    try:
        result = eval(expression)
        return result
    except Exception as e:
        return f"Error in calculation: {str(e)}"


for conversation_number, data in enumerate(train_data[150:170]):
    print(conversation_number)
    
    questions = data["annotation"]["dialogue_break"]
    answers = data["annotation"]["exe_ans_list"]
    pre_text = ", ".join(data["pre_text"])
    post_text = ", ".join(data["post_text"])

    table_list = []
    for row in data["table"]:
        table_list.append(", ".join(row))
    table_str = "| ".join(table_list)

    # table = "| ".join(data["table"])

    system_prompt = SYSTEM_PROMPT.format(pre_text=pre_text, table=table_str, post_text=post_text)
    message_list = [
            {"role": "system", "content": system_prompt}
        ]

    for question_number, question in enumerate(questions):
        conversation_number_list.append(conversation_number)
        question_list.append(question)
        answer_list.append(float(answers[question_number]))
        message_list.append(
            {"role": "user", "content": question},
        )

        # answer = generated_answer.split()[-1].strip(".")
        retries = 0
        while retries < 10:
            try:
                first_response = client.chat.completions.create(
                    model=model_name,
                    messages=message_list,
                    tools=tools,
                    tool_choice="auto",
                )

                response_message = first_response.choices[0].message
                tool_calls = response_message.tool_calls


                if not tool_calls:
                    print("FIRST CALL")
                    generated_answer = first_response.choices[0].message.content
                
                # Step 2: check if the model wanted to call a function
                if tool_calls:
                    message_list.append(response_message)
                    print("SECOND CALL")
                    print(response_message)
                    print(tool_calls)
                #     # Step 3: call the function
                    for tool_call in tool_calls:
                        function_call = tool_call.function.arguments
                        function_args = json.loads(function_call)
                        print(function_args)
                        function_response = evaluate_math_expression(function_args["expression"])
                        print(function_response)
                        message_list.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": tool_call.function.name,
                                "content": str(function_response),
                            }
                        )
                    print(message_list)
                    time.sleep(1)
                    second_response = client.chat.completions.create(
                        model=model_name,
                        messages=message_list,
                    )  # get a new response from the model where it can see the function response
                    print(second_response)
                    
                    generated_answer = second_response.choices[0].message.content
                
                print(question, generated_answer)

                handle_answer_response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_ANSWER_HANDLING},
                        {"role": "user", "content": "QUESTION: What is the total revenue last quarter ANSWER:total revenue last quarter is $25403"},
                        {"role": "assistant", "content": "25403"},
                        {"role": "user", "content": "QUESTION: What is the total profit last quarter ANSWER:The profit last quarter is 12.7%"},
                        {"role": "assistant", "content": "0.0127"},
                        {"role": "user", "content": "QUESTION: What is the net change in sales? ANSWER:There was a net decrease of -12.7%"},
                        {"role": "assistant", "content": "-0.0127"},
                        {"role": "user", "content": "QUESTION: What is the net change in sales? ANSWER:There was a net decrease of 12.7%"},
                        {"role": "assistant", "content": "-0.0127"},
                        {"role": "user", "content": "QUESTION: What is the total revenue last quarter? ANSWER: The revenue is $5363 million"},
                        {"role": "assistant", "content": "5363000000"},
                        {"role": "user", "content": "QUESTION: What is the total number of items sold last quarter? ANSWER: NUmber of items sold is 0.5 million"},
                        {"role": "assistant", "content": "500000"},
                        {"role": "user", "content": "QUESTION: What is the total profit last quarter? ANSWER:This is not able to be determined"},
                        {"role": "assistant", "content": "0"},
                        {"role": "user", "content": f"QUESTION: {question} ANSWER: {generated_answer}"},
                        ]
                    )

                handled_answer = handle_answer_response.choices[0].message.content
                float_handled_answer = float(handled_answer)
                break
            except Exception as e:
                print(f"Attempt {retries + 1}: Error {str(e)}.")
                retries += 1
                time.sleep(1)

        generated_answer_list.append(generated_answer)
        
        handled_answer_list.append(float(handled_answer))

        message_list.append(
            {"role": "assistant", "content": generated_answer},
        )

evaluation = {
    "conversation_number": conversation_number_list,
    "questions": question_list,
    "labelled answers": answer_list,
    "generated answers": generated_answer_list,
    "handled answers": handled_answer_list
}

tolerance = 0.001
correct_matches = 0


print (evaluation)

for question, labelled, generated, handled in zip(evaluation["questions"], evaluation["labelled answers"], evaluation["generated answers"], evaluation["handled answers"]):
    # Check if both are close enough (floats), exactly the same (ints), or return the same number but different in millions
    if labelled == handled:
        correct_matches += 1
    elif abs(labelled - handled) <= tolerance:
        correct_matches += 1
    elif labelled*1e6 == handled: # flexible compare units in millions
        correct_matches += 1
    elif labelled/100 == handled or labelled*100 == handled: # flexible compare percentages
        correct_matches += 1
    elif labelled*1e9 == handled: # flexible compare units in billions
        correct_matches += 1
    else:
        print(f"Incorrect sample found: question:{question}, labelled:{labelled}, handled: {handled}, generated: {generated},")

# Calculate incorrect matches
incorrect_matches = len(evaluation["labelled answers"]) - correct_matches

# Calculate percentage of correct answers
percentage_correct = (correct_matches / len(evaluation["labelled answers"])) * 100

print(f"Correct matches: {correct_matches}")
print(f"Incorrect matches: {incorrect_matches}")
print(f"Percentage of correct answers: {percentage_correct:.2f}%")