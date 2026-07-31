print("Starting evaluation...")
import json
import csv
import time
from chain import load_qa_chain
QUESTIONS = [
    "What is the horsepower rating of the SD40-2?",
    "How many cylinders does the 645E3 engine have?",
    "What is the normal engine operating temperature?",
    "What is the idle RPM of the SD40-2?",
    "What is the maximum RPM of the engine?",
    "What type of engine does the SD40-2 use?",
    "What is the compression ratio of the 645E3 engine?",
    "What is the function of the turbocharger on the SD40-2?",
    "What is the fuel tank capacity of the SD40-2?",
    "What type of fuel does the SD40-2 use?",
    "What is the fuel pressure specification?",
    "How does the fuel injection system work?",
    "What is the normal oil pressure range?",
    "What type of oil is recommended for the SD40-2?",
    "What is the oil sump capacity?",
    "At what oil pressure does the engine shut down?",
    "What is the function of the radiator on the SD40-2?",
    "How does the cooling fan system work?",
    "What is the coolant capacity?",
    "What temperature triggers the cooling fan?",
    "What is the function of the main generator?",
    "What voltage does the main generator produce?",
    "What is the function of the load regulator?",
    "How does dynamic braking work on the SD40-2?",
    "What is the function of the companion alternator?",
    "How does the air brake system work?",
    "What is the brake cylinder pressure specification?",
    "What is the function of the independent brake?",
    "How does dynamic braking differ from air braking?",
    "What is the emergency brake application pressure?",
    "How many traction motors does the SD40-2 have?",
    "What is the function of the traction motors?",
    "How are traction motors cooled?",
    "What is the wheel slip protection system?",
    "What is the function of the governor?",
    "How does the load regulator control engine output?",
    "What are the throttle notch positions on the SD40-2?",
    "How does the automatic transition system work?",
    "What is the recommended oil change interval?",
    "How do you perform a load test on the SD40-2?",
    "What are the steps to start the SD40-2 engine?",
    "What are the steps to shut down the SD40-2 engine?",
    "How do you test the air brake system?",
    "What is the procedure for changing the fuel filters?",
    "How do you inspect the traction motors?",
    "What triggers an automatic engine shutdown?",
    "What is the function of the wheel slip relay?",
    "How does the overspeed protection work?",
    "What is the low oil pressure shutdown threshold?",
    "What are the safety checks before operating the SD40-2?",
]
def evaluate():
    print("Loading chain...")
    chain = load_qa_chain()
    
    results = []
    total_questions = len(QUESTIONS)
    answered = 0
    unanswered = 0
    total_time = 0

    print(f"Running {total_questions} questions...\n")

    for i, question in enumerate(QUESTIONS):
        print(f"[{i+1}/{total_questions}] {question}")
        
        start = time.time()
        result = chain.invoke({"query": question})
        end = time.time()
        
        elapsed = round(end - start, 2)
        total_time += elapsed
        answer = result["result"]
        sources = [doc.metadata.get("page", "?") for doc in result["source_documents"]]
        
        not_found = "not in the context" in answer.lower() or "i don't know" in answer.lower()
        
        if not_found:
            unanswered += 1
        else:
            answered += 1

        results.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "time_seconds": elapsed,
            "answered": not not_found
        })

        print(f"  Time: {elapsed}s | Answered: {not not_found} | Sources: {sources}\n")

    return results, answered, unanswered, total_time
if __name__ == "__main__":
    results, answered, unanswered, total_time = evaluate()
    
    # Summary
    total = answered + unanswered
    answer_rate = round((answered / total) * 100, 1)
    avg_time = round(total_time / total, 2)
    
    print("=" * 50)
    print(f"EVALUATION SUMMARY")
    print(f"Total questions: {total}")
    print(f"Answered: {answered} ({answer_rate}%)")
    print(f"Unanswered: {unanswered}")
    print(f"Average response time: {avg_time}s")
    print(f"Total time: {round(total_time, 2)}s")
    print("=" * 50)
    
    # Save to JSON
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save to CSV
    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "sources", "time_seconds", "answered"])
        writer.writeheader()
        writer.writerows(results)
    
    print("\nResults saved to results.json and results.csv")