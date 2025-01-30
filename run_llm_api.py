import os
import json
import logging
import yaml
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Optional
from google.api_core import exceptions as google_exceptions

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("execution.log"),
        logging.StreamHandler()
    ]
)

class ProsQAEvaluator:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize evaluator and load configuration."""
        load_dotenv()
        self.config = self._load_config(config_path)
        self._validate_config()
        self.model = self._initialize_gemini()

    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _validate_config(self):
        """Ensure required configuration keys exist."""
        required = {
            'paths:subset', 'paths:output',
            'prompt:system', 'prompt:answer_prefix', 'prompt:required_format',
            'model:name', 'model:temperature'
        }
        existing = set(f"{section}:{key}" for section in self.config for key in self.config[section])
        missing = required - existing
        if missing:
            raise ValueError(f"Missing config keys: {missing}")

    def _initialize_gemini(self):
        """Configure Gemini API client."""
        genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
        return genai.GenerativeModel(self.config['model']['name'])

    def load_dataset(self) -> List[Dict]:
        """Load JSON dataset and validate structure."""
        path = self.config['paths']['subset']
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        for idx, item in enumerate(data):
            if not {'question', 'answer'}.issubset(item.keys()):
                raise ValueError(f"Invalid item format at index {idx}")
        return data

    def generate_prompt(self, question: str) -> str:
        """
        Strictly enforce CoT reasoning and final answer structure.
        - Requires "X is a Y" format.
        - Makes it harder for the model to output just "Y".
        """
        system_text = self.config['prompt']['system']
        required_format = self.config['prompt']['required_format']
        answer_prefix = self.config['prompt']['answer_prefix']

        return (
            f"{system_text}\n\n"
            "You must reason step-by-step and clearly state the answer at the end.\n"
            "Format your response as follows:\n"
            "Reasoning:\n"
            "[Step-by-step logical deduction]\n"
            f"{answer_prefix}[Subject] is a [Category]\n\n"
            f"NEW QUESTION: {question}\n\n"
            f"{required_format}\n"
            "Provide your response:"
        )

    def query_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM using greedy decoding for strict CoT reasoning."""
        temp = self.config['model']['temperature']
        max_toks = self.config['model'].get('max_tokens', 512)
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temp,  # Ensure purely greedy decoding
                    max_output_tokens=max_toks,
                    top_p=1.0
                )
            )
            if not response.candidates:
                return None

            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                return candidate.content.parts[0].text.strip()
            return None

        except google_exceptions.GoogleAPIError as e:
            logging.error(f"Gemini API Error: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return None

    def extract_answer(self, llm_output: str) -> Optional[str]:
        """Extract 'Final Answer: X is a Y' from the LLM response."""
        prefix = self.config['prompt']['answer_prefix']
        lines = llm_output.splitlines()

        for line in lines:
            if line.startswith(prefix):
                answer = line.split(prefix, 1)[-1].strip()
                if " is a " in answer:
                    return answer  # Properly formatted
                else:
                    logging.warning(f"Answer format incorrect: {answer}")
                    return None
        return None

    def evaluate(self) -> Dict:
        """
        Run evaluation on dataset:
        - Generates prompt
        - Extracts answer
        - Compares to expected
        - Tracks accuracy and case studies
        """
        data = self.load_dataset()
        results = []
        correct_count = 0
        case_studies = {"correct": [], "incorrect": []}

        for i, item in enumerate(data, start=1):
            question = item['question']
            expected = item['answer']

            prompt = self.generate_prompt(question)
            llm_output = self.query_llm(prompt)
            final_answer = self.extract_answer(llm_output) if llm_output else None

            is_correct = (final_answer == expected)
            if is_correct:
                correct_count += 1

            result = {
                "question": question,
                "expected": expected,
                "response": llm_output,
                "parsed_answer": final_answer,
                "correct": is_correct
            }
            results.append(result)

            if is_correct and len(case_studies["correct"]) < 3:
                case_studies["correct"].append(result)
            elif not is_correct and len(case_studies["incorrect"]) < 3:
                case_studies["incorrect"].append(result)

            if i % 5 == 0 or i == len(data):
                accuracy_so_far = correct_count / i
                logging.info(f"Processed {i}/{len(data)} | Accuracy: {accuracy_so_far:.1%}")

        accuracy = correct_count / len(data) if data else 0
        return {
            "summary": {
                "accuracy": accuracy,
                "correct": correct_count,
                "total": len(data),
            },
            "case_studies": case_studies,
            "full_results": results
        }

    def save_report(self, report: Dict):
        """Save evaluation report to JSON."""
        out_path = self.config['paths']['output']
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(report, f, indent=2)
        logging.info(f"Report saved to {out_path}")

if __name__ == "__main__":
    evaluator = ProsQAEvaluator()
    report = evaluator.evaluate()
    evaluator.save_report(report)

    print("\n", "="*15, "Evaluation Summary", "="*15)
    print(f"Accuracy: {report['summary']['accuracy']:.1%}")
    print(f"Correct: {report['summary']['correct']}/{report['summary']['total']}")

    print("\n", "="*15, "Case Studies", "="*15)
    for category in ["correct", "incorrect"]:
        items = report["case_studies"][category]
        if items:
            print(f"\n{category.upper()} EXAMPLES:")
            for ex in items:
                print(f"Q: {ex['question']}")
                print(f"Expected: {ex['expected']}")
                print(f"Parsed: {ex['parsed_answer']}")
                print(f"Full Response:\n{ex['response']}\n{'-'*40}")
