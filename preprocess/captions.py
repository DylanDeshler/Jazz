import json
import argparse
import sys
from google import genai
from google.genai.errors import APIError

SYSTEM_INSTRUCTION = """
You are an expert music metadata editor. I will give you a raw description of a song along with its ground-truth musical key and BPM. 
Your job is to rewrite it into prompts of three distinct lengths: short, medium, and long. These prompts will be used to train a generative music model.
Critically:
    - You must completely remove any mention of chords and lyrics as they are incorrect. 
    - If BPM or key are metnioned in the description, replace every mention with the ground-truth values provided. 
    - Otherwise,
        - 100% of the time add them to long captions
        - 50% of the time add them to medium captions
        - 5% of the time add them to short captions
Return ONLY a valid JSON object with the following keys: short_caption, medium_caption, long_caption.
"""

def test_prompt(input_file: str, output_file: str, limit: int):
    """Runs a synchronous test on a small sample of the data to validate prompt quality."""
    print(f"Testing prompt on the first {limit} entries from {input_file}...\n")
    
    try:
        client = genai.Client()
    except Exception as e:
        print(f"Failed to initialize client. Is GEMINI_API_KEY set? Error: {e}")
        sys.exit(1)

    results = []

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            for count, line in enumerate(f):
                if count >= limit:
                    break
                    
                data = json.loads(line)
                file_path = data.get("file_path", f"unknown_file_{count}")
                raw_caption = data.get("flamingo_raw_caption", "")
                actual_key = data.get("ground_truth_key", "")
                
                prompt = f"Ground Truth Key: {actual_key}\nRaw Description: {raw_caption}"
                
                print(f"--- Processing: {file_path} ---")
                
                try:
                    print(f"**LLM Input:***\n{prompt}\n")
                    response = client.models.generate_content(
                        model='gemini-3.5-flash',
                        contents=prompt,
                        config={
                            "system_instruction": SYSTEM_INSTRUCTION,
                            "response_mime_type": "application/json",
                            "temperature": 0.2
                        }
                    )
                    
                    output_text = response.text
                    print(f"**LLM Output:**\n{output_text}\n")
                    
                    # Store the result 
                    results.append({
                        "file_path": file_path,
                        "llm_output": json.loads(output_text) if output_text else {}
                    })
                    
                except APIError as e:
                    print(f"❌ API Error for {file_path}: {e}\n")
                except json.JSONDecodeError:
                    print(f"❌ Failed to parse LLM output as JSON. Raw output: {output_text}\n")

    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        sys.exit(1)

    # Save test results
    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print(f"✅ Test complete. Saved {len(results)} outputs to {output_file}")


def submit_job(input_file: str, batch_file: str):
    """Parses local data, formats it for the Batch API, and submits the job."""
    requests_data = []

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                file_path = data["file_path"]
                raw_caption = data["flamingo_raw_caption"]
                actual_key = data["ground_truth_key"]
                
                prompt = f"Ground Truth Key: {actual_key}\nRaw Description: {raw_caption}"
                
                batch_request = {
                    "key": file_path, 
                    "request": {
                        "systemInstruction": {
                            "parts": [{"text": SYSTEM_INSTRUCTION}]
                        },
                        "contents": [{
                            "role": "user",
                            "parts": [{"text": prompt}]
                        }],
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "temperature": 0.2
                        }
                    }
                }
                requests_data.append(batch_request)
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        sys.exit(1)

    with open(batch_file, "w", encoding="utf-8") as f:
        for req in requests_data:
            f.write(json.dumps(req) + "\n")

    print(f"Prepared {len(requests_data)} requests for the Batch API.")

    try:
        client = genai.Client()
        
        print("Uploading file to Google's servers...")
        uploaded_file = client.files.upload(
            file=batch_file, 
            config={'display_name': 'song-captions-input'}
        )
        print(f"Uploaded! File ID: {uploaded_file.name}")

        print("Submitting the Batch Job...")
        batch_job = client.batches.create(
            model="gemini-3.5-flash",
            src=uploaded_file.name,
            config={'display_name': 'caption-rephraser-job'}
        )

        print(f"\n✅ Job successfully created!")
        print(f"SAVE THIS JOB NAME: {batch_job.name}")
        print(f"Check status later using: python script.py retrieve {batch_job.name}")

    except APIError as e:
        print(f"API Error during submission: {e}")
        sys.exit(1)


def retrieve_results(job_name: str, output_file: str):
    """Checks the status of a specific batch job and downloads results if complete."""
    try:
        client = genai.Client()
        job = client.batches.get(name=job_name)
        
        print(f"Current Status: {job.state.name}")

        if job.state.name == "JOB_STATE_SUCCEEDED":
            print("Job complete! Downloading results...")
            
            result_file_name = job.dest.file_name
            file_content_bytes = client.files.download(file=result_file_name)
            
            with open(output_file, "wb") as f:
                f.write(file_content_bytes)
                
            print(f"✅ Download successful. Saved to {output_file}!")
            
        elif job.state.name == "JOB_STATE_FAILED":
            print("❌ Job failed. Check your API dashboard for error logs.")
        else:
            print("⏳ Job is still running. Check back later!")
            
    except APIError as e:
        print(f"API Error during retrieval: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Manage asynchronous Gemini API batch jobs and test prompts.",
        epilog="Ensure GEMINI_API_KEY is set in your environment variables."
    )
    
    subparsers = parser.add_subparsers(
        title="Commands", 
        dest="command", 
        required=True
    )

    # Subparser for the 'test' command
    test_parser = subparsers.add_parser(
        "test", 
        help="Run a synchronous test on a small sample of data."
    )
    test_parser.add_argument(
        "--input", 
        default="final_captions.jsonl", 
        help="Path to the input JSONL file (default: final_captions.jsonl)"
    )
    test_parser.add_argument(
        "--output", 
        default="test_outputs.jsonl", 
        help="Path to save the test results (default: test_outputs.jsonl)"
    )
    test_parser.add_argument(
        "--limit", 
        type=int,
        default=3, 
        help="Number of requests to test (default: 3)"
    )

    # Subparser for the 'submit' command
    submit_parser = subparsers.add_parser(
        "submit", 
        help="Prepare data and submit a new batch job."
    )
    submit_parser.add_argument(
        "--input", 
        default="final_captions.jsonl", 
        help="Path to the input JSONL file (default: final_captions.jsonl)"
    )
    submit_parser.add_argument(
        "--batch-file", 
        default="batch_requests.jsonl", 
        help="Path to save the intermediate batch requests (default: batch_requests.jsonl)"
    )

    # Subparser for the 'retrieve' command
    retrieve_parser = subparsers.add_parser(
        "retrieve", 
        help="Check job status and download results if finished."
    )
    retrieve_parser.add_argument(
        "job_name", 
        help="The job name returned from the submit command (e.g., 'batches/123abc456')"
    )
    retrieve_parser.add_argument(
        "--output", 
        default="final_llm_captions.jsonl", 
        help="Path to save the downloaded results (default: final_llm_captions.jsonl)"
    )

    args = parser.parse_args()

    if args.command == "test":
        test_prompt(args.input, args.output, args.limit)
    elif args.command == "submit":
        submit_job(args.input, args.batch_file)
    elif args.command == "retrieve":
        retrieve_results(args.job_name, args.output)

if __name__ == "__main__":
    main()