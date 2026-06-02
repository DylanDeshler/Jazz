import json
import argparse
import time
import sys
import os
from google import genai
from google.genai.errors import APIError
from google.cloud import storage

# --- CONFIGURATION ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "virtualitics-ai-team")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east5")
GCS_BUCKET_NAME = "virtualitics-caption-batch-data"
MODEL_NAME = "gemini-2.5-flash"
# ---------------------

SYSTEM_INSTRUCTION = """
You are an expert music metadata editor. I will give you a raw description of a song along with its ground-truth musical key and BPM. 
Your job is to rewrite it into prompts of three distinct lengths: short, medium, and long. Each prompt should provide as much information as possible given their length about the entire piece based on the raw description. These prompts will be used to train a generative music model.
Critically:
    - You must completely remove any mention of BPM, key, chords, and lyrics as they are incorrect. 
    - If not "None", add the ground truth BPM, key, or both to the prompt:
        - 80% of the time for long captions
        - 20% of the time for medium captions
        - 5% of the time for short captions
    - Short captions should range from several words to a short sentence.
    - Medium captions are between 1 and 2 sentences covering all high level detail and the most informative specifics.
    - Long captions should provide all of the details from the raw description.
Return ONLY a valid JSON object with the following keys: short_caption, medium_caption, and long_caption.
"""

def get_vertex_client():
    """Initializes the Vertex AI client using Application Default Credentials."""
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )

def test_prompt(input_file: str, output_file: str, limit: int):
    """Runs a synchronous test on a small sample of the data to validate prompt quality."""
    print(f"Testing prompt on the first {limit} entries from {input_file} using {MODEL_NAME}...\n")
    client = get_vertex_client()
    results = []

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            for count, line in enumerate(f):
                if count >= limit:
                    break
                    
                data = json.loads(line)
                file_path = data.get("file_path", f"unknown_file_{count}")
                raw_caption = data.get("caption", "")
                actual_key = data.get("key", "None")
                actual_bpm = data.get("bpm", "")
                try:
                    actual_bpm = int(actual_bpm)
                except:
                    actual_bpm = 'None'
                
                prompt = f"Ground Truth Key: {actual_key}\nGround Truth BPM: {actual_bpm}\nRaw Description: {raw_caption}"
                
                print(f"--- Processing: {file_path} ---")
                print(f"**LLM Input:**\n{prompt}\n")
                
                try:
                    response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=prompt,
                        config={
                            "system_instruction": SYSTEM_INSTRUCTION,
                            "response_mime_type": "application/json",
                            "temperature": 0.2
                        }
                    )
                    
                    output_text = response.text
                    print(f"**LLM Output:**\n{output_text}\n")
                    print("-" * 50)
                    
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

    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print(f"✅ Test complete. Saved {len(results)} sample outputs to {output_file}")


def process_synchronous(input_file: str, output_file: str, rps_limit: float = 1.0):
    """
    Processes entries one-by-one synchronously. 
    Maintains progress across failures by checkpointing existing records on disk.
    """
    print(f"Starting synchronous processing using {MODEL_NAME} in {LOCATION}...")
    client = get_vertex_client()
    
    processed_paths = set()
    if os.path.exists(output_file):
        print(f"Found existing output file '{output_file}'. Reading progress to resume...")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing_data = json.loads(line)
                    if "file_path" in existing_data:
                        processed_paths.add(existing_data["file_path"])
                except json.JSONDecodeError:
                    continue
        print(f"Skipping {len(processed_paths)} already processed entries.")

    try:
        with open(input_file, "r", encoding="utf-8") as infile, \
             open(output_file, "a", encoding="utf-8") as outfile:
             
            for count, line in enumerate(infile):
                data = json.loads(line)
                file_path = data.get("file_path")
                
                if file_path in processed_paths:
                    continue
                    
                raw_caption = data.get("caption", "")
                if "Failed due to exceptions" in raw_caption or raw_caption == "":
                    print(f"Skipping invalid caption for: {file_path}")
                    continue
                    
                actual_key = data.get("key", "None")
                actual_bpm = data.get("bpm", "")
                try:
                    actual_bpm = int(actual_bpm)
                except:
                    actual_bpm = 'None'
                
                prompt = f"Ground Truth Key: {actual_key}\nGround Truth BPM: {actual_bpm}\nRaw Description: {raw_caption}"
                print(f"[{count}] Processing: {file_path}")
                
                try:
                    response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=prompt,
                        config={
                            "system_instruction": SYSTEM_INSTRUCTION,
                            "response_mime_type": "application/json",
                            "temperature": 0.2
                        }
                    )
                    
                    output_text = response.text
                    
                    result_record = {
                        "file_path": file_path,
                        "llm_output": json.loads(output_text) if output_text else {}
                    }
                    outfile.write(json.dumps(result_record) + "\n")
                    outfile.flush()
                    
                    time.sleep(1.0 / rps_limit)
                    
                except APIError as e:
                    print(f"❌ API Error for {file_path}: {e}")
                    print("Cooling down for 10 seconds...")
                    time.sleep(10)
                except json.JSONDecodeError:
                    print(f"❌ Failed to parse LLM output as JSON. Raw output: {output_text}")
                except Exception as e:
                    print(f"❌ Unexpected error occurred: {e}")
                    print("Progress saved. Exiting gracefully.")
                    sys.exit(1)
                    
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        sys.exit(1)
        
    print(f"✅ Synchronous run finished processing.")


def submit_job(input_file: str, batch_file: str):
    """Parses local data, formats it for Vertex AI Batch API, and uploads via GCS."""
    requests_data = []
    skipped = 0
    no_bpm = 0
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                file_path = data["file_path"]
                raw_caption = data.get("caption", "")
                
                if "Failed due to exceptions" in raw_caption or raw_caption == "":
                    skipped += 1
                    continue
                
                actual_key = data.get("key", "None")
                actual_bpm = data.get("bpm", "")
                try:
                    actual_bpm = int(actual_bpm)
                except:
                    actual_bpm = 'None'
                    no_bpm += 1
                
                prompt = f"Ground Truth Key: {actual_key}\nGround Truth BPM: {actual_bpm}\nRaw Description: {raw_caption}"
                
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
    
    print(f"Filtered records: Skipped {skipped} bad captions. Found {no_bpm} unparseable BPMs.")

    with open(batch_file, "w", encoding="utf-8") as f:
        for req in requests_data:
            f.write(json.dumps(req) + "\n")

    print(f"Prepared {len(requests_data)} requests locally for the Batch API.")

    try:
        gcs_client = storage.Client(project=PROJECT_ID)
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        
        gcs_input_path = f"batch_inputs/{batch_file}"
        gcs_output_prefix = "batch_outputs/"
        
        blob = bucket.blob(gcs_input_path)
        print(f"Uploading file to gs://{GCS_BUCKET_NAME}/{gcs_input_path}...")
        blob.upload_from_filename(batch_file, content_type="application/jsonl")
        print("✅ Uploaded to Google Cloud Storage!")

        client = get_vertex_client()
        print(f"Submitting the Vertex AI Batch Job using {MODEL_NAME}...")
        
        batch_job = client.batches.create(
            model=MODEL_NAME,
            src=f"gs://{GCS_BUCKET_NAME}/{gcs_input_path}",
            dest=f"gs://{GCS_BUCKET_NAME}/{gcs_output_prefix}",
        )

        print(f"\n✅ Job successfully created!")
        print(f"SAVE THIS JOB NAME: {batch_job.name}")
        print(f"Check status later using: python script.py retrieve \"{batch_job.name}\"")

    except Exception as e:
        print(f"Error during batch submission: {e}")
        sys.exit(1)


def retrieve_results(job_name: str, output_file: str):
    """Checks the status of a Vertex AI batch job and downloads results from GCS."""
    try:
        client = get_vertex_client()
        job = client.batches.get(name=job_name)
        
        print(f"Current Status: {job.state.name}")

        if job.state.name == "JOB_STATE_SUCCEEDED":
            print("Job complete! Searching GCS bucket for results...")
            
            gcs_dest_uri = job.dest 
            prefix = "/".join(gcs_dest_uri.split("/")[3:])
            
            gcs_client = storage.Client(project=PROJECT_ID)
            bucket = gcs_client.bucket(GCS_BUCKET_NAME)
            blobs = list(bucket.list_blobs(prefix=prefix))
            
            output_blob = None
            for b in blobs:
                if b.name.endswith(".jsonl"):
                    output_blob = b
                    break
                    
            if output_blob:
                print(f"Downloading {output_blob.name}...")
                output_blob.download_to_filename(output_file)
                print(f"✅ Download successful. Saved to {output_file}!")
            else:
                print(f"❌ Job complete, but no .jsonl files found inside the GCS output prefix: {prefix}")
            
        elif job.state.name == "JOB_STATE_FAILED":
            print("❌ Job failed. Check your Vertex AI Google Cloud Console pipeline interface logs.")
        else:
            print("⏳ Job is still running. Check back later!")
            
    except Exception as e:
        print(f"Error during retrieval: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Manage Gemini 2.5 Pro processing workloads on Vertex AI.")
    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    # Re-introduced Test Option
    test_parser = subparsers.add_parser("test", help="Run a synchronous test on a small sample of data.")
    test_parser.add_argument("--input", default="final_captions.jsonl", help="Input file path.")
    test_parser.add_argument("--output", default="test_outputs.jsonl", help="Path to save test results.")
    test_parser.add_argument("--limit", type=int, default=3, help="Number of records to sample.")

    # Synchronous Resumable Option
    sync_parser = subparsers.add_parser("sync", help="Process entries sequentially with failure-resumption tracking.")
    sync_parser.add_argument("--input", default="final_captions.jsonl", help="Input file path.")
    sync_parser.add_argument("--output", default="final_llm_captions.jsonl", help="Output file path.")
    sync_parser.add_argument("--rps", type=float, default=2.0, help="Target Requests Per Second throttling.")

    # Asynchronous Submit Option
    submit_parser = subparsers.add_parser("submit", help="Prepare data and submit a Cloud Batch Job via GCS.")
    submit_parser.add_argument("--input", default="final_captions.jsonl", help="Input data path.")
    submit_parser.add_argument("--batch-file", default="batch_requests.jsonl", help="Local intermediate staging file path.")

    # Asynchronous Retrieval Option
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve outputs from GCS once a batch job is completed.")
    retrieve_parser.add_argument("job_name", help="Full path identifier of the job execution.")
    retrieve_parser.add_argument("--output", default="final_llm_captions.jsonl", help="Local output destination path.")

    args = parser.parse_args()

    if args.command == "test":
        test_prompt(args.input, args.output, args.limit)
    elif args.command == "sync":
        process_synchronous(args.input, args.output, args.rps)
    elif args.command == "submit":
        submit_job(args.input, args.batch_file)
    elif args.command == "retrieve":
        retrieve_results(args.job_name, args.output)

if __name__ == "__main__":
    main()