import subprocess
import time
import math

def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return result.stdout.strip()

# Get all changed and untracked files
status_output = run_cmd("git status --porcelain")
if not status_output:
    print("No changes to commit.")
    exit()

files = []
for line in status_output.split('\n'):
    if line:
        # line is like " M path/to/file" or "?? path/to/file" or "A  path/to/file"
        # Extract the file path (characters from index 3 onwards)
        file_path = line[3:]
        # handle quotes if any
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
        files.append(file_path)

chunk_size = 10
total_chunks = math.ceil(len(files) / chunk_size)

print(f"Total files to commit: {len(files)}")
print(f"Total batches: {total_chunks}")

# Unstage everything first just to be clean
run_cmd("git reset HEAD")

for i in range(0, len(files), chunk_size):
    chunk = files[i:i+chunk_size]
    batch_num = (i // chunk_size) + 1
    print(f"Processing batch {batch_num}/{total_chunks}...")
    
    # Add files
    for f in chunk:
        run_cmd(f'git add "{f}"')
    
    # Commit
    commit_msg = f"chore: upload batch {batch_num} of {total_chunks}"
    run_cmd(f'git commit -m "{commit_msg}"')
    
    # Push
    print(f"Pushing batch {batch_num}...")
    push_result = subprocess.run("git push", capture_output=True, text=True, shell=True)
    if push_result.returncode != 0:
        print(f"Failed to push batch {batch_num}: {push_result.stderr}")
        break
    
    print(f"Batch {batch_num} pushed successfully.")

print("All batches processed.")
