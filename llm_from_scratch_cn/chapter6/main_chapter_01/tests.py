import subprocess

def test_gpt_class_finetune():
    command = ["python",  "main06.py", "--test_mode"]
    res = subprocess.run(command, capture_output=True, text=True)
    assert res.returncode == 0, f"Script exited with errors: {res.stderr}"