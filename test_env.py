import subprocess

print("🚀 Running inference.py test...\n")

try:
    result = subprocess.run(
        ["python", "inference.py"],
        capture_output=True,
        text=True,
        timeout=60
    )

    print("✅ STDOUT:\n")
    print(result.stdout)

    if result.stderr:
        print("\n⚠️ STDERR:\n")
        print(result.stderr)

    if result.returncode == 0:
        print("\n🎉 TEST PASSED: inference.py ran successfully")
    else:
        print(f"\n❌ TEST FAILED: Exit code {result.returncode}")

except Exception as e:
    print("💥 Test crashed:", e)