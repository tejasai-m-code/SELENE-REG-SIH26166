import subprocess
import sys
from pathlib import Path

def test_frontend_hooks_ordering_regression():
    """
    Regression test for the React hooks ordering violation in Pair Registration.
    Verifies:
    1. Static AST / Hook order integrity of Evidence and PairRegistrationResults.
    2. Simulated React hook lifecycle across all state transitions:
       - Idle
       - 1 Image
       - 2 Images
       - Loading / Progress
       - Success result (PASS)
       - Correspondence filter switching (all/inliers/outliers)
       - Match point selection
       - Rejected result (FAILED quality gate)
       - Transport / Network error
       - Error dismissal / return to idle
    """
    repo_root = Path(__file__).resolve().parent.parent
    test_script = repo_root / "tests" / "test_pair_registration_hooks.mjs"
    assert test_script.exists(), f"Test script {test_script} does not exist"

    result = subprocess.run(
        ["node", str(test_script)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    assert result.returncode == 0, f"Hook ordering regression test failed with code {result.returncode}"
    assert "ALL PAIR REGISTRATION HOOK-ORDER TESTS PASSED" in result.stdout
