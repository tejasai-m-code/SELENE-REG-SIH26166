import subprocess
import sys
import os

node_dir = r"C:\Users\thann\AppData\Local\Microsoft\WinGet\Packages\OpenJS.NodeJS.LTS_Microsoft.Winget.Source_8wekyb3d8bbwe\node-v24.19.0-win-x64"
os.environ["PATH"] = node_dir + os.pathsep + os.environ.get("PATH", "")
os.environ["PORT"] = "5000"
os.environ["PYTHONPATH"] = os.path.join(r"d:\Branches\Projects\SIH26166\SELENE-REG-X-SIH26166", "artifacts", "api-server", "python")

launch_mjs = os.path.join(r"d:\Branches\Projects\SIH26166\SELENE-REG-X-SIH26166", "scripts", "launch.mjs")
p = subprocess.Popen(["node", launch_mjs], cwd=r"d:\Branches\Projects\SIH26166\SELENE-REG-X-SIH26166", env=os.environ)
p.wait()
