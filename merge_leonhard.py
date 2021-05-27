import os
import os.path

print("please type the ssh location of the repository (e.g. faberf@login.leonhard.ethz.ch:~/BPTT-Lipschitzness/ alternatively type 'f' or 'j'")

repo = input()

if repo=="f":
    repo = "faberf@login.leonhard.ethz.ch:~/BPTT-Lipschitzness/"
if repo=="j":
    repo = "jubuechle@login.leonhard.ethz.ch:~/BPTT-Lipschitzness/"

os.path.join(repo, "Sessions")
s = os.path.join(repo, "Sessions")
r = os.path.join(repo, "Resources")

os.system(f"scp -r {s} from_leonhard_sessions/")

os.system(f"scp -r {r} from_leonhard_resources/")

print("Press Enter to continue")

input()

os.system("python merge.py -sessions_dirs from_leonhard_sessions Sessions -resources_dirs from_leonhard_resources Resources")