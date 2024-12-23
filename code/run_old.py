import papermill, subprocess
import sys, yaml, shutil, time
from pathlib import Path
import general.helper as helper

# It is suggested to create the following alias on your system
#alias dbrun='conda run -n dbscripts python /pathto/run.py'
helper_files = ["**/helper.py", "**/config_adapter.py"]

param_path = Path(sys.argv[1]).resolve()
script_folder = Path(sys.argv[0]).parent.resolve()
params = yaml.safe_load(param_path.open("r"))
if "run_summary_folder" in params:
    variables =  params["variables"] if "variables" in params else {}
    summary_folder = Path(helper.replace_vals(params["run_summary_folder"], variables))
else:
    summary_folder = Path.home() /"dbrun.run.tmp"
if summary_folder.exists():
    shutil.rmtree(summary_folder)
(summary_folder/"code").mkdir(parents=True)
for file in helper_files:
    for f in list(script_folder.glob(file)):
        shutil.copyfile(f, summary_folder/"code"/f.name)
papermill.execute_notebook(script_folder/"run/run.ipynb", summary_folder/"code"/"run.ipynb", parameters=dict(param_path=str(param_path), scripts_folder=str(script_folder)), cwd=summary_folder/"code")
subprocess.run(f'jupyter nbconvert --to html {summary_folder/"code"/"run.ipynb"}', shell=True, check=True)
shutil.move(summary_folder/"code"/"run.html", summary_folder/"run.html")

print(f'Summary: file://{summary_folder/"run.html"}')