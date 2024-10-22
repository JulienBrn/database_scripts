import papermill, subprocess
import sys, yaml, shutil
from pathlib import Path
import helper

# It is suggested to create the following alias on your system
#alias dbrun='conda run -n dbscripts python /pathto/run.py'

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
summary_folder.mkdir(parents=True)
shutil.copy(script_folder/"helper.py", summary_folder/"helper.py")
papermill.execute_notebook(script_folder/"run.ipynb", summary_folder/"run.ipynb", parameters=dict(param_path=str(param_path), scripts_folder=str(script_folder)), cwd=summary_folder)
subprocess.run(f'jupyter nbconvert --to html {summary_folder/"run.ipynb"}', shell=True, check=True)