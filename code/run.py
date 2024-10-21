import papermill, subprocess
import sys, yaml, shutil
from pathlib import Path
import helper

param_path = Path(sys.argv[1])
params = yaml.safe_load(param_path.open("r"))
if "run_summary_folder" in params:
    variables =  params["variables"] if "variables" in params else {}
    summary_folder = Path(helper.replace_vals(params["run_summary_folder"], variables))
else:
    summary_folder = Path(".run.tmp")
if summary_folder.exists():
    shutil.rmtree(summary_folder)
summary_folder.mkdir(parents=True)
shutil.copy(Path(sys.argv[0]).parent/"helper.py", summary_folder/"helper.py")
papermill.execute_notebook(Path(sys.argv[0]).parent/"run.ipynb", summary_folder/"run.ipynb", parameters=dict(param_path=str(param_path), import_folder=str(Path(sys.argv[0]).parent)), cwd=summary_folder)
subprocess.run(f'jupyter nbconvert --to html {summary_folder/"run.ipynb"}', shell=True, check=True)