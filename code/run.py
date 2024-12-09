import papermill, subprocess
import sys, yaml, shutil, time, json
from pathlib import Path
import argparse
import general.helper as helper
import general.config_adapter as config_adapter
import nbformat as nbf
import pandas as pd
import nbconvert, nbformat
import tqdm.auto as tqdm, logging, beautifullogger
import datetime

logger=logging.getLogger()


# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_colwidth', 20)

class ExecutePreprocessorWithCallbacks(nbconvert.preprocessors.ExecutePreprocessor):
    def preprocess(
        self, nb, resources = None, km = None, callbacks=[], cell_callbacks=[]
    ):
        from nbclient.client import NotebookClient
        NotebookClient.__init__(self, nb, km)
        self.reset_execution_trackers()
        self._check_assign_resources(resources)

        with self.setup_kernel():
            assert self.kc
            info_msg = self.wait_for_reply(self.kc.kernel_info())
            assert info_msg
            self.nb.metadata["language_info"] = info_msg["content"]["language_info"]
            callbacks_dict = {i: [k for k in callbacks] for i in range(len(self.nb.cells))}
            for i, c in cell_callbacks:
                callbacks_dict[i].append(c)
            for index, cell in enumerate(self.nb.cells):
                self.preprocess_cell(cell, resources, index)
                for c in callbacks_dict[index]:
                    c(index, cell)
        self.set_widgets_metadata()

        return self.nb, self.resources


def execute_notebook(nb, kernel_name, dest_notebook_file: Path, dest_notebook_html: Path, callbacks=[], cell_callbacks=[]):
    tee_def=f"""
class Tee:
    def __init__(self, *files):
        self._files = files

    def __del__(self):
        # don't kill them here!
        '''
        if self._file1 != sys.stdout and self._file1 != sys.stderr:
            self.file1.close()
        if self._file2 != sys.stdout and self._file2 != sys.stderr:
            self.file2.close()
        '''

    def write(self, string):
        for file in self._files:
            file.write(string)

    def flush(self):
        for file in self._files:
            file.flush()
"""

    nice_stdout=u'''
import sys, os
sys.stdin = Tee(sys.__stdin__, sys.stdin)
sys.stdout = Tee(sys.__stdout__, sys.stdout)
sys.stderr = Tee(sys.__stderr__, sys.stderr)
    ''' 
    ep = ExecutePreprocessorWithCallbacks(timeout=-1, kernel_name=kernel_name)
    # nb["cells"].append(nbf.v4.new_code_cell(variable_initialization))
    nb["cells"] = [nbf.v4.new_code_cell(tee_def), nbf.v4.new_code_cell(nice_stdout)] + nb["cells"]
    try:
        ep.preprocess(nb, {'metadata': {'path': str(dest_notebook_file.parent)}}, callbacks=callbacks, cell_callbacks=[(i+2, c) for i,c in cell_callbacks])
    finally:
        nb["cells"] = nb["cells"][2:]
        try:
            with dest_notebook_file.open('w', encoding='utf-8') as f:
                nbformat.write(nb, f)
        except Exception as e:
            logger.error("Problem exporting notebook as notebook")
        html = nbconvert.exporters.HTMLExporter()
        try:
            html_str, rec = nbconvert.exporters.export(html, nb)
            with dest_notebook_html.open('w', encoding='utf-8') as f:
                f.write(html_str)
        except:
            logger.error("Problem exporting notebook as html")




script_folder = Path(sys.argv[0]).parent.resolve()

parser = argparse.ArgumentParser(
                    prog='run',
                    description='Tool to run a pipeline of jupyter notebooks described using a yaml file',
)

parser.add_argument("pipeline", type=str)
parser.add_argument("summary_folder", type=str, nargs='?', default="default")
parser.add_argument("--kernels", action="store_true")
sysargs = vars(parser.parse_args())


beautifullogger.setup(displayLevel=logging.INFO)

if sysargs["kernels"]:
    logger.info("Declaring kernels...")
    config_dir = subprocess.run("jupyter --config-dir", shell=True, stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
    with (Path(config_dir)/"jupyter_config.json").open("w") as f:
        json.dump({
        "CondaKernelSpecManager": {
        "kernelspec_path": "--user"
        }
    }, f)

    subprocess.run("python -m nb_conda_kernels list", shell=True, stderr=None, stdout=None, check=True)
    subprocess.run("jupyter kernelspec list", shell=True, stderr=None, stdout=None, check=True)

logger.info("Creating notebook")

nb = nbf.v4.new_notebook()
non_local_imports = """
from pathlib import Path
from datetime import datetime
import yaml, tqdm.auto as tqdm, shutil, subprocess, json
import pandas as pd, numpy as np
import networkx as nx
import itables
"""
local_imports = """
import helper
from helper import RenderJSON, singleglob
import config_adapter
"""
itables_configuration = """
itables.init_notebook_mode(all_interactive=True )
itables.options.maxBytes = "1MB"
itables.options.lengthMenu = [25, 10, 50, 100, 200]
itables.options.buttons = ["copyHtml5", "csvHtml5", "excelHtml5"]
itables.options.layout={"topEnd": "pageLength", "top1": "searchBuilder"}
"""

variable_initialization = f"""
script_folder = Path("{script_folder}")
""" + """
ctx = config_adapter.Context()
decl_runs = pd.DataFrame(columns=["id", "script", "script_params", "run_folder", "imports", "environment", "depends_on"])
run_df = pd.DataFrame()
done_runs = pd.DataFrame(columns=["id", "script", "script_params", "run_folder", "imports", "environment", "depends_on"])
dependency_graph = nx.DiGraph()
"""

declare_run_info = """
script_descriptions = pd.read_excel(script_folder/"../descriptions/actions.xlsx", sheet_name="declare_run").fillna("")
script_descriptions["helper_scripts_pattern"] = script_descriptions["helper_scripts_pattern"].apply(lambda l: [p.strip() for p in l.split(',')])
script_descriptions["script_required_params"] = script_descriptions["script_required_params"].apply(lambda l: [s.strip() for s in l.split(',')])
script_descriptions["script_optional_params"] = script_descriptions["script_optional_params"].apply(lambda l: [s.strip() for s in l.split(',')])
script_descriptions
"""


nb["cells"].append(nbf.v4.new_markdown_cell("# Initialization"))
nb["cells"].append(nbf.v4.new_markdown_cell("## Imports"))
nb["cells"].append(nbf.v4.new_code_cell(non_local_imports))
nb["cells"].append(nbf.v4.new_code_cell(local_imports))
nb["cells"].append(nbf.v4.new_markdown_cell("## Configuration"))
nb["cells"].append(nbf.v4.new_code_cell(itables_configuration))
nb["cells"].append(nbf.v4.new_markdown_cell("## Variable initialization"))
nb["cells"].append(nbf.v4.new_code_cell(variable_initialization))
nb["cells"].append(nbf.v4.new_markdown_cell("## Declare run info"))
nb["cells"].append(nbf.v4.new_code_cell(declare_run_info))

action_list = config_adapter.load(sysargs["pipeline"])

def stop_exec(i, cell):
    logger.info("Stop requested")
    raise KeyboardInterrupt


def display_runs(i, cell):
    df = pd.read_json(summary_folder/'code'/"run_desc.json")
    df["script"] = df["script"].apply(lambda p: "*/"+Path(p).name)
    df["run_folder"] = df["run_folder"].apply(lambda p: "*/"+Path(p).name)
    df["imports"] ="n="+ df["imports"].apply(lambda l: len(l)).astype(str)
    df["script_params"] ="n="+ df["script_params"].apply(lambda d: len(d)).astype(str)
    df["kernel"] = df["kernel"].apply(lambda p: "*/"+Path(p).name)
    print("============================================ Run Display ==========================================")
    print("")
    print(df)
    print("===================================================================================================")
    print("")

def execute_runs(i, cell):
    df = pd.read_json(summary_folder/'code'/"run_desc.json")
    n_tasks = len(df.index)
    already_done = df["id"].loc[(~df["should_run"]) & (df["status"]=="done")].to_list()
    # logger.info(f'{len(already_done)} results skipped')
    import numpy as np
    ignored_df = df.loc[~df["should_run"]].assign(dyn_status="skipped")
    ignored_df["notebook"] = ignored_df["run_folder"].apply(lambda p: Path(p)/"notebook.html" if (Path(p)/"notebook.html").exists() else np.nan)
    def get_metadata(p, val):
        if (Path(p)/"run_metadata.yaml").exists():
            with (Path(p)/"run_metadata.yaml").open("r") as f:
                d = yaml.safe_load(f)
                if val in d:
                    return d[val]
        return np.nan
    ignored_df["duration"] = ignored_df["run_folder"].apply(lambda p:  get_metadata(p, "run_duration"))
    ignored_df = ignored_df[["id", "dyn_status", "duration", "notebook"]]
    df = df.loc[df["should_run"]]
    tasks = list(df.to_dict(orient="index").values())
    # progress = tqdm.tqdm(desc="Executing", total=len(tasks), disable=len(tasks)==0)
    results=[]
    errors = []
    print("")
    while len(tasks) > 0:
        task = tasks[0]
        id = task["id"]
        display_df = pd.concat([d for d in [
            ignored_df, pd.DataFrame(results), pd.DataFrame([dict(id=id, dyn_status="running", duration=np.nan, notebook=np.nan)]),
            pd.DataFrame([t["id"] for t in tasks[1:]], columns=["id"]).assign(dyn_status="queued", duration=np.nan, notebook=np.nan)
        ] if not len(d.index)==0], ignore_index=True)
        display_str = display_df.assign(notebook=('file://'+display_df.pop("notebook").astype(str)).str.replace('file://nan', 'nan', regex=False)).to_markdown()
        print(display_str)
        # progress.set_postfix_str(f"running {id}")
        if len(set(task["rec_depends_on"]) - set(already_done)) > 0:
            raise Exception(f'Problem {set(task["rec_depends_on"]) - set(already_done)}')
        run_folder = Path(task["run_folder"]+".tmp")
        if run_folder.exists():
            shutil.rmtree(run_folder)
        run_folder.mkdir(exist_ok=True, parents=True)
        for im in task["imports"]:
            shutil.copyfile(Path(im), run_folder/Path(im).name)
        with Path(task["script"]).open("r") as f:
            nb = nbformat.read(f, as_version=4)
        with (run_folder / "params.yaml").open("w") as f:
            yaml.dump(task["script_params"], f)
        start_time = time.time()
        try:
            execute_notebook(nb, task["kernel"], run_folder/"notebook.ipynb", run_folder/"notebook.html")
            status="sucess"
        except Exception as e:
            e.add_note(f'While executing task {id}')
            errors.append(e)
            status=f"failed"
            consequently_failed = [id for t in tasks if id in t["rec_depends_on"]]
            already_done+=consequently_failed
            left_tasks = [t for t in tasks if not id in t["rec_depends_on"]]
            for cid in consequently_failed:
                results.append(dict(id=cid, dyn_status=f"fail_dynpred_{id}"))
            tasks = left_tasks
        finally:
            duration = time.time() - start_time
            with (run_folder/"run_metadata.yaml").open("w") as f:
                yaml.dump(dict(run_duration=duration), f)
            # progress.total-= len(consequently_failed)
        
        if status=="sucess":
            if Path(task["run_folder"]).exists():
                shutil.rmtree(Path(task["run_folder"]))
            shutil.move(run_folder, task["run_folder"])
            notebook_html = f"{Path(task['run_folder']).resolve()/'notebook.html'}"
            # logger.info(f"Task {id} was a success. Notebook html available at: file://{Path(task['run_folder']).resolve()/'notebook.html'}")
        else: 
            notebook_html = f"{run_folder.resolve()/'notebook.html'}"
            # print(errors[-1])
            # logger.error(f"Task {id} failed. Notebook html available at: file://{run_folder.resolve()/'notebook.html'}")
        results.append(dict(id=id, dyn_status=status, duration=duration, notebook=notebook_html))
        # progress.update(1)
        already_done.append(id)
        tasks = tasks[1:]
        print(f"\033[{n_tasks+2}A", end= "")
        print("\033[J", end= "")
    results = pd.DataFrame(results, columns=["id", "dyn_status", "duration"])
    results.to_json(summary_folder/'code'/'run_result.json')
    print("")
    logger.info("Summary table")
    display_df = pd.concat([d for d in [
        ignored_df, pd.DataFrame(results),
    ] if not len(d.index) ==0], ignore_index=True)
    display_str = display_df.assign(notebook=('file://'+display_df.pop("notebook").astype(str)).str.replace('file://nan', 'nan')).to_markdown()
    print(display_str)


templates = dict(
    declare_var=dict(
        required_args=["name", "value"],
        optional_args = {},
        callbacks=[],
        update_tags={},
        tag_triggers=[],
        code=[
r"""
items_added = []
for item in unfolded_items:
    item = ctx.evaluate(item)
    if item["name"] in ctx.variables and ctx.variables[item["name"]] != item["value"]:
      raise Exception(f'Variable {item["name"]} already exists')
    ctx.variables[item["name"]] = item["value"]
    items_added.append(item["name"])
display(RenderJSON(dict(added_variables={k: ctx.variables[k] for k in items_added})))
"""
        ]
    ),
    declare_table=dict(
        required_args=["name", "table"],
        optional_args = {},
        callbacks=[],
        update_tags={},
        tag_triggers=[],
        code=[
r"""
tables_added = []
for item in unfolded_items:
    item = ctx.evaluate(item)
    name = item["name"]
    table = item["table"]
    table.style.set_caption(f'Table {name}')
    if name in ctx.tables and ctx.variables[name] != table:
      raise Exception(f'Table {name} already exists')
    ctx.tables[name] = table
    items_added.append(name)
    display(table)
display(RenderJSON(dict(added_tables=tables_added)))
"""
        ]
    ),
    stop=dict(
        required_args=[],
        optional_args = {},
        callbacks=[(0, stop_exec)],
        update_tags={},
        tag_triggers=[],
        code=[]
    ),
    declare_run=dict(
        required_args=["id", "script", "script_params", "run_folder"],
        optional_args = {"imports": "_auto_", "environment": "_auto_", "depends_on": [], "recompute": "on_error", "metadata":{}},
        callbacks=[],
        update_tags=dict(declarations_changed=True),
        tag_triggers=[],
        code=[
r"""
new_decl_runs = pd.DataFrame([ctx.evaluate(k) for k in unfolded_items])
if len(new_decl_runs.index) ==0:
    new_decl_runs = pd.DataFrame([], columns =["id", "script", "script_params", "run_folder", "imports", "environment", "depends_on", "recompute", "metadata"])
    print(f"WARNING, no runs being declared from {base_item}")
else:
    new_decl_runs = new_decl_runs.drop(columns="action")
tmp = new_decl_runs.merge(script_descriptions, how="left", on="script", indicator="_has_info")
unmatched_scripts = tmp[tmp["_has_info"]!= "both"]["script"].drop_duplicates().to_list()
if len(unmatched_scripts) >0:
    print(f'The following scripts where not matched {unmatched_scripts}')
tmp["script"] = np.where(tmp["_has_info"] == "both", tmp["script_path_pattern"], tmp["script"])
for col, dcol in [("environment", "conda_env"), ("imports", "helper_scripts_pattern")]:
  if ((tmp[col] == "_auto_") & (tmp["_has_info"] != "both")).any():
    raise Exception(f'Automatic parameter detection for column {col} not available as script not found in excel sheet')
  tmp[col] = np.where(tmp[col] == "_auto_", tmp[dcol], tmp[col])
def check_params(d, req, opt, script):
  mismatch = set(req).symmetric_difference(set(req)) - set(opt)
  if len(mismatch) > 0:
    raise Exception(f'''
    Params for script {script} do not match specification.
    Missing fields: {list(mismatch.difference(set(req)))}.
    Unknown fields:  {list(mismatch.difference(set(d.keys())))}
''')
tmp.apply(lambda row: check_params(row["script_params"], row["script_required_params"], row["script_optional_params"], row["script"]) if row["_has_info"]=="both" else None, axis=1)
new_decl_runs = tmp[new_decl_runs.columns]
decl_runs = pd.concat([decl_runs, new_decl_runs], ignore_index=True)
if decl_runs["id"].duplicated().any():
    raise Exception("All runs must have unique ids")
if decl_runs["run_folder"].duplicated().any():
    raise Exception("All runs must have a unique run_folder")
for row in new_decl_runs.to_dict(orient="index").values():
    ctx.variables[row["id"]+"_run_folder"] = row["run_folder"]
display(new_decl_runs)
""",
        ]
    ),
    process_new_declarations=dict(
        required_args=[],
        optional_args = {},
        callbacks=[],
        update_tags=dict(declarations_changed=False),
        tag_triggers=[],
        code=[
r"""
valid_ids = set(done_runs["id"]).union(set(decl_runs["id"]))
for row in decl_runs.to_dict(orient='index').values():
    dependency_graph.add_node(row["id"])
    for dep in row["depends_on"]:
        if not dep in valid_ids:
            raise Exception(f"Depends on references an unknown id {dep}")
        dependency_graph.add_edge(dep, row["id"])
if not nx.is_directed_acyclic_graph(dependency_graph):
    cycle = nx.find_cycle(dependency_graph)
    raise Exception(f"Cycle found in task dependencies.\n{cycle}")
""",
r"""
current_graph = dependency_graph.subgraph(list(decl_runs["id"]))
stratified_nodes = [(i, n) for i, strat in  enumerate(nx.topological_generations(current_graph)) for n in strat]
for i, node in stratified_nodes:
    depends_on = []
    for pred in dependency_graph.predecessors(node):
      if "status" in dependency_graph.nodes[pred]:
        if isinstance(dependency_graph.nodes[pred]["status"], list):
          if not pred in list(decl_runs["id"]):
            raise Exception(f'Only already computed status expected here, got {dependency_graph.nodes[pred]["status"]}')
          depends_on+=dependency_graph.nodes[pred]["status"] + [pred]
        elif dependency_graph.nodes[pred]["status"].startswith("fail"):
            depends_on = f"fail_{pred}"
            continue
    dependency_graph.nodes[node]["status"] = depends_on
    dependency_graph.nodes[node]["run_group"] = i
""",
r"""
run_df = decl_runs.copy()
run_df["script"] = run_df["script"].apply(lambda s: str(helper.singleglob(script_folder, s).resolve()))
run_df["imports"] = run_df["imports"].apply(lambda l: [str(helper.singleglob(script_folder, s).resolve()) for s in l])
run_df["rec_depends_on"] = run_df["id"].apply(lambda id: dependency_graph.nodes[id]["status"])
run_df["run_group"] = run_df["id"].apply(lambda id: dependency_graph.nodes[id]["run_group"])
run_df["status"] = run_df["run_folder"].apply(lambda p: "error" if Path(p + ".tmp").exists() else "done" if Path(p).exists()  else "todo")
run_df["should_run"] = ((run_df["recompute"] == "always") | (run_df["status"] == "todo") | ((run_df["recompute"] == "on_error") & (run_df["status"] == "error"))) & ~run_df["rec_depends_on"].astype(str).str.startswith("fail")
run_df["kernel"] = "conda-env-"+run_df["environment"].astype(str)+"-py"
run_df["initial_index"] = np.arange(len(run_df.index))
run_df = run_df.sort_values(["run_group", "initial_index"]).drop(columns="initial_index")
display(run_df)
""",
        ]
    ),
    display_run=dict(
        required_args=[],
        optional_args = {},
        callbacks=[(1, display_runs)],
        update_tags={},
        tag_triggers=[("declarations_changed", dict(action="process_new_declarations"))],
        code=[
r"""
run_df.to_json(Path("run_desc.json"))
""",
r"""
Path("run_desc.json").unlink()
""",
        ],
    ),

    run=dict(
        required_args=[],
        optional_args = {},
        callbacks=[(1, execute_runs)],
        update_tags=dict(declarations_changed=True),
        tag_triggers=[("declarations_changed", dict(action="process_new_declarations"))],
        code=[
r"""
run_df.to_json(Path("run_desc.json"))
""",
r"""
Path("run_desc.json").unlink()
result_df = pd.read_json("run_result.json")
result_df = run_df.merge(result_df, on="id", how="left")
done_runs=pd.concat([done_runs, result_df])
run_df = run_df.iloc[0:0, :]
decl_runs = decl_runs.iloc[0:0, :]
col_display_order = ["id", "status", "should_run", "dyn_status","duration",  "environment", "depends_on", "rec_depends_on"]
display(result_df[col_display_order+[col for col in result_df.columns if not col in col_display_order]])
""",
r"""
for row in result_df.to_dict(orient="index").values():
  if not pd.isna(row["dyn_status"]):
    dependency_graph.nodes[row["id"]]["status"] = row["dyn_status"]
  elif row["status"]=="done":
    dependency_graph.nodes[row["id"]]["status"] = "already_computed"
""",
        ],
    ),
)

nb["cells"].append(nbf.v4.new_markdown_cell("# Processing"))
cell_callbacks=[]
update_tags = {}
i=0
summary_folder=None
while action_list[0]["action"] == "set":
    if "summary_folder" in action_list[0]:
        summary_folder= Path(config_adapter.Context().evaluate(action_list[0]["summary_folder"]))
    action_list = action_list[1:]
while len(action_list) > 0:
    action = action_list[0]
    if isinstance(action, str):
        action = dict(action=action)
    if not isinstance(action, dict):
        raise Exception("Actions must be dictionaries")
    action_name = action["action"]
    if  action_name not in templates:
        raise Exception(f'Unknown action {action_name}.\nKnown actions are {templates.keys}')
    template = templates[action["action"]]
    has_trigger = False
    for k, v in template["tag_triggers"]:
        if update_tags[k]:
            action_list = [v] + action_list
            has_trigger = True
    if has_trigger:
        continue
    nb["cells"].append(nbf.v4.new_markdown_cell(f"## Processing {i}: {action_name}"))
    action = template["optional_args"] | action
    action = dict(duplicate_over={}) | action
    mismatch = set(action.keys()).symmetric_difference(set(template["required_args"]).union(set(template["optional_args"].keys()), {"action", "duplicate_over"}))
    if len(mismatch) > 0:
        raise Exception(f"""
Fields for action {action_name} do not match specification.
    Missing fields: {list(mismatch.difference(set(action.keys())))}.
    Unknown fields:  {list(mismatch.difference(set(template["required_args"]).union(set(template["optional_args"].keys()), {"action", "duplicate_over"})))}
                         """)
    item_decl = f'base_item = json.loads(r"""{json.dumps(action, indent=2)}""")'
    duplication_handling = """
duplicate_table = config_adapter.get_duplicate_table(ctx, base_item)
unfolded_items = config_adapter.handle_duplicate_over(duplicate_table, base_item)
    """
    duplication_display = "display(RenderJSON(dict(item=base_item, unfolded=unfolded_items)))"
    nb["cells"].append(nbf.v4.new_code_cell("\n".join([item_decl, duplication_handling, duplication_display])))
    cell_callbacks=cell_callbacks+[(i+len(nb["cells"])-1, c) for i, c in template["callbacks"]]
    for c in template["code"]:
        nb["cells"].append(nbf.v4.new_code_cell(c))
    for k, v in template["update_tags"].items():
        update_tags[k] = v
    i+=1
    action_list = action_list[1:]

nb["cells"].append(nbf.v4.new_markdown_cell("## End"))   
# nbf.write(nb, summary_folder/'code'/'run_notebook.ipynb')

logger.info("Adding execution context")

if sysargs["summary_folder"] =="default":
    if summary_folder is None:
        summary_folder = Path(f"~/Documents/Other/dbruns/{datetime.datetime.now().timestamp()}").expanduser()
else:
    summary_folder = Path(sysargs["summary_folder"])
if summary_folder.exists():
    if (summary_folder/'code').exists():
        shutil.rmtree(summary_folder)
    else:
        raise Exception(f'Summary folder {summary_folder} should be empty or already a run folder')
(summary_folder/'code').mkdir(exist_ok=True, parents=True)

helper_files = ["**/helper.py", "**/config_adapter.py"]
for file in helper_files:
    for f in list(script_folder.glob(file)):
        shutil.copyfile(f, summary_folder/"code"/f.name)


logger.info("Executing notebook")
def print_cell(i, cell):
    print(dict(index=i) | {k:v if k!="outputs" else len(v) for k,v in cell.items()})

try:
    execute_notebook(nb, "python3", summary_folder/'code'/'notebook.ipynb', summary_folder/'notebook.html', cell_callbacks=cell_callbacks)
    logger.info(f"Ouput notebook html: file://{summary_folder.resolve()/'notebook.html'}")
except Exception:
    logger.error(f"Ouput notebook html: file://{summary_folder.resolve()/'notebook.html'}")
    raise
except KeyboardInterrupt:
    logger.info(f"Ouput notebook html: file://{summary_folder.resolve()/'notebook.html'}")
    

# print(cell_callbacks)
# with (summary_folder/'code'/'run_notebook.ipynb').open("r") as f:
#     nb = nbformat.read(f, as_version=4)
# ep = ExecutePreprocessorWithCallbacks(timeout=-1, kernel_name='python3')
# try:
#     ep.preprocess(nb, {'metadata': {'path': str(summary_folder/'code')}}, callbacks=[], cell_callbacks=cell_callbacks)
# finally:
#     try:
#         with (summary_folder/'code'/    'run_notebook.ipynb').open('w', encoding='utf-8') as f:
#             nbformat.write(nb, f)
#     except Exception as e:
#         logger.error("Problem exporting notebook as notebook")
#     html = nbconvert.exporters.HTMLExporter()
#     try:
#         with (summary_folder/'code'/'run_notebook.ipynb').open("r") as f:
#             nb_executed = nbformat.read(f, as_version=4)
#         html_str, rec = nbconvert.exporters.export(html, nb)
#         with (summary_folder/'run_notebook.html').open('w', encoding='utf-8') as f:
#             f.write(html_str)
#         print(f"Ouput notebook html: file://{summary_folder/'run_notebook.html'}")
#         print("")
#     except:
#         logger.error("Problem exporting notebook as html")

    