import pandas as pd, numpy as np, re, xarray as xr
from pathlib import Path

def generate_duplicate_table(param_data, selection={}, *, drop_column= "__drop__", ):
    if not isinstance(param_data, list):
        param_data = [param_data]
    dfs= []
    for item in param_data:
        df = pd.DataFrame([[np.nan]], columns=[drop_column])
        for k, matches in item.items():
            if k==drop_column:
                raise Exception("drop_column problem")
            if not isinstance(matches, list):
                matches = [matches]
            if k in selection.keys():
                values = []
                for val in selection[k]:
                    for m in matches:
                        if isinstance(m, str): 
                            if re.fullmatch(m, str(val)):
                                values.append(val)
                        elif m==val:
                            values.append(val)
            else:
                values = matches
            df = pd.merge(df, pd.Series(values).to_frame(k), how="cross")
        dfs.append(df.drop(columns=drop_column))
    return pd.concat(dfs)


def replace_vals(d, replacements):
    if isinstance(d, list):
        return [replace_vals(i, replacements) for i in d]
    elif isinstance(d, dict):
        return {k: replace_vals(v, replacements) for k, v in d.items()}
    elif isinstance(d, str):
        try:
            return d.format(**replacements)
        except:
            print(d)
            print(replacements)
            raise
    else:
        return d
    

def json_merge(*d, incompatible="raise"):
    def json_merge_impl(d1, d2):
        if d1 == {}:
            return d2
        if d2 == {}:
            return d1
        if isinstance(d1, dict) and isinstance(d2, dict):
            merge = {k:json_merge(v, d2[k], incompatible=incompatible) for k,v in d1.items() if k in d2}
            return {k:v for k,v in d1.items() if not k in d2} | {k:v for k,v in d2.items() if not k in d1} |  {k:v for k,v in merge.items() if not v=={}}
        elif isinstance(d1, list) and isinstance(d2, list):
            return d1+d2
        else:
            if not d1==d2:
                if pd.isna(d1):
                    return d2
                if pd.isna(d2):
                    return d1
                if incompatible == "raise":
                    raise Exception(f"problem merging dictionaries... {d1} {d2} {pd.isna(d2)} {pd.isna(d1)}")
                elif incompatible == "remove":
                    return {}
            return d1  
    if len(d) ==0:
        return {}
    curr = d[0]
    for di in d[1:]:
        curr= json_merge_impl(curr, di)
    return curr

def singleglob(p: Path, *patterns, search_upward_limit=None, error_string='Found {n} candidates for pattern {patterns} in folder {p}', only_ok=False):
    all = [path for pat in patterns for path in p.glob(pat)]
    if search_upward_limit is not None and len(all) == 0 and search_upward_limit != p:
        return singleglob(p.parent, *patterns, search_upward_limit=search_upward_limit, error_string=error_string)
    if only_ok:
        return len(all)==1
    if len(all) >1:
        raise FileNotFoundError(error_string.format(p=p, n=len(all), patterns=patterns))
    if len(all) ==0:
        raise FileNotFoundError(error_string.format(p=p, n=len(all), patterns=patterns))
    return all[0]

def compute_relevant(event_channels_df, config, copy_columns=[], myeval=lambda df, expr: df.eval(expr)):
    df = pd.DataFrame()
    df["t"] = event_channels_df["t"]
    for col in copy_columns:
        df[col] = event_channels_df[col]
    for param, expr in config["method_params"].items():
        if param.endswith("_expr"):
            df[param.replace("_expr", "_value")] = myeval(event_channels_df, expr)
        else:
            df[param] = expr
    if "metadata" in config:
        metadata = pd.DataFrame()
        for k, v in config["metadata"].items():
            if k.endswith("_expr"):
                metadata[k.replace("_expr", "")] = myeval(event_channels_df, v)
            else:
                metadata[k] = v
        df["metadata"] = metadata.apply(lambda row: {k:v for k, v in row.items() if not pd.isna(v)}, axis=1)
    else:
        df["metadata"] = [{}] * len(df.index)
    filtered = df.loc[df["filter_value"]] if "filter_expr" in config["method_params"] else df
    if "state_expr" in config["method_params"]:
        relevant = filtered.loc[filtered["state_value"] != filtered["state_value"].shift(1)].copy()
        relevant["duration"] = relevant["t"].shift(-1) - relevant["t"]
    else:
        relevant = filtered.copy()
    return relevant



import uuid
import json, yaml
class RenderJSON(object):
    def __init__(self, json_data):
        if isinstance(json_data, dict) or isinstance(json_data, list):
            self.json_str = json.dumps(json_data, indent=2)
            self.yaml_str = yaml.dump(json_data)
        else:
            self.json_str = json_data
            self.yaml_str = json_data
        self.uuid = str(uuid.uuid4())
        self.json_data_uuid = str(uuid.uuid4())
        self.yamlbutton = str(uuid.uuid4())
        self.yaml_data_uuid = str(uuid.uuid4())
        self.jsonbutton = str(uuid.uuid4())
        self.data= json_data
    def get_fullhtml(self):
        return f"""
            <script src="https://unpkg.com/@alenaksu/json-viewer@2.0.0/dist/json-viewer.bundle.js"></script>
            <button type="button" id="{self.yamlbutton}">YAML</button>
            <button type="button" id="{self.jsonbutton}">JSON</button>
            <json-viewer id="{self.uuid}" ></json-viewer>
            <script id="{self.json_data_uuid}" type="application/json">{self.json_str}</script>
            <script id="{self.yaml_data_uuid}" type="application/yaml">{self.yaml_str}</script>
            <script>
                (function (){{
                const jsontext = document.getElementById('{self.json_data_uuid}').textContent
                const yamltext = document.getElementById('{self.yaml_data_uuid}').textContent
                const data = JSON.parse(jsontext)
                document.getElementById('{self.uuid}').data = data;
                const yamlbutton = document.getElementById('{self.yamlbutton}'); 
                yamlbutton.addEventListener('click', function() {{ 
                  const blob = new Blob([yamltext], {{ type: 'text/yaml' }}); 
                  const url = window.URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = 'exported_data.yaml';
                  document.body.appendChild(a);
                  a.click();
                  document.body.removeChild(a);
                  window.URL.revokeObjectURL(url);
                }}); 
                const jsonbutton = document.getElementById('{self.jsonbutton}'); 
                jsonbutton.addEventListener('click', function() {{ 
                  const blob = new Blob([jsontext], {{ type: 'text/json' }}); 
                  const url = window.URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = 'exported_data.json';
                  document.body.appendChild(a);
                  a.click();
                  document.body.removeChild(a);
                  window.URL.revokeObjectURL(url);
                }}); 
                }})();
            </script>
        """
    def get_html(self):
        self.json_str = json.dumps([self.data])
        return f"""<json-viewer id="{self.uuid}">{self.json_str}</json-viewer>"""

    def _ipython_display_(self):
        from IPython.display import display_javascript, display_html, display
        # style=<style>
        #     json-viewer {{
        #     --background-color: white;
        #     --color: 0d37f6;
        #     --string-color:pink;
        #     }}
        #     </style>
        display_html(self.get_fullhtml(), raw=True)


