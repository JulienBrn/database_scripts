from __future__ import annotations
from pathlib import Path
import yaml
import re
import pandas as pd, numpy as np
from typing import Dict, Any, Callable, Tuple

class method: pass


class TagLoader(yaml.SafeLoader):pass
def construct_undef(self, node):
    if isinstance(node, yaml.nodes.ScalarNode):
        value = self.construct_scalar(node)
    elif isinstance(node, yaml.nodes.SequenceNode):
        value = self.construct_sequence(node)
    elif isinstance(node, yaml.nodes.MappingNode):
        value = self.construct_mapping(node)
    return dict(method=node.tag[1:], method_params=value)
TagLoader.add_constructor(None, construct_undef)

def load(s):
    if isinstance(s, str):
        s = Path(s)
    if isinstance(s, Path):
        with s.open("r") as f:
            return yaml.load(f, Loader=TagLoader)
    else:
        return yaml.load(s, Loader=TagLoader)
    
def loads(s):
    return yaml.load(s, Loader=TagLoader)

def normalize_yaml_paramlist(config, format):
    if config is None:
        return []
    if isinstance(config, dict):
        if "key" in format:
            key = format["key"]
            def get_item(k, v):
                if isinstance(v, dict):
                    if key in v:
                        raise Exception("Problem")
                    if "implicit" in format and format["implicit"] not in v:
                        item={key:k, format["implicit"]:v}
                    else: 
                        item = {key:k} | v
                elif "implicit" in format:
                    item = {key:k, format["implicit"]:v}
                return item
            config = [get_item(k,v) for k,v in config.items()]
    if isinstance(config, list):
        ret = []
        for item in config:
            if not isinstance(item, dict):
                if "implicit" in format:
                    item={format["implicit"]:item}
                else:
                    raise Exception("Problem")
            ret.append(item)
        if format["key_unique"]:
            if pd.Series([i[format["key"]] for i in ret]).duplicated().any():
                raise Exception("Problem")
        return ret
            
variable_param_format = dict(
    key="var_name",
    key_unique=True,
    implicit="value"
)

table_param_format = dict(
    key="table_name",
    key_unique=True,
    implicit="table_def"
)

def raw(ctx, value):
    return value

def expand_envvars(ctx, value):
    import os
    return os.path.expandvars(str(ctx.evaluate(value)))

def find_files(ctx, params):
    if set(params.keys()) != {"column_name", "base_folder"}:
        raise Exception(f"find_files param problem {set(params.keys())}")
    base_folder = Path(ctx.evaluate(params["base_folder"]))
    files = [str(f.resolve()) for f in base_folder.glob("**/*")]
    return pd.Series(files, name=ctx.evaluate(params["column_name"])).to_frame().reset_index(drop=True)

def regex_filter(ctx, params):
    if set(params.keys()) != {"table", "column_name", "pattern"}:
        raise Exception(f"regex_filter param problem {set(params.keys())}")
    table: pd.DataFrame = ctx.tables[params["table"]]
    column: str = ctx.evaluate(params["column_name"])
    pattern = ctx.evaluate(params["pattern"])
    filter = table[column].str.fullmatch(pattern)
    return table.loc[filter]

def from_rows(ctx, params):
    if set(params.keys()) != {"column_names", "data"}:
        raise Exception(f"from_rows param problem {set(params.keys())}")
    data = ctx.evaluate(params["data"])
    columns = ctx.evaluate(params["column_names"])
    return pd.DataFrame(data, columns=columns)

def hash(ctx, params: str):
    import hashlib
    m = hashlib.sha256()
    m.update(params.encode("utf-8"))
    return m.hexdigest()[:20]

def longest_prefix_join(ctx, params):
    start_column = ctx.evaluate(params["start_table"]["column"])
    start_table = ctx.tables[ctx.evaluate(params["start_table"]["name"])].sort_values(start_column)
    for other in params["other_tables"]:
        other_column =  ctx.evaluate(other["column"])
        other_table = ctx.tables[ctx.evaluate(other["name"])].sort_values(other_column)
        matches = np.searchsorted(other_table[other_column], start_table[start_column])
        matches = np.where(matches >= len(other_table.index), len(other_table.index)-1, matches)
        print(matches.min(), matches.max(), len(other_table.index), matches.size, len(start_table.index))
        start_table["left_match"] = other_table[other_column].iloc[matches].to_numpy()
        start_table["right_match"] = other_table[other_column].iloc[matches-1].to_numpy()
        # start_table = start_table.merge(other_table[["match_num", other_column]].rename(columns=dict(other_column="left_match")), how="left", on="match_num")
        # start_table = start_table.merge(other_table[["match_num", other_column]].shift(-1).rename(columns=dict(other_column="right_match")), how="left", on="match_num")
        print(start_table.to_string())
        import os
        start_table["n_left"] = start_table.apply(lambda row: len(os.path.commonprefix([row[start_column], row["left_match"]])), axis=1)
        start_table["n_right"] = start_table.apply(lambda row: len(os.path.commonprefix([row[start_column], row["right_match"]])), axis=1)
        start_table[other_column] = np.where(start_table["n_left"] > start_table["n_right"],  start_table["left_match"],  start_table["right_match"])
        start_table = start_table.merge(other_table, on=other_column, how="left")
    return start_table.drop(columns=["n_left", "n_right", "left_match", "right_match"])
        



class Context:
    variables: Dict[str, Any]
    tables: Dict[str, pd.DataFrame]
    methods: Dict[str, Callable[[Context, Any], Any]]
    def __init__(self, default_methods="minimal"):
        self.variables = {}
        self.tables = {}
        if default_methods=="minimal":
            self.methods=dict(raw=raw, hash=hash, expand_envvars=expand_envvars, from_rows=from_rows)
        else:
            self.methods= {}

    def evaluate(self, value, on_undef="raise"):
        if isinstance(value, str):
            if on_undef =="ignore":
                class FormatDict(dict):
                    def __missing__(self, key):
                        return "{" + key + "}"
                import string
                formatter = string.Formatter()
                mapping = FormatDict(**self.variables)
                return formatter.vformat(value, (), mapping)
            else:
                return value.format(**self.variables)
        elif isinstance(value, dict) and set(value.keys()) == {"method", "method_params"}:
            method = value["method"]
            params = value["method_params"]
            if method in self.methods:
                return self.methods[method](self, params)
            else:
                if on_undef=="ignore":
                    return {k: self.evaluate(v, on_undef=on_undef) for k,v in value.items()}
                else:
                    raise Exception(f"Unknown method {method}")
        elif isinstance(value, dict):
            return {k: self.evaluate(v, on_undef=on_undef) for k,v in value.items()}
        elif isinstance(value, list):
            return [self.evaluate(v, on_undef=on_undef) for v in value]
        else: return value

    def __str__(self):
        return f"""
        vars = {self.variables}
        tables = {self.tables}
        methods = {self.methods}
        """
    def __repr__(self):
        return f"""
        vars = {self.variables}
        tables = {self.tables}
        methods = {self.methods}
        """

            

def add_variable_context(current: Context, item):
    if set(item.keys()) != {"var_name", "value"}:
        raise Exception("Problem")
    var_name = item["var_name"]
    var_value = item["value"]
    if var_name in current.variables:
        raise Exception("problem")
    current.variables[var_name] = current.evaluate(var_value)

def add_table_context(current: Context, item):
    # if set(item.keys()) != {"table_name", "table_def"}:
    #     raise Exception("Problem")
    if "duplicate_over" in item:
        tables = item["duplicate_over"]["tables"]
        if not isinstance(tables, list): tables = [tables]
        join = item["duplicate_over"]["join"] if "join" in item["duplicate_over"] else "auto"
        if len(tables) ==1:
            table = current.tables[tables[0]]
        else:
            raise Exception("not handled yet")
        variables = list(table.columns) if not "variables" in item["duplicate_over"] else item["duplicate_over"]["variables"]
        if not isinstance(variables, list): variables = [variables]

        table = table[variables]
        filters = [] if not "filters" in item["duplicate_over"] else  item["duplicate_over"]["filters"] 
        if not isinstance(filters, list): filters = [filters]
        for filter in filters:
            raise Exception("not handled yet")
        duplication = table.drop_duplicates()
    else:
        duplication = pd.DataFrame([[]])
    for _, row in duplication.iterrows():
        for k,v in row.items():
            if k in current.variables:
                raise Exception("problem")
            else:
                current.variables[k] = v
        name = current.evaluate(item["table_name"])
        var_value = item["table_def"]
        if name in current.variables or name in current.tables:
            raise Exception("problem")
        current.tables[name] = current.evaluate(var_value)
        if "postprocessing" in item:
            post = [item["postprocessing"]] if not isinstance(item["postprocessing"], list) else item["postprocessing"]
            for i in post:
                i["method_params"]["table"] = name
                current.tables[name] = current.evaluate(i)
        for k in row.keys():
            del current.variables[k]

def get_duplicate_table(ctx: Context, item):
    if "duplicate_over" in item and item["duplicate_over"]:
        tables = item["duplicate_over"]["tables"]
        if not isinstance(tables, list): tables = [tables]
        join = item["duplicate_over"]["join"] if "join" in item["duplicate_over"] else "auto"
        if len(tables) ==1:
            table = ctx.tables[tables[0]]
        else:
            raise Exception("not handled yet")
        variables = list(table.columns) if not "variables" in item["duplicate_over"] else item["duplicate_over"]["variables"]
        if not isinstance(variables, list): variables = [variables]

        table = table[variables]
        filters = [] if not "filters" in item["duplicate_over"] else  item["duplicate_over"]["filters"] 
        if not isinstance(filters, list): filters = [filters]
        for filter in filters:
            raise Exception("not handled yet")
        duplication = table.drop_duplicates()
    else:
        duplication = pd.DataFrame([[]])
    # print(len(duplication.index))
    return duplication

def handle_duplicate_over(duplication: pd.DataFrame, item):
    if "duplicate_over" in item:
        del item["duplicate_over"]
    items = []
    # print(duplication.to_dict(orient="index"))
    for _, row in duplication.to_dict(orient="index").items():
        ctx=Context(default_methods=None)
        for k,v in row.items():
            ctx.methods["raw"] = raw
            ctx.variables[k] = v
        items.append(ctx.evaluate(item, on_undef="ignore"))
    return items
