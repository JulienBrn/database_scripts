import pandas as pd, numpy as np, re

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
    
def add_to_error(f):
    def new_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            for arg in args:
                display(arg)
            raise
    return new_f

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