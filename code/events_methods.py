import pandas as pd, numpy as np, xarray as xr
import re
from helper import json_merge

def compute_metadata(join_metadata, **warnings):
    kept_warnings = {}
    for k, v in warnings.items():
        if isinstance(v, dict):
            vals = {v for v in v.values() if not pd.isna(v)}
            if len(vals) > 1:
                kept_warnings[k] = v
        else:
            if v[0]:
                kept_warnings[k] = v[1]
    if len(kept_warnings) > 0:
        metadata = json_merge(*join_metadata, dict(warnings=kept_warnings))
    else:
        metadata = json_merge(*join_metadata)
    return metadata

def processing_method(goal):
    class PM:
        def __init__(self, fn):
            if callable(goal):
                self.fn = goal(fn)
            else:
                self.fn = fn

        def __set_name__(self, cls, name):
            self.fn.class_name = cls.__name__
            cls.register_method(goal.__name__ if callable(goal) else goal, self.fn.__name__)
            setattr(cls, name, self.fn)
    return PM



class Processings:
    methods={}
    @classmethod
    def call(cls, f_name, *args, _processing_goal=None, **kwargs):
        if not f_name in Processings.methods:
            raise Exception(f"Unknown processing method {f_name}. Suggested: {Processings.methods}")
        if not _processing_goal is None:
            if  Processings.methods[f_name] != _processing_goal:
                raise Exception("Wrong goal for method {f_name}. Goal for {f_name} is {Processings.methods[f_name]}")
        return getattr(cls, f_name)(*args, **kwargs)
    
    @classmethod
    def register_method(cls, goal, f_name):
        if not goal in Processings.methods:
            if f_name in Processings.methods and not Processings.methods[f_name]==goal:
                raise Exception("Several processing methods with same name declared with different goals...")
            Processings.methods[f_name] = goal
        

class Discretize(Processings):
    @processing_method(goal="discretize_method")
    def interupted_sine_wave(data: xr.DataArray, config):
        sine_fs = config["method_params"]["sine_fs"]
        fs = 1/(data["t"] - data["t"].shift(t=1)).mean().item()
        if not (np.abs(data["t"] - np.arange(data.size)/fs) < 0.01/fs).all():
            raise Exception(f"not regular {fs}")

        mabs : xr.DataArray= np.abs(data).rolling(t=int(np.ceil(fs/sine_fs)), center=True).mean().dropna("t")
        threshold = mabs.quantile(0.9, "t").item() *0.6
        # print(threshold, mabs)
        drops = mabs.where((mabs > threshold) & (mabs.shift(t=-1) < threshold), drop=True)
        rises = mabs.where((mabs < threshold) & (mabs.shift(t=-1) > threshold), drop=True)
        resd = pd.DataFrame().assign(t=drops["t"].to_numpy()+0.5/sine_fs, channel_name=config["dest_channel"], State=1)
        resr = pd.DataFrame().assign(t=rises["t"].to_numpy()+0.5/sine_fs, channel_name=config["dest_channel"], State=0)
        # import matplotlib.pyplot as plt
        # f, ax = plt.subplots(1)
        # ax: plt.Axes
        # data.plot(x="t", ax=ax)
        # mabs.plot(x="t", ax=ax, color="green")
        # ax.vlines(resd["t"], data.min().item(), data.max().item(), color="green")
        # ax.vlines(resr["t"], data.min().item(), data.max().item(), color="red")
        # ax.hlines([threshold], 0, data["t"].max(), color="gray")
        # ax.set_xlim(1150, 1153)
        return pd.concat([resd, resr]).sort_values("t")
    
    @processing_method(goal="discretize_method")
    def analog_to_binary(data: xr.DataArray, config):
        rel = config["method_params"]["threshold_rel_max"]
        threshold = data.max().item()*rel
        # print(threshold, mabs)
        drops = data.where((data > threshold) & (data.shift(t=-1) < threshold), drop=True)
        rises = data.where((data < threshold) & (data.shift(t=-1) > threshold), drop=True)
        resd = pd.DataFrame().assign(t=drops["t"].to_numpy(), channel_name=config["dest_channel"], State=0)
        resr = pd.DataFrame().assign(t=rises["t"].to_numpy(), channel_name=config["dest_channel"], State=1)
        import matplotlib.pyplot as plt
        # f, ax = plt.subplots(1)
        # ax: plt.Axes
        # data.plot(x="t", ax=ax)
        # mabs.plot(x="t", ax=ax, color="green")
        # ax.vlines(resd["t"], data.min().item(), data.max().item(), color="green")
        # ax.vlines(resr["t"], data.min().item(), data.max().item(), color="red")
        # ax.hlines([threshold], 0, data["t"].max(), color="gray")
        # ax.set_xlim(1150, 1153)
        return pd.concat([resd, resr]).sort_values("t")
    
class EventProcessing(Processings):
    @staticmethod
    def event_method(f):
        columns = ["event_name", "t", "duration", "n_segments", "metadata", "waveform_changes", "waveform_values"]
        
        def new_f(*args, **kwargs):
            res = f(*args, **kwargs)
            counts = res.count().to_dict()
            if not "metadata" in res:
                res["metadata"] = [{}] *len(res.index)
            # print({k: type(v) for k, v in counts.items()})
            return res[[c for c in columns if c in res.columns and (counts[c] > 0)]]
        new_f.__name__ = f.__name__
        return new_f
    
    @processing_method(goal=event_method)
    def input_binary_wave(relevant: pd.DataFrame, config):
        if relevant["state_value"].iat[0] == 0:
            relevant = relevant.iloc[1:, :].copy()
        relevant["next_metadata"] = relevant["metadata"].shift(-1)
        rises = relevant.iloc[::2].copy()
        rises["metadata"] = rises.apply(lambda row: compute_metadata([row["metadata"], row["next_metadata"]]) , axis=1)
        return rises[["t", "duration", "metadata"]].assign(event_name=config["event_name"], n_segments=1, waveform_changes=[[0]] * len(rises.index), waveform_values = [[0, 1, 0]] * len(rises.index))
    
    @processing_method(goal=event_method)
    def step_wave(relevant, config):
        events = relevant.assign(curr_value= relevant["state_value"], prev_value=relevant["state_value"].shift(1), next_value=relevant["state_value"].shift(-1))
        events["waveform_info"] = events[["curr_value", "prev_value", "next_value"]].apply(lambda row: row.to_dict(), axis=1)
        return relevant[["t", "duration", "metadata"]].assign(event_name=config["event_name"], n_segments=1, 
            waveform_changes=[[0]] * len(relevant.index), 
            waveform_values=[[b, c, a] for b, c, a in zip(relevant["state_value"].shift(1), relevant["state_value"], relevant["state_value"].shift(-1))])

    @processing_method(goal=event_method)
    def event_pulse(relevant, config):
        return relevant[["t", "metadata"]].assign(event_name=config["event_name"], duration=np.nan, n_segments=0)
    
    @staticmethod
    def compute_evdataframe(event_channels_df, config, copy_columns=[], eval_func=lambda df, expr: df.eval(expr)):
        df = pd.DataFrame()
        df["t"] = event_channels_df["t"]
        for col in copy_columns:
            df[col] = event_channels_df[col]
        for param, expr in config["method_params"].items():
            if param.endswith("_expr"):
                df[param.replace("_expr", "_value")] = eval_func(event_channels_df, expr)
            else:
                df[param] = expr
        if "metadata" in config:
            metadata = pd.DataFrame()
            for k, v in config["metadata"].items():
                if k.endswith("_expr"):
                    metadata[k.replace("_expr", "")] = eval_func(event_channels_df, v)
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
    
    @staticmethod
    def process_info(channels, processing_info, dest_name="event_name"):
        event_spec = []
        for item in processing_info:
            if "duplicate_over" not in item:
                item["duplicate_over"] = {}
            from helper import generate_duplicate_table, replace_vals
            duplication = generate_duplicate_table(item["duplicate_over"], dict(channel_name=channels))
            for _, row in duplication.iterrows():
                final_d = replace_vals(item, row.to_dict())
                del final_d["duplicate_over"]
                event_spec.append(final_d)

        unique_df = pd.DataFrame(event_spec)[dest_name].value_counts().reset_index()
        if not (unique_df["count"] == 1).all():
            raise Exception(f'Name duplication ({dest_name} duplicated)\n{unique_df.loc[unique_df["count"] > 1]}')
        return {v[dest_name]: v for v in event_spec}
    
    @staticmethod
    def summarize(data: pd.DataFrame, precision=0.01, min_score=1.1):
        data = data.copy()
        def compute_grp(duration: pd.Series):
            from scipy.cluster.hierarchy import DisjointSet
            res=pd.DataFrame()
            res["duration"]=duration
            res["id"] = np.arange(len(res.index))
            sorted_res = res.dropna(subset="duration").sort_values("duration")
            if len(sorted_res) ==0:
                return duration.apply(lambda v: "nan")
            uf = DisjointSet(sorted_res["id"])
            for i in range(len(sorted_res.index) -1):
                if sorted_res["duration"].iat[i+1] - sorted_res["duration"].iat[i] < precision:
                    uf.merge(sorted_res["id"].iat[i+1],sorted_res["id"].iat[i])
            cluster_df = pd.DataFrame([dict(durations=res["duration"].iloc[list(c)].to_numpy(), ids=list(c)) for i, c in enumerate(uf.subsets())])
            cluster_df["min"] = cluster_df["durations"].apply(lambda l: np.min(l))
            cluster_df["max"] = cluster_df["durations"].apply(lambda l: np.max(l))
            cluster_df["width"] = cluster_df["max"] - cluster_df["min"] 
            cluster_df["size"] = cluster_df["durations"].apply(lambda l: len(l))
            cluster_df["isok"] = cluster_df["width"] < 2*precision
            cluster_df=cluster_df.sort_values("min")
            cluster_df["separation_bef"] = cluster_df["max"] - cluster_df["min"].shift(1)
            cluster_df["separation_aft"] = cluster_df["min"].shift(-1) - cluster_df["max"]
            cluster_df["separation"] = (cluster_df["separation_bef"].fillna(cluster_df["separation_aft"]) + cluster_df["separation_aft"].fillna(cluster_df["separation_bef"])).fillna(1)
            
            filtered_cluster_df = cluster_df.loc[cluster_df["isok"]].copy()
            filtered_cluster_df["score"] = filtered_cluster_df["separation"]/filtered_cluster_df["separation"].max() + filtered_cluster_df["size"]/filtered_cluster_df["size"].sum()
            kept_clusters = filtered_cluster_df.loc[filtered_cluster_df["score"] > min_score]
            res["cluster"] = res["id"].map({v:i for i, c in kept_clusters["ids"].items() for v in c})

            def mk_cluster_name(d):
                min = d.min()
                max = d.max()
                if max-min > precision:
                    name = f'[{min:.3f},{max:.3f}]'
                else:
                    center = (max+min)/2
                    name = f'{center:.3f}±{max-center:.3f}'
                return np.where(d.isna(), "nan", name)
            res["cluster_name"] = res.groupby("cluster", dropna=False)["duration"].transform(mk_cluster_name)
            return res["cluster_name"]
           

        data["duration_goup"]= data.groupby(["event_name", "n_segments"])["duration"].transform(compute_grp)
        warning_list = data["metadata"].apply(lambda d: list(d["warnings"].keys()) if not d is None and "warnings"in d else [])
        all_warnings = set(warning_list.sum())

        def compute_info(grp):
            warnings = {k: 0 for k in all_warnings}
            for m in grp["metadata"].values:
                if m is None: continue
                if "warnings" in m:
                    for k in m["warnings"].keys():
                        if not k in warnings:
                            warnings[k]=0
                        warnings[k]+=1
            return pd.Series(dict(n=len(grp.index)) | {f'warning_{k}': v for k, v in warnings.items()})
        summary = data.groupby(["event_name", "n_segments", "duration_goup"]).apply(compute_info, include_groups=False).reset_index()
        return summary

class FiberEventProcessing(EventProcessing):pass

class PolyEventProcessing(EventProcessing):
    @staticmethod
    def compute_evdataframe(event_channels_df, config):
        def task_eval(df, expr):
            task_expr_pattern = re.compile(r'task\[(?P<col_num>\d+)\]')
            task_expr_df = pd.DataFrame()
            def handle_col(match):
                num = int(match["col_num"])
                name = f'__task_{num}'
                task_expr_df[name] = df["task_params"].apply(lambda l: l[num] if isinstance(l, list) and len(l) > num else np.nan)
                return name
            new_expr=re.sub(task_expr_pattern, handle_col, expr)
            return pd.concat([df, task_expr_df], axis=1).eval(new_expr)

        return EventProcessing.compute_evdataframe(event_channels_df, config, copy_columns=["task_data", "task_node"], eval_func=task_eval)
    
    @processing_method(goal=EventProcessing.event_method)
    def output_accumulator_binary_wave(relevant: pd.DataFrame, config):
        res = []
        for _,grp in  relevant.groupby((relevant["state_value"] ==1).cumsum()):
            starts = grp[grp["state_value"]==1]
            if len(starts.index) ==0: continue
            elif len(starts.index) > 1:
                raise Exception(f"Problem several _starts{len(starts.index)}")
            else:
                start = starts["t"].iat[0]
                rises = [0] + (grp["t"].loc[grp["state_value"] > grp["state_value"].shift(1)] -start).to_list()
                duration_on = grp["duration_on_value"].iat[0]
                expected_count = grp["expected_count_value"].iat[0]
                duration = rises[-1] + duration_on
                metadata_join = grp["metadata"].to_list()
                if pd.isnull(grp["task_data"].iat[0]) or grp["task_data"].iat[0].startswith("on"):
                    metadata_join.append(dict(warnings=dict(free_event=f'task_data_at_event_is {grp["task_data"].iat[0]}')))
                metadata = compute_metadata(grp["metadata"], 
                    count_mismatch=dict(read=len(rises), expected=expected_count),
                    free_event=(pd.isnull(grp["task_data"].iat[0]) or not grp["task_data"].iat[0].startswith("on"), f'task_data_at_event_is {grp["task_data"].iat[0]}')
                )
                res.append(dict(t=start, metadata=metadata, duration=duration, n_segments=len(rises), 
                                waveform_changes=sorted(rises+[r + duration_on for r in rises[:-1]]),
                                waveform_values=[0]+[1, 0]*len(rises)))
        
        return pd.DataFrame(res).assign(event_name=config["event_name"])
    
    @processing_method(goal=EventProcessing.event_method)
    def output_binary_wave(relevant: pd.DataFrame, config):
        if relevant["state_value"].iat[0] == 0:
            relevant = relevant.iloc[1:, :].copy()
        else:
            relevant= relevant.copy()
        relevant["next_metadata"] = relevant["metadata"].shift(-1)
        relevant["count_value"] = relevant["count_value"].shift(-1).replace(0, 1)
        rises = relevant.iloc[::2].copy()

        rises["read_duration"] = rises["duration"]
        rises["duration"] = np.where(
            (rises["duration"] <= 0.001) & rises["duration_on_value"].notna(), 
            rises["duration_on_value"], rises["duration"])
        rises["duration_on_value"] = rises["duration_on_value"].fillna(rises["duration"])
        rises["cycle_duration"] = rises["duration_on_value"] + rises["duration_off_value"]
        rises["rises"] = rises.apply(lambda row: [i * row["cycle_duration"] for i in range(int(row["count_value"]))] if pd.notna(row["count_value"]) else [0], axis=1)

        rises["metadata"] = rises.apply(lambda row: compute_metadata(
            [row["metadata"], row["next_metadata"]], 
            count_mismatch=dict(read=row["count_value"], expected=row["expected_count_value"]),
            duration_correction=dict(read=row["read_duration"], corrected=row["duration"]),
            free_event=(pd.isnull(row["task_data"]) or not row["task_data"].startswith("on"), f'task_data_at_event_is {row["task_data"]}')
        ), axis=1)
        
        
        rises["waveform_info"] = rises.apply(lambda row: 
            dict(type="binary", rises=row["rises"], durations=[row["duration_on_value"]]* len(row["rises"])), axis=1)
        
        return rises[["t","duration", "metadata"]].assign(n_segments=rises["count_value"], event_name = config["event_name"],
               waveform_changes=rises.apply(lambda row: sorted(row["rises"]+[t+row["duration_on_value"] for t in row["rises"]]), axis=1),
               waveform_values=rises["count_value"].apply(lambda c: [0] + [1, 0] * int(c))
        )