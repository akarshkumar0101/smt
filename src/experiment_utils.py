import copy
import dataclasses

def dataclass_to_flat_dict(args):
    """
    Converts a tyro dataclass to a flat dictionary.
    """
    a = {}
    for k, v in vars(args).items():
        if v is None or isinstance(v, int) or isinstance(v, float) or isinstance(v, str) or isinstance(v, bool) or isinstance(v, list) or isinstance(v, tuple):
            a[k] = v
        else:
            assert dataclasses.is_dataclass(v), "v is not a dataclass"
            aa = dataclass_to_flat_dict(v)
            for kk, vv in aa.items():
                a[f"{k}.{kk}"] = vv
    return a

def align_configs(cfgs, default_cfg, prune=True):
    """
    Makes sure all cfgs have the default keys.
    Prunes away keys where all cfgs have the default val.
    """
    cfgs = copy.deepcopy(cfgs)
    for k in default_cfg.keys():  # make sure all cfgs have the default keys
        for cfg in cfgs:
            if k not in cfg:
                cfg[k] = default_cfg[k]
    # assert all(c.keys() == default_config.keys() for c in configs)
    if prune:  # prune away keys where all cfgs have the default val
        for k in default_cfg.keys():
            if all([cfg[k] == default_cfg[k] for cfg in cfgs]):
                for cfg in cfgs:
                    del cfg[k]
    return cfgs


def _create_arg_list(cfg):
    def format_value(v):
        return f'"{v}"' if isinstance(v, str) else str(v)

    arg_list = []
    for key, val in cfg.items():
        if isinstance(val, list) or isinstance(val, tuple):
            arg_list.append(f'--{key} {" ".join([format_value(v) for v in val])}')
        else:
            arg_list.append(f"--{key}={format_value(val)}")
    return arg_list


def _create_commands_from_arg_lists(arg_lists, prefix=None):
    n_coms, n_args = len(arg_lists), len(arg_lists[0])
    arg_lens = [max([len(arg_lists[i_com][i_arg]) for i_com in range(n_coms)]) for i_arg in range(n_args)]
    commands = [" ".join([arg_lists[i_com][i_arg].ljust(arg_lens[i_arg]) for i_arg in range(n_args)]) for i_com in
                range(n_coms)]
    if prefix is not None:
        commands = [f"{prefix} {com}" for com in commands]
    return commands


def create_commands(cfgs, prefix=None, out_file=None):
    if dataclasses.is_dataclass(cfgs[0]):
        cfgs = [dataclass_to_flat_dict(cfg) for cfg in cfgs]
    
    assert all([set(cfg.keys()) == set(cfgs[0].keys()) for cfg in cfgs])  # make sure all cfgs have the same keys
    arg_lists = [_create_arg_list(config) for config in cfgs]
    commands = _create_commands_from_arg_lists(arg_lists, prefix=prefix)
    if out_file is not None:
        with open(out_file, "w") as f:
            f.write("\n".join(commands) + "\n")
    return commands