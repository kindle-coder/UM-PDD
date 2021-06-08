def extract_arg_value(arg, key):
    if arg.lower().__contains__(key.lower()):
        param = arg[arg.index("=") + 1:]
        return param
