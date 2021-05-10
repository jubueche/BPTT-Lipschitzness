from datajuicer.table import Table

def visualizer(dim=None):
    def _func(func, table_args, kwargs):
        table = Table(*table_args)
        return func(table, **kwargs)
    return lambda func: lambda grid, independent_keys, dependent_keys, label_dict={}, order=None, **kwargs: _func(func, [grid, independent_keys, dependent_keys, label_dict, dim, order], kwargs)

@visualizer(dim=4)
def latex(table, decimals=2):
    shape = table.shape()
    cols = "lc" + "c".join(["l"*shape[3]]*shape[1])
    string = r"\resizebox{\columnwidth}{!}{%" + "\n"
    string += r"\begin{tabular}{" + cols + "}\n"

    def format_value(val):
        if type(val) is float:
            return f"%.{decimals}f" % (val)
        if type(val) is str:
            return val.replace("_", r"\_")
        return str(val)

    for i0 in range(shape[0]):
        struts = r"\bigBstrut"
        if i0>0:
            struts += r"\bigTstrut"
        string += r"\multicolumn{" + str(len(cols)) + r"}{l}{\bfseries " \
            + format_value(table.get_label(axis=0, index=i0)) \
            + r"}" + struts + r"\\ \hline" + "\n"

        string += "".join([r"&& \multicolumn{" + str(shape[3]) +r"}{l}{" + format_value(table.get_label(axis=1, index=i)) + r"} " for i in range(shape[1])]) \
            + r"\Bstrut\Tstrut\\" \
            + "".join([r"\cline{" + f"{i*(shape[3]+1) +3}-{(i+1)*(shape[3]+1) + 1}" + "}" for i in range(shape[1])]) \
            + "\n"
        
        string += format_value(table.get_label(axis=2)) \
            + (" && " + " & ".join([format_value(table.get_label(axis=3, index=i)) for i in range(shape[3])]) ) * shape[1] \
            + r" \Tstrut\\" + "\n"
        
        for i2 in range(shape[2]):
            string += format_value(table.get_label(axis=2, index=i2)) \
                + "".join([" && "  + " & ".join([format_value(table.get_val(i0, i1, i2, i3)) for i3 in range(shape[3])]) for i1 in range(shape[1]) ]) \
                + r"\\" + "\n"

    string += r"\end{tabular}%" + "\n"
    string += "}"
    return string