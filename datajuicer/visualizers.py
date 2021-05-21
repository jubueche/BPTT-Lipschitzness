from datajuicer.table import Table

METHOD_COLORS = {"Standard":"#4c84e6", "AWP":"#fc033d", "ABCD":"#03fc35", "ESGD":"#aaa2b8", "Beta":"#9b32a8", "Forward + Beta":"#000000", "Beta + Forward":"#000000", "Forward Noise":"#ed64ed", "Dropout":"#32a83a"}
METHOD_LINESTYLE = {"Standard":"solid", "AWP":"solid", "ABCD":"solid", "ESGD":"solid", "Beta":"dashed", "Forward + Beta":"solid", "Beta + Forward":"solid", "Forward Noise":"solid", "Dropout":"solid"}
default_lw = 1.0
ours_lw = 3.0
METHOD_LINEWIDTH = {"Standard":default_lw, "AWP":default_lw, "ABCD":default_lw, "ESGD":default_lw, "Beta":ours_lw, "Forward + Beta":ours_lw, "Beta + Forward":ours_lw, "Forward Noise":default_lw, "Dropout":default_lw}

def visualizer(dim=None):
    def _func(func, table_args, kwargs):
        table = Table(*table_args)
        return func(table, **kwargs)
    return lambda func: lambda grid, independent_keys, dependent_keys, label_dict={}, order=None, **kwargs: _func(func, [grid, independent_keys, dependent_keys, label_dict, dim, order], kwargs)

@visualizer(dim=4)
def latex(table, decimals=2, bold_order=None):
    shape = table.shape()
    cols = "lc" + "c".join(["l"*shape[3]]*len(list(set([table.get_label(1,idx) for idx in range(shape[1])]))))
    string = r"\resizebox{\columnwidth}{!}{%" + "\n"
    string += r"\begin{tabular}{" + cols + "}\n"
    string += r"\toprule" + "\n"

    def format_value(val):
        if type(val) is float:
            return f"%.{decimals}f" % (val)
        elif type(val) is str:
            return val.replace("_", r"\_")
        return str(val)

    for i0 in range(shape[0]):
        struts = r""
        if i0 > 0:
            string += r"\midrule" + " \n"
        string += r"\multicolumn{" + str(len(cols)) + r"}{l}{\bfseries " \
            + format_value(table.get_label(axis=0, index=i0)) \
            + r"}" + struts + r"\\ \midrule" + "\n"

        relevant = set([])

        for i1 in range(shape[1]):
            for i2 in range(shape[2]):
                for i3 in range(shape[3]):
                    if not format_value(table.get_val(i0, i1, i2, i3)) == 'None':
                        relevant.add(i1)
        
        # relevant = sorted(list(relevant))
        # diff = shape[1] - len(relevant)
        diff = 0
        padding = " & " * (diff * shape[3] + diff - 1)

        string += "".join([r"&& \multicolumn{" + str(shape[3]) +r"}{l}{" + format_value(table.get_label(axis=1, index=i)) + r"} " for i in relevant]) \
            + padding + r"\\" \
            + "".join([r"\cmidrule(r){" + f"{i*(shape[3]+1) +3}-{(i+1)*(shape[3]+1) + 1}" + "}" for i in range(len(relevant))]) \
            + "\n"
        
        if shape[3] > 1:
            string += format_value(table.get_label(axis=2)) \
                + (" && " + " & ".join([format_value(table.get_label(axis=3, index=i)) for i in range(shape[3])]) ) * len(relevant) \
                + padding + r" \\" + "\n"
        
        for i2 in range(shape[2]):
            vals = [[format_value(table.get_val(i0, i1, i2, i3)) for i3 in range(shape[3])] for i1 in relevant]
            vals = [[vv if not vv=='None' else '-1' for vv in v] for v in vals]
            if not all([all([val =='None' for val in v]) for v in vals]):
                if bold_order is None:
                    val_bold = [max([float(v[i]) for v in vals]) for i in range(len(vals[0]))]
                else:
                    val_bold = [bold_order[i]([float(v[i]) for v in vals]) for i in range(len(vals[0]))]
                bold_vals = [[f"{v}" if float(v) != val_bold[idx] else r"\bf{"+str(v)+r"}" for idx,v in enumerate(vv)] for vv in vals]
                string += format_value(table.get_label(axis=2, index=i2)) \
                    + "".join([" && "  + " & ".join(v) for v in bold_vals]) \
                    + padding + r"\\" + "\n"

    string += r"\bottomrule" + "\n"
    string += r"\end{tabular}%" + "\n"
    string += "}"
    return string