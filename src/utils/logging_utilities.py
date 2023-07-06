def create_markdown_from_dict(log_dict):
    md_full_str = []
    for key, val in log_dict.items():
        md_sub_str = """## {}\n""".format(key.title())
        if isinstance(val, dict):
            md_sub_str += create_markdown_parameter_table(val, level=0)
        md_full_str.append(md_sub_str)
    return "".join(md_full_str)


def create_markdown_parameter_table(values, header=None, include_table_header=True, level=0):
    level += 1

    md_table_str = ""
    if header:
        md_table_str += header

    if include_table_header:
        md_table_str += "|Parameter|Value|\n"
        md_table_str += "|---|---|\n"

    for sub_key, sub_val in values.items():
        if isinstance(sub_val, dict):
            md_table_str += "\n{} {}\n".format("#"*(2+level), sub_key)
            md_table_str += create_markdown_parameter_table(sub_val, level=level)
        else:
            md_table_str += "| {} | {} |\n".format(sub_key, str(sub_val))
    return md_table_str
