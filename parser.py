def parse_model_names(model_list_file):
    model_names = []
    with open(model_list_file) as f:
        for line in f:
            if line.strip() == "": continue
            model_names.append(line.strip())
    return model_names