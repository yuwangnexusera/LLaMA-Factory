import pandas

MODEL_CONFIG = pandas.read_json("./model_conf.json", orient="records").to_dict(orient="records")


def mapping_model_name_path(model_alias):
    for item in MODEL_CONFIG:
        if item["model_alias"] == model_alias:
            return item