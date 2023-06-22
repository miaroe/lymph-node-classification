import json
def parse_json(filepath):
    with open(filepath) as json_file:
        try:
            file = json.load(json_file)
            return file
        except json.decoder.JSONDecodeError:
            raise Exception(filepath + ": is not a valid JSON file")