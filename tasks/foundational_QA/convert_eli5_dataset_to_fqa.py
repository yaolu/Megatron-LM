import json


def read_nemo_data(input_filename):

    data = []
    with open(input_filename, "r") as f:
        for line in f.readlines():
            item = json.loads(line)
            data.append(item)

    return data


def to_fqa_data(input_text, output_text, paragraph_id="", sub_paragraphs="", ctxs=[]):

    ctxs = [{"id": "", "title": "", "text": x} for x in ctxs]
    output_text = [x["answer"] for x in output_text if "answer" in x]

    item = {"paragraph_id": paragraph_id, "question": input_text, "answer": output_text,
            "sub-paragraphs": sub_paragraphs, "word count": "", "Date": "", "ctxs": ctxs}

    return item


def convert_nemo_to_fqa(data):

    fqa_data = []
    for item in data:
        try:
            item = to_fqa_data(item["input"], item["output"], ctxs=item["neighbors"], paragraph_id=item["id"])
        except:
            print(item)
        fqa_data.append(item)

    return fqa_data


def write_fqa_data(fqa_data, output_filename):

    with open(output_filename, "w") as f:
        json.dump(fqa_data,f, indent=2)

def split_train_dev(fqa_data):

    total = len(fqa_data)
    num_train = int(total)
    train_data = fqa_data[:]
    dev_data = fqa_data[:]

    return train_data, dev_data

if __name__ == "__main__":

    input_filename = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/ELI5-oracle/eli5-dev-kilt-with-neighbours-and-multiple-answers-oracle.jsonl"
    data = read_nemo_data(input_filename)
    fqa_data = convert_nemo_to_fqa(data)
    train_data, dev_data = split_train_dev(fqa_data)
    # output_filename = "/lustre/fsw/adlr/adlr-nlp/pengx/data/foundational_qa/s3_data/quiet_cockatoo/quiet_cockatoo_QA_{}.json"
    # output_filename = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/quiet-cockatoo_commercial/quiet_cockatoo_QA_{}.json"
    output_filename = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/ELI5-oracle/{}.json"
    write_fqa_data(train_data, output_filename.format("train"))
    write_fqa_data(dev_data, output_filename.format("dev"))
