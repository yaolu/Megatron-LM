
import os
import json


def remove_noanswer_case(input_datapath, output_datapath):
    print("reading from %s" % input_datapath)
    with open(input_datapath, "r") as f:
        data_list = json.load(f)

    count_noanswer = 0
    new_data_list = []
    for item in data_list:
        if item['answer'] == "Sorry. I cannot find the answer based on the context.":
            count_noanswer += 1
            continue
        new_data_list.append(item)
    
    print("count_noanswer:", count_noanswer)
    print("length of data_list:", len(data_list))
    print("length of new_data_list:", len(new_data_list))
    
    print("writing to %s" % output_datapath)
    with open(output_datapath, "w") as f:
        json.dump(new_data_list, f, indent=2)


def main():
    # remove no answer case from convqa
    data_folder = "/lustre/fsw/adlr/adlr-nlp/pengx/data/foundational_qa/s3_data"
    convqa_trainpath = os.path.join(data_folder, "convqa/convqa_QA_train.json")
    convqa_devpath = os.path.join(data_folder, "convqa/convqa_QA_dev.json")

    output_trainpath = os.path.join(data_folder, "convqav2/convqav2_QA_train.json")
    output_devpath = os.path.join(data_folder, "convqav2/convqav2_QA_dev.json")

    remove_noanswer_case(convqa_trainpath, output_trainpath)
    remove_noanswer_case(convqa_devpath, output_devpath)

    
if __name__ == "__main__":
    main()