
from tqdm import tqdm
import copy
import json
import csv
import os


def format_qa_history_to_question(qa_history, num_turn=7, all_turns=False):
    question = ""
    if not all_turns:
        qa_history = qa_history[-num_turn:]
    for item in qa_history:
        if item[0] == "user":
            question += "User: " + item[1] + "\n\n"
        else:
            assert item[0] == "agent"
            question += "Assistant: " + item[1] + "\n\n"

    question += "Assistant:"
    
    return question


def parse_fetaqa_data(datapath):
    data_list = []
    with open(datapath, "r") as f:
        for line in f:
            json_item = json.loads(line)
            data_list.append(json_item)

    avg_doc_len = 0
    data_for_fqa_list = []
    print("length of data samples:", len(data_list))
    for item in data_list:
        table_page_title = item['table_page_title']
        table_section_title = item['table_section_title']

        ## get title
        title = table_page_title + ", " + table_section_title

        ## get table text
        table = item['table_array']
        table_text_list = []
        table_text_list.append("<<Table>>")
        first_row = table[0]
        # print("first_row:", first_row)
        for row in table[1:]:
            row_text = ""
            for i, row_item in enumerate(row):
                if i == 0:
                    if first_row[0] != "":
                        row_text += "| " + row_item + "(" + first_row[0] + ") | "
                    else:
                        row_text += "| " + row_item + " | "
                else:
                    if first_row[i] != "":
                        row_text += first_row[i] + ": " + row_item + " | "
                    else:
                        row_text += row_item + " | "

            row_text = row_text.strip()
            table_text_list.append(row_text)
        table_text_list.append("<</Table>>")
        document = "\n".join(table_text_list)

        question = item['question']
        answer = item['answer']

        avg_doc_len += len(document.split())
        data_item = {
            "ctxs": [{"title": title, "text": document}],
            "sub-paragraphs": document,
            "question": question,
            "answer": answer
        }
        data_for_fqa_list.append(data_item)

    print("average document length: %.4f" % (avg_doc_len / len(data_for_fqa_list)))
    return data_for_fqa_list


def parse_wikitablequestions(datapath, datafolder):

    def _get_tsv_text(_datapath):
        data_list = []
        with open(_datapath, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                data_list.append(row)
        header = data_list[0]
        data_list = data_list[1:]
        return header, data_list

    header, data_list = _get_tsv_text(datapath)
    print("number of data samples:", len(data_list))
    
    avg_doc_len = 0
    data_for_fqa_list = []

    for item in data_list:
        question = item[1].strip()
        answer = item[3].strip()

        table_datafile = item[2].replace(".csv", ".tsv")
        table_datapath = os.path.join(datafolder, table_datafile)
        
        table_text_list = ["<<Table>>"]
        table_header, table_list = _get_tsv_text(table_datapath)
        for row in table_list:
            row_text = ""
            for i, row_item in enumerate(row):
                row_item = row_item.strip()
                if i == 0:
                    if table_header[0] != "":
                        row_text += "| " + row_item + "(" + table_header[0] + ") | "
                    else:
                        row_text += "| " + row_item + " | "
                else:
                    if table_header[i] != "":
                        row_text += table_header[i] + ": " + row_item + " | "
                    else:
                        row_text += row_item + " | "
            row_text = row_text.strip()
            table_text_list.append(row_text)

        table_text_list.append("<</Table>>")
        document = "\n".join(table_text_list)

        avg_doc_len += len(document.split())
        data_item = {
            "ctxs": [{"title": "", "text": document}],
            "sub-paragraphs": document,
            "question": question,
            "answer": answer
        }
        data_for_fqa_list.append(data_item)

    print("average document length: %.4f" % (avg_doc_len / len(data_for_fqa_list)))
    return data_for_fqa_list


def parse_sqa_dataset(datapath, datafolder):
    def _get_tsv_csv_text(_datapath):
        data_list = []
        with open(_datapath, "r") as f:
            if _datapath.endswith(".tsv"):
                reader = csv.reader(f, delimiter="\t")
            else:
                reader = csv.reader(f, delimiter=",")
            for row in reader:
                data_list.append(row)
        header = data_list[0]
        data_list = data_list[1:]
        return header, data_list
    
    header, data_list = _get_tsv_csv_text(datapath)
    # print(header)
    avg_doc_len = 0
    data_for_fqa_list = []
    for item in data_list:
        position = item[2]
        if position == "0":
            dialog_history = []
        table_datafile = item[4]
        table_datapath = os.path.join(datafolder, table_datafile)
        table_header, table_list = _get_tsv_csv_text(table_datapath)

        table_text_list = ["<<Table>>"]
        for row in table_list:
            row_text = ""
            for i, row_item in enumerate(row):
                row_item = row_item.strip()
                if i == 0:
                    if table_header[0] != "":
                        row_text += "| " + row_item + "(" + table_header[0] + ") | "
                    else:
                        row_text += "| " + row_item + " | "
                else:
                    if table_header[i] != "":
                        row_text += table_header[i] + ": " + row_item + " | "
                    else:
                        row_text += row_item + " | "
            row_text = row_text.strip()
            table_text_list.append(row_text)
        table_text_list.append("<</Table>>")
        document = "\n".join(table_text_list)
        avg_doc_len += len(document.split())

        question = item[3]
        answer = item[-1]
        if answer.startswith("["):
            answer = answer[1:]
        if answer.endswith("]"):
            answer = answer[:-1]
        # print(answer)
        dialog_history.append(("user", question))
        formatted_question = format_qa_history_to_question(dialog_history)
        data_dict = {
            "ctxs": [{"title": "", "text": document}],
            "sub-paragraphs": document,
            "question": formatted_question,
            "answer": answer
        }
        dialog_history.append(("agent", answer))
        
        data_for_fqa_list.append(data_dict)

    avg_doc_len /= len(data_for_fqa_list)
    print("average document length: %.4f" % avg_doc_len)
    print("number of samples:", len(data_for_fqa_list))
    return data_for_fqa_list


def parse_hybridqa_dataset(datapath, datafolder):
    print("reading data from %s" % datapath)
    with open(datapath, "r") as f:
        data_list = json.load(f)

    avg_doc_len = 0
    data_for_fqa_list = []
    for item in tqdm(data_list):
        question = item['question']
        answer = item['answer-text']

        table_datafile = item['table_id']
        answer_nodes = item['answer-node']
        answer_nodes_passage_ids = [node[2] for node in answer_nodes if node[2] is not None]

        passage_datapath = os.path.join(datafolder, "request_tok/%s.json" % table_datafile)
        table_datapath = os.path.join(datafolder, "tables_tok/%s.json" % table_datafile)
       
        with open(passage_datapath, "r") as f:
            passage_data_dict = json.load(f)
        with open(table_datapath, "r") as f:
            table_data_dict = json.load(f)

        passage_list = []
        for passage_id in answer_nodes_passage_ids:
            passage = passage_data_dict[passage_id]
            passage_list.append(passage)

        intro = table_data_dict['intro']
        title = table_data_dict['title']
        table_header = [item[0] for item in table_data_dict['header']]
        table_list = table_data_dict['data']
        
        table_text_list = ["<<Table>>"]
        for row in table_list:
            row_text = ""
            for i, row_item in enumerate(row):
                row_item = row_item[0].strip()
                if i == 0:
                    if table_header[0] != "":
                        row_text += "| " + row_item + "(" + table_header[0] + ") | "
                    else:
                        row_text += "| " + row_item + " | "
                else:
                    if table_header[i] != "":
                        row_text += table_header[i] + ": " + row_item + " | "
                    else:
                        row_text += row_item + " | "
            row_text = row_text.strip()
            table_text_list.append(row_text)
        table_text_list.append("<</Table>>")
        
        context_text_list = [intro] + table_text_list + passage_list
        document = "\n".join(context_text_list)

        avg_doc_len += len(document.split())
        data_item = {
            "ctxs": [{"title": title, "text": document}],
            "sub-paragraphs": document,
            "question": question,
            "answer": answer
        }
        data_for_fqa_list.append(data_item)

    avg_doc_len /= len(data_for_fqa_list)
    print("average document length: %.4f" % avg_doc_len)
    print("number of samples:", len(data_for_fqa_list))

    return data_for_fqa_list


def parse_hybriddial(datapath, datafolder):
    with open(datapath, "r") as f:
        data_list = json.load(f)
    
    all_conversation_keys = list(data_list['conversations'].keys())
    print("all_conversation_keys length:", len(all_conversation_keys))
    test_startidx = int(0.95 * len(all_conversation_keys))
    test_conv_keys = {}
    for key in all_conversation_keys[test_startidx:]:
        test_conv_keys[key] = True
    print("test_conv_keys size:", len(test_conv_keys))
    
    all_candidates = data_list['all_candidates']

    all_turns = data_list['qas']
    qa_history_list = []
    answer_list = []
    qa_history = []
    cands_ids = []
    output_datalist = []
    num_dial = 0
    avg_num_table = 0
    avg_num_table_tok = 0
    avg_num_passage_tok = 0    
    ## add random one at the end to make sure to add all samples
    all_turns["random_q_id"] = {"conversation_id": "random_conv_id"}

    prev_id = None
    # print("test_conv_keys:", test_conv_keys)
    for q_id, turn_dict in all_turns.items():
        conv_id = turn_dict['conversation_id']
        if prev_id is not None and prev_id != conv_id:
            if len(qa_history_list) > 0:
                num_dial += 1
                assert len(qa_history_list) == len(answer_list)
                table_list = []
                for cand in cands_ids:
                    if all_candidates[cand]['table_key']:
                        table_list.append(all_candidates[cand]['table_key'])
                table_list = list(set(table_list))
                avg_num_table += len(table_list)
                table_text_all = []
                passage_text_all = []
                for table in table_list:
                    table_datapath = os.path.join(datafolder, "traindev_tables_tok/%s.json" % table)
                    passage_datapath = os.path.join(datafolder, "traindev_request_tok/%s.json" % table)
                    
                    with open(table_datapath, "r") as f:
                        table_data_dict = json.load(f)
                    with open(passage_datapath, "r") as f:
                        passage_data_dict = json.load(f)

                    intro = table_data_dict['intro']
                    title = table_data_dict['title']
                    table_header = [item[0] for item in table_data_dict['header']]
                    table_list = table_data_dict['data']
                    table_text_list = ["<<Table>>"]
                    for row in table_list:
                        row_text = ""
                        for i, row_item in enumerate(row):
                            row_item = row_item[0].strip()
                            if i == 0:
                                if table_header[0] != "":
                                    row_text += "| " + row_item + "(" + table_header[0] + ") | "
                                else:
                                    row_text += "| " + row_item + " | "
                            else:
                                if table_header[i] != "":
                                    row_text += table_header[i] + ": " + row_item + " | "
                                else:
                                    row_text += row_item + " | "
                        row_text = row_text.strip()
                        table_text_list.append(row_text)
                    table_text_list.append("<</Table>>")
                    table_text = intro + "\n" + "\n".join(table_text_list)
                    table_text_all.append(table_text)
                    passage_text_all.append(passage_data_dict)

                    avg_num_table_tok += len(table_text.split())
                    avg_num_passage_tok += len("\n".join(passage_data_dict.values()).split())

                for qa_history, answers in zip(qa_history_list, answer_list):
                    question = format_qa_history_to_question(qa_history)
                    data_dict = {
                        "question": question,
                        "answers": answers,
                        "table_text": table_text_all,
                        "passage_text": passage_text_all
                    }
                    output_datalist.append(data_dict)
        
            qa_history = []
            cands_ids = []
            qa_history_list = []
            answer_list = []
            
        if conv_id not in test_conv_keys:
            qa_history = []
            cands_ids = []
            qa_history_list = []
            answer_list = []
            continue

        qa_history.append(("user", turn_dict['current_query']))
        long_answer = turn_dict['long_response_to_query']
        short_answer = turn_dict['short_response_to_query']
        # assert turn_dict['correct_next_cands_ids'][0] in turn_dict['possible_next_cands_ids']
        cands_ids.append(turn_dict['correct_next_cands_ids'][0])

        qa_history_list.append(copy.deepcopy(qa_history))
        qa_history.append(("agent", long_answer))
        answer_list.append([long_answer, short_answer])
        prev_id = conv_id

    print("number of dialogs:", num_dial)
    print("number of turns:", len(output_datalist))

    avg_num_table = avg_num_table / num_dial
    avg_num_table_tok = avg_num_table_tok / num_dial
    avg_num_passage_tok = avg_num_passage_tok / num_dial
    print("avg_num_table per dialog:", avg_num_table)
    print("avg_num_table_tok per dialog:", avg_num_table_tok)
    print("avg_num_passage_tok per dialog:", avg_num_passage_tok)


    return output_datalist


def save_fqa_data_list(data_for_fqa_list, output_datapath):
    
    print("writing data_for_fqa_list to %s" % output_datapath)
    with open(output_datapath, "w") as f:
        json.dump(data_for_fqa_list, f, indent=2)


def main_hybridqa():
    hybridqa_train = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/HybridQA/HybridQA/released_data/train.traced.json"
    hybridqa_dev = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/HybridQA/HybridQA/released_data/dev.traced.json"
    hybridqa_datafolder = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/HybridQA/HybridQA/WikiTables-WithLinks"

    data_for_fqa_list_train = parse_hybridqa_dataset(hybridqa_train, hybridqa_datafolder)
    data_for_fqa_list_dev = parse_hybridqa_dataset(hybridqa_dev, hybridqa_datafolder)
    
    output_datapath_train = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/HybridQA/HybridQA_QA_train.json"
    output_datapath_dev = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/HybridQA/HybridQA_QA_dev.json"

    save_fqa_data_list(data_for_fqa_list_train, output_datapath_train)
    save_fqa_data_list(data_for_fqa_list_dev, output_datapath_dev)


def main_wikitablequsetions():
    datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/WikiTableQuestions/WikiTableQuestions/data/pristine-unseen-tables.tsv"
    datafolder = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/WikiTableQuestions/WikiTableQuestions"

    data_for_fqa_list = parse_wikitablequestions(datapath, datafolder)

    output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/WikiTableQuestions/WikiTableQuestions_QA_test.json"
    save_fqa_data_list(data_for_fqa_list, output_datapath)


def main_fetaqa():
    fetaqa_dev = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/fetaqa/FeTaQA/data/fetaQA-v1_dev.jsonl"

    data_for_fqa_dev_list = parse_fetaqa_data(fetaqa_dev)

    output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/fetaqa/fetaqa_QA_dev.json"
    save_fqa_data_list(data_for_fqa_dev_list, output_datapath)


def main_sqa():
    datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/sqa/SQA_Release_1.0/test.tsv"
    datafolder = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/sqa/SQA_Release_1.0"
    
    data_for_fqa_test_list = parse_sqa_dataset(datapath, datafolder)
    output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/sqa/sqa_QA_test.json"
    save_fqa_data_list(data_for_fqa_test_list, output_datapath)


def main_hybriddial():
    datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/HybridDial/experimental_data_indent.json"
    datafolder = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/ott_qa/OTT-QA/data"

    output_datalist = parse_hybriddial(datapath, datafolder)
    output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/HybridDial/test_all.json"
    save_fqa_data_list(output_datalist, output_datapath)


if __name__ == "__main__":
    # main_hybridqa()
    # main_fetaqa()
    # main_wikitablequsetions()
    # main_sqa()
    main_hybriddial()
