
import json


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


def table_row_to_text(header, row):
    '''
    use templates to convert table row to text
    '''
    res = ""
    
    if header[0]:
        res += (header[0] + " ")

    for head, cell in zip(header[1:], row[1:]):
        res += ("the " + row[0] + " of " + head + " is " + cell + " ; ")
    
    res = remove_space(res)
    return res.strip()

def remove_space(text_in):
    res = []

    for tmp in text_in.split(" "):
        if tmp != "":
            res.append(tmp)

    return " ".join(res)


def parse_convfinqa_data(datapath, general_table=False, evaluation=False):
    print("reading data from %s" % datapath)
    with open(datapath, "r") as f:
        data_list = json.load(f)

    avg_doc_len = 0
    data_for_fqa_list = []
    for item in data_list:
        pre_text_list = item["pre_text"]
        post_text_list = item["post_text"]

        ## get text before and after Table
        pre_text_list = [text.strip() for text in pre_text_list if len(text.split()) > 3]
        post_text_list = [text.strip() for text in post_text_list if len(text.split()) > 3]

        query_list = item['annotation']['dialogue_break']
        answer_list = item['annotation']['turn_program']
        exe_ans_list = item['annotation']['exe_ans_list']
        answer_list_for_evaluation = []

        assert len(query_list) == len(answer_list)

        ## get question_list and answer_list
        qa_history = []
        question_list = []
        for query, answer in zip(query_list, answer_list):
            if evaluation and ("add" in answer or "subtract" in answer or "divide" in answer or "multiply" in answer):
                ## convert add(10,20) to 10 + 20
                arithmetic_list = []
                split_items = answer.split("), ")

                for item2 in split_items:
                    item2 = item2.replace("(", "").replace(")", "")
                    if "add" in item2:
                        item2 = item2.replace("add", "")
                        arithmetic_string = " + "
                    elif "subtract" in item2:
                        item2 = item2.replace("subtract", "")
                        arithmetic_string = " - "
                    elif "divide" in item2:
                        item2 = item2.replace("divide", "")
                        arithmetic_string = " / "
                    else:
                        item2 = item2.replace("multiply", "")
                        arithmetic_string = " * "

                    digits = item2.split(", ")
                    digit0, digit1 = digits[0], digits[1]
                    if digit0[0] == "#" and digit1[0] != "#":
                        idx0 = int(digit0.replace("#", ""))
                        digit0 = arithmetic_list[idx0]
                        digit1 = digit1.replace("const_", "")
                        if arithmetic_string == " + " or arithmetic_string == " - ":
                            arithmetic_list.append(digit0 + arithmetic_string + digit1)
                        else:
                            arithmetic_list.append("(" + digit0 + ")" + arithmetic_string + digit1)
                    elif digit0[0] != "#" and digit1[0] == "#":
                        idx1 = int(digit1.replace("#", ""))
                        digit1 = arithmetic_list[idx1]
                        digit0 = digit0.replace("const_", "")
                        if arithmetic_string == " + " or arithmetic_string == " - ":
                            arithmetic_list.append(digit0 + arithmetic_string + digit1)
                        else:
                            arithmetic_list.append(digit0 + arithmetic_string + "(" + digit1 + ")")
                    elif digit0[0] == "#" and digit1[0] == "#":
                        idx0 = int(digit0.replace("#", ""))
                        idx1 = int(digit1.replace("#", ""))
                        digit0, digit1 = arithmetic_list[idx0], arithmetic_list[idx1]
                        if arithmetic_string == " + " or arithmetic_string == " - ":
                            arithmetic_list.append(digit0 + arithmetic_string + digit1)
                        else:
                            arithmetic_list.append("(" + digit0 + ")" + arithmetic_string + "(" + digit1 + ")")
                    else:
                        digit0 = digit0.replace("const_", "")
                        digit1 = digit1.replace("const_", "")
                        arithmetic_list.append(digit0 + arithmetic_string + digit1)

                answer = arithmetic_list[-1]

                # answer_reformat = arithmetic_list[-1]
                # print("="*80)
                # print(answer)
                # print(arithmetic_list)
                # print(answer_reformat)
                # answer = answer_reformat

            qa_history.append(("user", query))
            question = format_qa_history_to_question(qa_history)
            question_list.append(question)

            qa_history.append(("agent", answer))
            answer_list_for_evaluation.append(answer)
        
        if evaluation:
            answer_list = answer_list_for_evaluation

        assert len(question_list) == len(answer_list) == len(exe_ans_list)

        ## process Table
        table = item['table']
        table_text_list = []
        assert len(table) >= 2
        if general_table:
            ## a general way for processing Table
            table_text_list.append("<<Table>>")
            first_row = [column[0] for column in table]
            # print("first_row:", first_row)
            num_row = len(table[0])
            for i in range(1, num_row):
                row_text = ""
                for j, column in enumerate(table):
                    if j == 0:
                        if first_row[0] != "":
                            row_text += "| " + column[i] + "(" + first_row[0] + ") | "
                        else:
                            row_text += "| " + column[i] + " | "
                    else:
                        if first_row[j] != "":
                            row_text += first_row[j] + ": " + column[i] + " | "
                        else:
                            row_text += column[i] + " | "
                
                row_text = row_text.strip()
                table_text_list.append(row_text)
            table_text_list.append("<</Table>>")
        else:
            ## get the text from Table based on row to text
            header = table[0]
            for row in table[1:]:
                table_text = table_row_to_text(header, row)
                # print(table_text)
                table_text_list.append(table_text)
            
        chunk_list = pre_text_list + table_text_list + post_text_list
        document = "\n".join(chunk_list)
        avg_doc_len += len(document.split())
        
        for question, answer, exe_ans in zip(question_list, answer_list, exe_ans_list):
            data_item = {
                "ctxs": [{"title": "", "text": document}],
                "sub-paragraphs": document,
                "question": question,
                "answer": answer,
                "exe_answer": exe_ans
            }
            # print(answer)
            data_for_fqa_list.append(data_item)

    print("number of dialogs: %d" % len(data_list))
    print("number of turns: %d" % len(data_for_fqa_list))
    print("average document length: %.4f" % (avg_doc_len / len(data_list)))

    return data_for_fqa_list


def parse_finqa_data(datapath, general_table=False):
    print("reading data from %s" % datapath)
    with open(datapath, "r") as f:
        data_list = json.load(f)

    num_skip_table = 0
    avg_doc_len = 0
    data_for_fqa_list = []
    for item in data_list:
        pre_text_list = item["pre_text"]
        post_text_list = item["post_text"]

        ## get text before and after Table
        pre_text_list = [text.strip() for text in pre_text_list if len(text.split()) > 3]
        post_text_list = [text.strip() for text in post_text_list if len(text.split()) > 3]

        ## get question and answer
        question = item['qa']['question']
        answer = item['qa']['program']
        exe_answer = item['qa']['exe_ans']

        ## get table
        table = item['table']
        table_text_list = []

        # assert len(table) >= 2
        if len(table) >= 2:
            if general_table:
                ## a general way for processing Table
                table_text_list.append("<<Table>>")
                first_row = [column[0] for column in table]
                num_row = len(table[0])
                for i in range(1, num_row):
                    row_text = ""
                    for j, column in enumerate(table):
                        if j == 0:
                            if first_row[0] != "":
                                row_text += "| " + column[i] + "(" + first_row[0] + ") | "
                            else:
                                row_text += "| " + column[i] + " | "
                        else:
                            if first_row[j] != "":
                                row_text += first_row[j] + ": " + column[i] + " | "
                            else:
                                row_text += column[i] + " | "
                    table_text_list.append(row_text)
                
                row_text = row_text.strip()
                table_text_list.append("<</Table>>")
            else:
                header = table[0]
                for row in table[1:]:
                    table_text = table_row_to_text(header, row)
                    # print(table_text)
                    table_text_list.append(table_text)
        
        chunk_list = pre_text_list + table_text_list + post_text_list
        document = "\n".join(chunk_list)
        avg_doc_len += len(document.split())

        data_item = {
            "ctxs": [{"title": "", "text": document}],
            "sub-paragraphs": document,
            "question": question,
            "answer": answer,
            "exe_answer": exe_answer
        }
        data_for_fqa_list.append(data_item)

    print("number of qa pairs: %d" % len(data_for_fqa_list))
    print("average document length: %.4f" % (avg_doc_len / len(data_list)))

    return data_for_fqa_list


def parse_tatqa_data(datapath, general_table=False, training=False, multispan=False):
    print("reading data from %s" % datapath)

    with open(datapath, "r") as f:
        data_list = json.load(f)
    
    count_complex_arithmetic = 0
    count_arithmetic = 0

    avg_doc_len = 0
    data_for_fqa_list = []
    for item in data_list:
        table = item['table']['table']
        paragraphs = item['paragraphs']
        qa_pair_list = item['questions']

        plain_text_list = []
        for paragraph in paragraphs:
            plain_text_list.append(paragraph['text'])

        table_text_list = []
        assert len(table) >= 2
        if general_table:
            ## a general way for processing Table
            table_text_list.append("<<Table>>")
            first_row = [column[0] for column in table]
            num_row = len(table[0])
            for i in range(1, num_row):
                row_text = ""
                for j, column in enumerate(table):
                    if j == 0:
                        if first_row[0] != "":
                            row_text += "| " + column[i] + "(" + first_row[0] + ") | "
                        else:
                            row_text += "| " + column[i] + " | "
                    else:
                        if first_row[j] != "":
                            row_text += first_row[j] + ": " + column[i] + " | "
                        else:
                            row_text += column[i] + " | "
                table_text_list.append(row_text)
            
            row_text = row_text.strip()
            table_text_list.append("<</Table>>")
            
        else:
            header = table[0]
            for row in table[1:]:
                table_text = table_row_to_text(header, row)
                # print(table_text)
                table_text_list.append(table_text)
        
        chunk_list = plain_text_list + table_text_list
        document = "\n".join(chunk_list)
        avg_doc_len += len(document.split())
        
        for qa_pair in qa_pair_list:
            if training and (qa_pair['answer_type'] == 'count' or qa_pair['answer_type'] == 'multi-span'):
                continue
            if training and type(qa_pair['answer']) == list and len(qa_pair['answer'][0].split()) > 1:
                continue
            
            if multispan and (qa_pair['answer_type'] == 'arithmetic' or (qa_pair['answer_type'] == "span" and len(qa_pair['answer'][0].split()) == 1)):
                continue

            question = qa_pair['question']
            answer_tmp = qa_pair['answer']
            # print("answer_tmp:", answer_tmp)
            if type(answer_tmp) == list:
                arithmetic = False
                assert qa_pair['answer_type'] != "arithmetic" and qa_pair['answer_type'] != "count"
                
                if len(answer_tmp) > 1:
                    answer_tmp = ['"' + item + '"' for item in answer_tmp]
                answer = ", ".join(answer_tmp)
                
            else:
                assert qa_pair['answer_type'] in ["arithmetic", "count"]
                derivation = qa_pair['derivation']
                assert derivation != ""
                if qa_pair['answer_type'] == "arithmetic":
                    arithmetic = True
                    count_arithmetic += 1

                    if training:
                        ## for training, take the derivation part (step by step)
                        answer = derivation.strip()
                        if answer.startswith("-"):
                            answer_tmp = list(answer)
                            answer_tmp[0] = "<first_minus>"
                            answer = "".join(answer_tmp)
                        answer = answer.replace("+-", "-").replace("/-", "/ -")
                        answer = answer.replace("(-", "<bracket_minus>").replace("/ -", "<divide_minus>")
                        answer = answer.replace("+", " + ").replace("-", " - ").replace("*", " * ").replace("/", " / ")
                        answer = " ".join(answer.split())
                        answer = answer.replace("<first_minus>", "-").replace("<bracket_minus>", "(-").replace("<divide_minus>", "/ -")
                        # print(answer)
                    else:
                        ## for evaluation, directly take the final answer, and compare the calculated results
                        answer = answer_tmp

                else:
                    arithmetic = False
                    count_list = derivation.split("##")
                    if len(count_list) > 1:
                        count_list = ['"' + item.strip() + '"' for item in count_list]
                    else:
                        count_list = [item.strip() for item in count_list]
                    # answer = answer_tmp + ". " + ", ".join(count_list)
                    answer = ", ".join(count_list)

            data_dict = {
                "arithmetic": arithmetic,
                "ctxs": [{"title": "", "text": document}],
                "sub-paragraphs": document,
                "question": question,
                "answer": answer
            }
            # print(answer)
            data_for_fqa_list.append(data_dict)
    
    print("length of data_for_fqa_list:", len(data_for_fqa_list))
    print("count_arithmetic:", count_arithmetic)
    print("number of document: %d; average document length: %.4f" % (len(data_list), avg_doc_len/len(data_list)))

    return data_for_fqa_list


def save_fqa_data_list(data_for_fqa_list, output_datapath):
    
    print("writing data_for_fqa_list to %s" % output_datapath)
    with open(output_datapath, "w") as f:
        json.dump(data_for_fqa_list, f, indent=2)


def main_convfinqa():
    train_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/convfinqav2/ConvFinQA/data/train.json"
    dev_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/convfinqav2/ConvFinQA/data/dev.json"

    data_for_fqa_train_list = parse_convfinqa_data(train_datapath, general_table=True)
    data_for_fqa_dev_list = parse_convfinqa_data(dev_datapath, general_table=True)
    
    output_traindatapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/convfinqav2/convfinqav2_QA_train.json"
    output_devdatapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/convfinqav2/convfinqav2_QA_dev.json"
    save_fqa_data_list(data_for_fqa_train_list, output_traindatapath)
    save_fqa_data_list(data_for_fqa_dev_list, output_devdatapath)

def main_convfinqav3():
    dev_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/convfinqav2/ConvFinQA/data/dev.json"

    data_for_fqa_dev_list = parse_convfinqa_data(dev_datapath, general_table=True, evaluation=True)
    output_devdatapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/convfinqav3/convfinqav3_QA_dev.json"
    save_fqa_data_list(data_for_fqa_dev_list, output_devdatapath)


def main_finqa():
    train_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/finqav2/FinQA/dataset/train.json"
    dev_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/finqav2/FinQA/dataset/test.json"

    data_for_fqa_train_list = parse_finqa_data(train_datapath, general_table=True)
    data_for_fqa_dev_list = parse_finqa_data(dev_datapath, general_table=True)

    output_traindatapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/finqav2/finqav2_QA_train.json"
    output_devdatapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/finqav2/finqav2_QA_dev.json"
    save_fqa_data_list(data_for_fqa_train_list, output_traindatapath)
    save_fqa_data_list(data_for_fqa_dev_list, output_devdatapath)


def main_tatqa():
    train_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/tatqav2/dataset_raw/tatqa_dataset_train.json"
    dev_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/tatqav2/dataset_raw/tatqa_dataset_dev.json"

    data_for_fqa_train_list = parse_tatqa_data(train_datapath, general_table=True, training=True)
    data_for_fqa_dev_list = parse_tatqa_data(dev_datapath, general_table=True, training=True)

    # data_for_fqa_dev_list = parse_tatqa_data(dev_datapath, general_table=True)
    # output_devdatapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/tatqav2/dev.json"

    output_traindatapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/tatqav3/tatqav3_QA_train.json"
    output_devdatapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/tatqav3/tatqav3_QA_dev.json"
    save_fqa_data_list(data_for_fqa_train_list, output_traindatapath)
    save_fqa_data_list(data_for_fqa_dev_list, output_devdatapath)


def main_tatqa_multispan():
    train_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/tatqav2/dataset_raw/tatqa_dataset_train.json"
    dev_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/tatqav2/dataset_raw/tatqa_dataset_dev.json"

    data_for_fqa_train_list = parse_tatqa_data(train_datapath, general_table=True, multispan=True)
    data_for_fqa_dev_list = parse_tatqa_data(dev_datapath, general_table=True, multispan=True)

    output_traindatapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/tatqamultispan/tatqamultispan_QA_train.json"
    output_devdatapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/tatqamultispan/tatqamultispan_QA_dev.json"
    save_fqa_data_list(data_for_fqa_train_list, output_traindatapath)
    save_fqa_data_list(data_for_fqa_dev_list, output_devdatapath)


if __name__ == "__main__":
    # main_convfinqa()
    # main_finqa()
    # main_tatqa()

    # main_convfinqav3()
    main_tatqa_multispan()
