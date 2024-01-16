
import json


def parse_llmware_data(llmware_testpath):
    
    output_list = []
    output_list_unanswerable = []
    table_sample_list = [165, 166, 167, 168, 169]
    with open(llmware_testpath, "r") as f:
        for i, line in enumerate(f):
            data_dict = json.loads(line)
            
            question = data_dict['query']
            answer = data_dict['answer'].replace("\n", " ")
            context = data_dict['context']
            category = data_dict['category']
            sample_number = data_dict['sample_number']

            if sample_number in table_sample_list:
                ## convert table format
                table_lines = ["<<Table>>"]
                rows = context.split("\n")
                table_header = [item for item in rows[0].split("\t")]
                for row in rows[1:-1]:
                    row_items = row.split("\t")
                    row_text = ""
                    for i, item in enumerate(row_items):
                        item = item.strip()
                        if i == 0:
                            if table_header[0] != "":
                                row_text += "| " + item + "(" + table_header[0] + ") | "
                            else:
                                row_text += "| " + item + " | "
                        else:
                            if table_header[i] != "":
                                row_text += table_header[i] + ": " + item + " | "
                            else:
                                row_text += item + " | "
                    row_text = row_text.strip()
                    table_lines.append(row_text)

                table_lines.append("<</Table>>")
                table_lines.append(rows[-1])
                
                context = "\n".join(table_lines)

            if category == "not_found_classification":
                answer = "Sorry. I cannot find the answer based on the context."

            item = {
                "sample_number": sample_number,
                "category": category,
                "question": question,
                "ctxs": [{"title": "", "text": context}],
                "sub-paragraphs": context,
                "answer": answer,
            }
            output_list.append(item)

            if category == "not_found_classification":
                output_list_unanswerable.append(item)

    return output_list, output_list_unanswerable


def save_fqa_data_list(data_for_fqa_list, output_datapath):
    
    print("writing data_for_fqa_list to %s" % output_datapath)
    with open(output_datapath, "w") as f:
        json.dump(data_for_fqa_list, f, indent=2)


def main_llmware():
    llmware_testpath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/llmware/rag_instruct_benchmark_tester.jsonl"
    output_list, output_list_unanswerable = parse_llmware_data(llmware_testpath)

    print("number of samples in output_list:", len(output_list))

    llmware_outputpath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/llmware/test.json"
    save_fqa_data_list(output_list, llmware_outputpath)

    llmware_outputpath_unanswerable = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/llmware/test_unanswerable.json"
    save_fqa_data_list(output_list_unanswerable, llmware_outputpath_unanswerable)


if __name__ == "__main__":
    main_llmware()
